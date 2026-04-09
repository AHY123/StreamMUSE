"""
Handles saving the performance to a MIDI file (tick-native via mido).

Notes are stored as integer-tick (note_on, note_off) pairs and written
directly as delta-tick MIDI messages. No seconds conversion, no float
rounding. The tempo meta-event is written once at t=0.
"""

import os

import mido


class MidiFileHandler:
    """
    Collects user and model notes during a session and saves them to a MIDI
    file using tick-native encoding (mido). Public API matches the previous
    pretty_midi-based implementation so callers do not need to change.
    """

    def __init__(self, tempo: float, ticks_per_beat: int):
        """
        Args:
            tempo (float): Performance tempo in BPM (written as a tempo meta).
            ticks_per_beat (int): App-side ticks per beat. Used directly as
                the MIDI file's ticks_per_beat so every app tick maps to
                exactly one MIDI tick.
        """
        self.tempo = float(tempo)
        self.ticks_per_beat = int(ticks_per_beat)

        # Each event: {"type": "note_on"|"note_off", "pitch": int,
        #              "tick": int, "velocity": int}
        self._user_events: list[dict] = []
        self._model_events: list[dict] = []

        # Track the most recent note_on tick per pitch for same-tick
        # min-1-tick enforcement at ingest.
        self._last_user_on_tick: dict[int, int] = {}
        self._last_model_on_tick: dict[int, int] = {}

        # Track currently-open notes (for finalize flush and retrigger close).
        # Maps pitch -> tick of the open note_on.
        self._open_user: dict[int, int] = {}
        self._open_model: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def _add_event(
        self,
        note_or_event: dict,
        events: list,
        last_on_tick: dict,
        open_notes: dict,
        default_velocity: int,
    ):
        """Append a note_on/note_off (or legacy duration note) to `events`."""
        if "tick" not in note_or_event or "pitch" not in note_or_event:
            return

        pitch = int(note_or_event["pitch"])
        tick = int(note_or_event["tick"])

        # Legacy duration-based note: expand to a (note_on, note_off) pair.
        if "duration" in note_or_event and note_or_event.get("duration") is not None:
            duration = int(note_or_event["duration"])
            if duration <= 0:
                return
            velocity = int(note_or_event.get("velocity", default_velocity))
            events.append(
                {"type": "note_on", "pitch": pitch, "tick": tick, "velocity": velocity}
            )
            events.append(
                {
                    "type": "note_off",
                    "pitch": pitch,
                    "tick": tick + max(1, duration),
                    "velocity": 0,
                }
            )
            return

        event_type = note_or_event.get("type")
        if event_type not in ("note_on", "note_off"):
            return

        if event_type == "note_on":
            # Retrigger close: if this pitch is already open, emit an
            # implicit note_off first at tick (floored to open_tick+1 so
            # the closing note has at least 1 tick of duration).
            if pitch in open_notes:
                open_tick = open_notes[pitch]
                off_tick = max(tick, open_tick + 1)
                events.append(
                    {
                        "type": "note_off",
                        "pitch": pitch,
                        "tick": off_tick,
                        "velocity": 0,
                    }
                )

            velocity = int(note_or_event.get("velocity", default_velocity))
            events.append(
                {
                    "type": "note_on",
                    "pitch": pitch,
                    "tick": tick,
                    "velocity": velocity,
                }
            )
            last_on_tick[pitch] = tick
            open_notes[pitch] = tick
            return

        # note_off: enforce min-1-tick duration if it lands on the same tick
        # as its matching note_on. Otherwise the mido writer would emit a
        # zero-length note that DAWs silently drop.
        if pitch not in open_notes:
            return
        open_tick = open_notes.pop(pitch)
        off_tick = max(tick, open_tick + 1)
        events.append(
            {
                "type": "note_off",
                "pitch": pitch,
                "tick": off_tick,
                "velocity": 0,
            }
        )

    def add_user_note(self, note: dict):
        self._add_event(
            note_or_event=note,
            events=self._user_events,
            last_on_tick=self._last_user_on_tick,
            open_notes=self._open_user,
            default_velocity=100,
        )

    def add_model_note(self, note: dict):
        self._add_event(
            note_or_event=note,
            events=self._model_events,
            last_on_tick=self._last_model_on_tick,
            open_notes=self._open_model,
            default_velocity=80,
        )

    # ------------------------------------------------------------------
    # Finalize / save
    # ------------------------------------------------------------------

    def finalize(self):
        """Close any still-open notes at the max observed tick (+1 if same)."""

        def flush(events: list, open_notes: dict):
            if not open_notes:
                return
            end_tick = max((e["tick"] for e in events), default=0)
            for pitch, open_tick in list(open_notes.items()):
                off_tick = max(end_tick, open_tick + 1)
                events.append(
                    {
                        "type": "note_off",
                        "pitch": int(pitch),
                        "tick": off_tick,
                        "velocity": 0,
                    }
                )
                open_notes.pop(pitch, None)

        flush(self._user_events, self._open_user)
        flush(self._model_events, self._open_model)

    @staticmethod
    def _sort_key(event: dict):
        # At the same tick, note_off before note_on so retriggers flow
        # through the MIDI stream correctly.
        return (int(event["tick"]), 0 if event["type"] == "note_off" else 1)

    def _build_track(
        self,
        events: list,
        program: int,
        name: str,
        include_tempo: bool,
    ) -> mido.MidiTrack:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        if include_tempo:
            track.append(
                mido.MetaMessage(
                    "set_tempo", tempo=mido.bpm2tempo(self.tempo), time=0
                )
            )
        track.append(
            mido.Message("program_change", program=program, time=0, channel=0)
        )

        sorted_events = sorted(events, key=self._sort_key)
        prev_tick = 0
        for ev in sorted_events:
            abs_tick = int(ev["tick"])
            delta = max(0, abs_tick - prev_tick)
            prev_tick = abs_tick
            track.append(
                mido.Message(
                    ev["type"],
                    note=int(ev["pitch"]),
                    velocity=int(ev.get("velocity", 0)),
                    time=delta,
                    channel=0,
                )
            )
        track.append(mido.MetaMessage("end_of_track", time=0))
        return track

    def save_to_midi(
        self, session_log_dir: str, log_file_prefix="performance", midi_file_name=None
    ):
        """Save collected notes to a tick-native MIDI file."""

        # DEBUG: stats before finalize
        print(f"  [DEBUG MIDI] Before finalize:")
        print(f"    User events recorded: {len(self._user_events)}")
        print(f"    Model events recorded: {len(self._model_events)}")
        print(f"    Open user notes: {len(self._open_user)}")
        print(f"    Open model notes: {len(self._open_model)}")

        if self._user_events:
            print(f"  [DEBUG MIDI] First 5 user events (sorted):")
            for i, ev in enumerate(sorted(self._user_events, key=self._sort_key)[:5]):
                print(
                    f"    [{i}] type={ev['type']} pitch={ev['pitch']} "
                    f"tick={ev['tick']} vel={ev.get('velocity', 0)}"
                )

        self.finalize()

        print(f"  [DEBUG MIDI] After finalize:")
        print(f"    User events total: {len(self._user_events)}")
        print(f"    Model events total: {len(self._model_events)}")

        if self._user_events:
            u_min = min(e["tick"] for e in self._user_events)
            u_max = max(e["tick"] for e in self._user_events)
            print(f"    User tick range: {u_min} - {u_max}")
        if self._model_events:
            m_min = min(e["tick"] for e in self._model_events)
            m_max = max(e["tick"] for e in self._model_events)
            print(f"    Model tick range: {m_min} - {m_max}")

        if not self._user_events and not self._model_events:
            print("\nNo notes were played, MIDI file will not be saved.")
            return

        mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
        # Track 0: user melody (tempo meta lives here).
        mid.tracks.append(
            self._build_track(
                self._user_events, program=0, name="Guitar", include_tempo=True
            )
        )
        # Track 1: model accompaniment.
        mid.tracks.append(
            self._build_track(
                self._model_events, program=0, name="Piano", include_tempo=False
            )
        )

        if midi_file_name is None:
            midi_file_name = f"{log_file_prefix}.mid"
            log_filepath = os.path.join(session_log_dir, midi_file_name)
        else:
            midi_file_name = f"{midi_file_name}.mid"
            log_filepath = f"{session_log_dir}/{midi_file_name}"

        try:
            mid.save(log_filepath)
            print(f"Performance MIDI file saved to: {log_filepath}")
        except IOError as e:
            print(f"\nError saving MIDI file: {e}")
