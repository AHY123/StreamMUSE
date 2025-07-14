from miditok import MusicTokenizer, TokenizerConfig

from miditok import Event
from symusic import Note, TimeSignature, Track, Score
from collections.abc import Mapping, Sequence
from miditok.classes import TokSequence
from miditok.constants import MIDI_INSTRUMENTS, TIME_SIGNATURE

from miditok.utils import compute_ticks_per_bar
import numpy as np
from .config import FnsTokenizerConfig


class FnsTokenizer(MusicTokenizer):
    """
    FnsTokenizer is a custom tokenizer for processing MIDI files.
    It inherits from miditok.MusicTokenizer and can be used to convert MIDI files to token sequences and vice versa.
    """

    def __init__(self, tokenizer_config: TokenizerConfig):
        self.pitch_num = 0
        self.duration_num =0
        super().__init__(tokenizer_config=tokenizer_config)
        # You can add any additional initialization here if needed
        self.track_name_to_program = dict(
            {
                "MELODY": 0,
                "BRIDGE": 1,
                "PIANO": 1,
            }
        )


    def _create_base_vocabulary(self) -> list[str]:
        r"""
        Create the vocabulary, as a list of string tokens.
        """
        vocab = []
        # Frame
        vocab += ["Frame_None"]

        # Program
        vocab += [f"Program_{program}" for program in self.config.programs]
        # Pitch
        pitch = [f"Pitch_{i}" for i in range(*self.config.pitch_range)]
        self.pitch_num = len(pitch)
        vocab += pitch

        # Duration
        self.duration_num = len(self.durations)
        vocab += [f"Duration_{'.'.join(map(str, duration))}" for duration in self.durations]

        return vocab

    def _create_track_events(
        self,
        track: Track,
        ticks_per_beat: np.ndarray,
        time_division: int,
        ticks_bars: Sequence[int],
        ticks_beats: Sequence[int],
        attribute_controls_indexes: Mapping[int, Sequence[int] | bool] | None = None,
    ) -> list[Event]:
        r"""
        Extract the tokens/events from a track (``symusic.Track``).

        Concerned events are: *Pitch*, *Velocity*, *Duration*, *NoteOn*, *NoteOff* and
        optionally *Chord*, *Pedal* and *PitchBend*.
        **If the tokenizer is using pitch intervals, the notes must be sorted by time
        then pitch values. This is done in**
        :py:func:`miditok.MusicTokenizer.preprocess_score`.

        :param track: ``symusic.Track`` to extract events from.
        :param ticks_per_beat: array indicating the number of ticks per beat per
            section. The numbers of ticks per beat depend on the time signatures of
            the Score being parsed. The array has a shape ``(N,2)``, for ``N`` changes
            of ticks per beat, and the second dimension representing the end tick of
            each portion and the number of ticks per beat respectively.
            This argument is not required if the tokenizer is not using *Duration*,
            *PitchInterval* or *Chord* tokens. (default: ``None``)
        :param time_division: time division in ticks per quarter note of the file.
        :param ticks_bars: ticks indicating the beginning of each bar.
        :param ticks_beats: ticks indicating the beginning of each beat.
        :param attribute_controls_indexes: indices of the attribute controls to compute
            This argument has to be provided as a dictionary mapping attribute control
            indices (indexing ``tokenizer.attribute_controls``) to a sequence of
            bar indexes if the AC is "bar-level" or anything if it is "track-level".
            Its structure is as: ``{ac_idx: Any (track ac) | [bar_idx, ...] (bar ac)}``
            This argument is meant to be used when training a model in order to make it
            learn to generate tokens accordingly to the attribute controls.
        :return: sequence of corresponding ``Event``s.
        """
        # program = track.program if not track.is_drum else -1

        # Get the program by the track's name
        
        program = self.track_name_to_program[track.name]

        use_durations = program in self.config.use_note_duration_programs
        events = []
        # max_time_interval is adjusted depending on the time signature denom / tpb
        max_time_interval = 0
        if self.config.use_pitch_intervals:
            max_time_interval = ticks_per_beat[0, 1] * self.config.pitch_intervals_max_time_dist
        previous_note_onset = -max_time_interval - 1
        previous_pitch_onset = -128  # lowest at a given time
        previous_pitch_chord = -128  # for chord intervals

        tpb_idx = 0
        for note in track.notes:
            # Program
            events.append(
                Event(
                    type_="Program",
                    value=program,
                    time=note.start,
                    program=program,
                    desc=note.end,
                )
            )
            # Pitch
            events.append(
                Event(
                    type_="Pitch",
                    value=note.pitch,
                    time=note.start,
                    program=program,
                    desc=note.end,
                )
            )
            # Duration
            events.append(
                self._create_duration_event(
                    note=note,
                    _program=program,
                    _ticks_per_beat=ticks_per_beat,
                    _tpb_idx=tpb_idx,
                )
            )

        return events

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        dic: dict[str, set[str]] = {}

        # Frame -> (Program-> Pitch -> Duration, Frame )
        dic["Frame"] = {"Program", "Postion"}
        dic["Program"] = {"Pitch"}
        dic["Pitch"] = {"Duration"}

        # Duration -> (Program, Bar, Frame)
        dic["Duration"] = {"Program", "Bar", "Frame"}

        dic["Bar"] = {"Frame"}

        return dic
    def _add_time_events(self, events: list[Event], time_division: int) -> list[list[Event]]:
        r"""
        Create the time events from a list of global and track events.

        Internal method intended to be implemented by child classes.
        The returned sequence is the final token sequence ready to be converted to ids
        to be fed to a model.

        :param events: sequence of global and track events to create tokens time from.
        :param time_division: time division in ticks per quarter of the
            ``symusic.Score`` being tokenized.
        :return: the same events, with time events inserted.
        """
        # Add time events
        all_events = []
        current_tick = 0
        ticks_per_frame = self._compute_ticks_per_frame(time_division)
        events.sort(key=lambda e: e.time)
        for event in events:
            # Add all Frame events needed to advance the timeline up to the current event's time
            while current_tick <= event.time:
                self._add_position_event(all_events, current_tick, ticks_per_frame)
                current_tick += ticks_per_frame

            # Now that the timeline is up-to-date, add the musical event
            all_events.append(event)

        return all_events

    def _compute_ticks_per_frame(self, time_division: int) -> int:
        """
        计算每 Frame (1/4 拍) 的 tick 数。
        time_division 是每四分音符的tick数 (TPQ)。
        假设 1 拍 = 1 四分音符，所以每拍的 tick 数就是 time_division。
        一个 Frame 是 1/4 拍，所以是 (time_division / 4) ticks。
        """
        ticks_per_frame = time_division // 4
        ticks_per_frame = time_division // 4
        if ticks_per_frame == 0:
            ticks_per_frame = 1
            ticks_per_frame = 1
        return ticks_per_frame

    def _add_position_event(self, all_events: list[Event], current_tick: int, ticks_per_frame: int):
        """
        添加一个Position事件，作为“frame”事件来插入。
        这里的 current_tick 是即将插入的 Frame 事件的时间。
        Frame 的 value 直接是其在整个序列中的帧索引。
        """
        if ticks_per_frame == 0:
            return
        frame_index = current_tick // ticks_per_frame
        if current_tick % ticks_per_frame == 0:
            all_events.append(
                Event(
                    type_="Frame",
                    value="None",
                    time=current_tick,
                    desc=f"Frame {frame_index} (at {current_tick} ticks)",
                )
            )
    def _tokens_to_score(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

        This is an internal method called by ``self.decode``, intended to be
        implemented by classes inheriting :class:`miditok.MusicTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :return: the ``symusic.Score`` object.
        """
        # Unsqueeze tokens in case of one_token_stream
        if self.config.one_token_stream_for_programs:  # ie single token seq
            tokens = [tokens]
        for i, tokens_i in enumerate(tokens):
            tokens[i] = tokens_i.tokens
        score = Score(self.time_division)
        dur_offset = 2 if self.config.use_velocities else 1

        # RESULTS
        tracks: dict[int, Track] = {}
        tempo_changes, time_signature_changes = [], []

        def check_inst(prog: int) -> None:
            if prog not in tracks:
                tracks[prog] = Track(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

        def is_track_empty(track: Track) -> bool:
            return len(track.notes) == len(track.controls) == len(track.pitch_bends) == 0

        current_track = None
        for si, seq in enumerate(tokens):
            # First look for the first time signature if needed
            if si == 0:
                if self.config.use_time_signatures:
                    for token in seq:
                        tok_type, tok_val = token.split("_")
                        if tok_type == "TimeSig":
                            time_signature_changes.append(TimeSignature(0, *self._parse_token_time_signature(tok_val)))
                            break
                        if tok_type in [
                            "Pitch",
                            "PitchDrum",
                            "Velocity",
                            "Duration",
                            "PitchBend",
                            "Pedal",
                        ]:
                            break
                if len(time_signature_changes) == 0:
                    time_signature_changes.append(TimeSignature(0, *TIME_SIGNATURE))
            current_time_sig = time_signature_changes[-1]
            ticks_per_bar = compute_ticks_per_bar(current_time_sig, score.ticks_per_quarter)
            ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
            # ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)
            ticks_per_frame = self._compute_ticks_per_frame(self.time_division)

            # Set tracking variables
            # current_tick = tick_at_last_ts_change = tick_at_current_bar = 0
            # current_bar = -1
            current_tick = tick_at_last_ts_change = tick_at_current_frame = 0
            current_frame = -1
            bar_at_last_ts_change = 0
            current_program = 0
            previous_note_end = 0
            previous_pitch_onset = {prog: -128 for prog in self.config.programs}
            previous_pitch_chord = {prog: -128 for prog in self.config.programs}
            active_pedals = {}

            # Decode tokens
            for ti, token in enumerate(seq):
                tok_type, tok_val = token.split("_")

                if token == "Frame_None":
                    current_frame += 1
                    if current_frame > 0:
                        current_tick = tick_at_current_frame + ticks_per_frame
                    tick_at_current_frame = current_tick

                elif tok_type == "Program":
                    program_val = tok_val

                    pitch_type, pitch_val = seq[ti + 1].split("_")
                    duration_type, duration_val = seq[ti + 2].split("_")
                    (beat_num, tick_num, tick_in_beat) = duration_val.split(".")

                    duration_time = int(int(beat_num) * ticks_per_beat) + int(tick_num)
                    new_note = Note(
                        time=current_tick,
                        duration=int(duration_time),
                        pitch=int(pitch_val),
                        velocity=50,
                    )
                    # tracks[program_val].append(new_note)
                    if program_val in tracks.keys():
                        tracks[program_val].notes.append(new_note)
                    else:
                        tracks[program_val] = Track(program=program_val)
                        tracks[program_val].notes.append(new_note)

            if not self.config.one_token_stream_for_programs:
                for track in tracks.values():
                    if not is_track_empty(track):
                        score.tracks.append(track)
            else:
                for track in tracks.values():
                    if not is_track_empty(track):
                        score.tracks.append(track)
        print(score)
        return score

    def _ids_to_tokens(self, ids: list[int | list[int]], as_str: bool = True) -> list[str | Event | list[str | Event]]:
        r"""
        Convert a sequence of ids (int) to their tokens format (str or Event).

        **This method will not work with ids encoded with the tokenizer's model. You
        will need to decode them first (
        :py:meth:`miditok.MusicTokenizer.decode_token_ids`)**.

        :param ids: sequence of ids (int) to convert.
        :param as_str: return the tokens as string objects, otherwise Event objects
            (default: True)
        :return: the sequence of corresponding tokens (str or Event).
        """
        tokens = []
        if len(ids) == 0:
            return tokens
        if isinstance(ids[0], list):  # multiple vocabularies
            for multi_ids in ids:  # cannot use recursion here because of the vocabulary type id
                multi_event = []
                for i, token in enumerate(multi_ids):
                    event_str = self[token]
                    multi_event.append(event_str if as_str else Event(*event_str.split("_")))
                tokens.append(multi_event)
            tokens = tokens[0]
            return tokens

        for id_ in ids:
            event_str = self[id_]
            tokens.append(event_str if as_str else Event(*event_str.split("_")))
        
        return tokens



if __name__ == "__main__":
    fns_tokenizer_config = FnsTokenizerConfig()
    tokenizer = FnsTokenizer(fns_tokenizer_config.config)
    # tokenizer.one_token_stream = True
    tokenizer.config.one_token_stream_for_programs = True
    tokens = tokenizer.encode("datasets/Seperated-POP909-Dataset/original/001.mid")
    # print(tokens.ids)
    import miditok
    from pathlib import Path

    _tokens = miditok.TokSequence(ids=tokens.ids, are_ids_encoded=True)
    # import miditok
    # miditok.pytorch_data.
    decode = tokenizer.decode(tokens.ids)
    tokenizer.tokenize_dataset(Path("datasets/Seperated-POP909-Dataset/mel").resolve(), Path("datasets/FNS-Seperated-POP909-Dataset/mel").resolve())
    tokenizer.tokenize_dataset(Path("datasets/Seperated-POP909-Dataset/acc").resolve(), Path("datasets/FNS-Seperated-POP909-Dataset/acc").resolve())
    tokenizer.tokenize_dataset(Path("datasets/Seperated-POP909-Dataset/original").resolve(), Path("datasets/FNS-Seperated-POP909-Dataset/original").resolve())
    # decode.dump_midi("x.mid")
