import sys
import threading
import unittest
from pathlib import Path
from queue import Queue
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from input_handlers import input_handler as input_handler_module
from output_handlers.midi_file_handler import MidiFileHandler


class FakePort:
    def __init__(self):
        self.closed = False
        self.name = "fake-port"

    def poll(self):
        return None

    def close(self):
        self.closed = True


def drain_queue(event_queue: Queue) -> list[dict]:
    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())
    return events


def notes_to_tick_tuples(handler: MidiFileHandler):
    notes = []
    for note in sorted(handler.user_instrument.notes, key=lambda n: (n.start, n.pitch)):
        start_tick = int(round(note.start / handler.seconds_per_tick))
        end_tick = int(round(note.end / handler.seconds_per_tick))
        notes.append((int(note.pitch), start_tick, max(1, end_tick - start_tick)))
    return notes


class RecordingOrderIssueTests(unittest.TestCase):
    def test_read_midi_input_stop_event_preserves_queue_order(self):
        event_queue = Queue()
        expected_events = [
            {"type": "note_on", "pitch": 65, "tick": 26},
            {"type": "note_off", "pitch": 62, "tick": 24},
            {"type": "note_on", "pitch": 64, "tick": 25},
            {"type": "note_off", "pitch": 64, "tick": 26},
            {"type": "note_off", "pitch": 65, "tick": 27},
        ]
        for event in expected_events:
            event_queue.put(event)

        stop_event = threading.Event()
        stop_event.set()
        fake_port = FakePort()

        with patch.object(input_handler_module.mido, "open_input", return_value=fake_port):
            input_handler_module.read_midi_input(
                event_queue=event_queue,
                device_name="fake-device",
                stop_event=stop_event,
            )

        self.assertTrue(fake_port.closed)
        self.assertEqual(drain_queue(event_queue), expected_events)

    def test_midi_file_handler_still_requires_ordered_events(self):
        handler = MidiFileHandler(tempo=120, ticks_per_beat=4)
        reordered_events = [
            {"type": "note_off", "pitch": 65, "tick": 27},
            {"type": "note_on", "pitch": 65, "tick": 26},
            {"type": "note_on", "pitch": 67, "tick": 29},
            {"type": "note_off", "pitch": 67, "tick": 30},
        ]
        for event in reordered_events:
            handler.add_user_note(event)
        handler.finalize()

        tick_notes = notes_to_tick_tuples(handler)
        self.assertIn((65, 26, 4), tick_notes)

    def test_preserved_queue_order_records_correct_short_note(self):
        event_queue = Queue()
        ordered_events = [
            {"type": "note_on", "pitch": 65, "tick": 26},
            {"type": "note_off", "pitch": 65, "tick": 27},
            {"type": "note_on", "pitch": 67, "tick": 29},
            {"type": "note_off", "pitch": 67, "tick": 30},
        ]
        for event in ordered_events:
            event_queue.put(event)

        stop_event = threading.Event()
        stop_event.set()
        fake_port = FakePort()

        with patch.object(input_handler_module.mido, "open_input", return_value=fake_port):
            input_handler_module.read_midi_input(
                event_queue=event_queue,
                device_name="fake-device",
                stop_event=stop_event,
            )

        handler = MidiFileHandler(tempo=120, ticks_per_beat=4)
        for event in drain_queue(event_queue):
            handler.add_user_note(event)
        handler.finalize()

        tick_notes = notes_to_tick_tuples(handler)
        self.assertIn((65, 26, 1), tick_notes)
        self.assertNotIn((65, 26, 4), tick_notes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
