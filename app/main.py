import time
import threading
from queue import Queue
import argparse
import torch

# Import our application components
from app.input_handlers.pc_keyboard_input import read_keyboard_input
from app.inference_engines.transformer_engine import TransformerInferenceEngine
from app.output_handlers.cli_output import CLIOutputHandler

def tick_loop(event_queue: Queue, inference_engine: TransformerInferenceEngine, output_handler: CLIOutputHandler, tempo: float, ticks_per_beat: int):
    """
    The Conductor loop, now with improved logic for capturing input
    and a debug flag for the display.
    """
    seconds_per_tick = (60.0 / tempo) / ticks_per_beat
    tick_count = -1
    playback_schedule = {}
    
    # This list will accumulate all notes played between downbeats.
    notes_played_since_last_downbeat = []

    # --- Benchmarking State ---
    inference_times = []
    warmup_completed = False
    # --- End Benchmarking State ---

    while True:
        tick_count += 1
        
        # Harvest all note events that have accumulated since the last tick.
        user_notes_this_tick = []
        while not event_queue.empty():
            event = event_queue.get()
            if event is None:
                print("\r\nMetronome thread exiting.")
                return
            user_notes_this_tick.append(event)
        
        # Add any notes played on this tick to our accumulator.
        if user_notes_this_tick:
            notes_played_since_last_downbeat.extend(user_notes_this_tick)

        is_downbeat_tick = (tick_count % ticks_per_beat) == 0
        
        debug_info = {
            "events_this_tick": user_notes_this_tick,
            "inference_triggered": False
        }
        
        # On the downbeat, check if any notes were played during the previous beat.
        if is_downbeat_tick and notes_played_since_last_downbeat:
            debug_info["inference_triggered"] = True
            
            # --- Benchmarking Procedure ---
            torch.cuda.synchronize()  # Ensure all GPU operations are complete before timing.
            start_time = time.perf_counter()
            future_events = inference_engine.generate_accompaniment(notes_played_since_last_downbeat, tick_count)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            last_inference_time = end_time - start_time
            
            if not warmup_completed:
                debug_info["warmup_time"] = last_inference_time
                warmup_completed = True
            else:
                inference_times.append(last_inference_time)
                debug_info["last_inference_time"] = last_inference_time
                if inference_times:
                    debug_info["avg_inference_time"] = sum(inference_times) / len(inference_times)
                    debug_info["inference_count"] = len(inference_times)
            # --- End Benchmarking Procedure ---
            
            for event in future_events:
                play_tick = event["play_at_tick"]
                if play_tick not in playback_schedule:
                    playback_schedule[play_tick] = []
                playback_schedule[play_tick].append({'type': 'note_on', 'pitch': event['pitch']})

                duration_in_seconds = event['duration']
                duration_in_ticks = max(1, round(duration_in_seconds / seconds_per_tick))
                note_off_tick = play_tick + duration_in_ticks
                
                if note_off_tick not in playback_schedule:
                    playback_schedule[note_off_tick] = []
                playback_schedule[note_off_tick].append({'type': 'note_off', 'pitch': event['pitch']})
            
            # Clear the accumulator after running inference.
            notes_played_since_last_downbeat = []

        model_events_this_tick = playback_schedule.pop(tick_count, [])

        # Pass all necessary info to the output handler for display.
        output_handler.update_and_display(tick_count, debug_info, model_events_this_tick)

        time.sleep(seconds_per_tick)

def main():
    """
    The main entry point of the application.
    Sets up the components and threads.
    """
    parser = argparse.ArgumentParser(description="StreamMUSE: A real-time interactive music generation CLI.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint (.ckpt) file.")
    args = parser.parse_args()

    TICKS_PER_BEAT = 4
    TEMPO = 120.0

    event_queue = Queue()
    
    try:
        inference_engine = TransformerInferenceEngine(checkpoint_path=args.checkpoint_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    output_handler = CLIOutputHandler(ticks_per_beat=TICKS_PER_BEAT)

    input_thread = threading.Thread(
        target=read_keyboard_input, 
        args=(event_queue,),
        daemon=True
    )

    conductor_thread = threading.Thread(
        target=tick_loop,
        args=(event_queue, inference_engine, output_handler, TEMPO, TICKS_PER_BEAT),
        daemon=True
    )

    print("Starting StreamMUSE...")
    # Create space for the rolling display PLUS the new debug line.
    print("\n" * (TICKS_PER_BEAT + 1))

    input_thread.start()
    conductor_thread.start()

    input_thread.join()
    output_handler.save_log_on_exit()
    print("\r\nExiting application.")

if __name__ == "__main__":
    main()