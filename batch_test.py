import os
import sys
import time
import json
import argparse
import itertools
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

# Default Configurations
DEFAULT_INPUT_DIR = "input/mel"
DEFAULT_OUTPUT_BASE = "output_batch"
CLIENT_SCRIPT = "app/client_lekai.py"
SERVER_URL = "http://localhost:8988/generate_accompaniment"

# Test Dimensions
TEMPOS = [90, 120]
PROMPT_LENGTHS_TICKS = [0, 16]  # 0 = no prompt, 16 = 1 bar (assuming 4 ticks/beat, 4/4)
GENERATION_LENGTHS_TICKS = [128, 256]  # 32 beats (8 bars * 4)

# Sampling Configurations
SAMPLING_CONFIGS = [
    {"temperature": 1.1, "top_k": 10, "top_p": 0.95, "name": "t1.1_k10"},
    {"temperature": 1.0, "top_k": 50, "top_p": 0.95, "name": "t1.0_k50"},
    {"temperature": 0.9, "top_k": 5, "top_p": 0.90, "name": "t0.9_k5"},
    {"temperature": 0.0, "top_k": 1, "top_p": 0.0, "name": "t0.0_k1"},
]


def check_server_health(url):
    try:
        import requests

        # Simple health check if endpoint exists, or just check root
        # Server API docs at /docs
        requests.get(f"{url}/docs", timeout=1)
        return True
    except:
        return False


def run_batch_test(args):
    """
    Main batch testing loop (Client Integration Mode).
    """
    if not os.path.exists(CLIENT_SCRIPT):
        print(f"Error: Client script not found at {CLIENT_SCRIPT}")
        sys.exit(1)

    # Check server
    print(f"Checking server at {SERVER_URL}...")
    # Optional: could enforce check, but for now just warn
    # if not check_server_health(SERVER_URL):
    #     print("Warning: Server might be down.")

    # 2. Prepare Config Combinations
    input_files = sorted(list(Path(args.input_dir).glob("*.mid")))
    if not input_files:
        print(f"No MIDI files found in {args.input_dir}")
        sys.exit(1)

    combinations = list(
        itertools.product(input_files, TEMPOS, PROMPT_LENGTHS_TICKS, GENERATION_LENGTHS_TICKS, SAMPLING_CONFIGS)
    )

    # Filter for single test mode
    if args.test_mode:
        print("Running in TEST MODE (single combination)")
        combinations = combinations[:1]

    print(f"Starting batch run with {len(combinations)} combinations (x{args.samples} samples each)...")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_output_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Create consolidated MIDI output directory
    midi_output_dir = run_output_dir / "midi_files"
    midi_output_dir.mkdir(exist_ok=True)

    metrics = []

    for input_file, tempo, prompt_len, gen_len, sampling in tqdm(combinations, desc="Processing"):
        # Unique ID for this specific config
        config_name = f"{input_file.stem}_T{tempo}_P{prompt_len}_L{gen_len}_{sampling['name']}"

        # Loop for N samples
        for sample_idx in range(args.samples):
            sample_name = f"{config_name}_s{sample_idx}"
            # Client output dir (stores logs, raw midi)
            output_subdir = run_output_dir / "logs" / sample_name

            try:
                # --- Construct Client Command ---
                # "Frames" logic: Client divides arg by 2 to get ticks.
                # So to get L ticks, we pass 2*L frames.
                gen_len_frames = gen_len * 2

                cmd = [
                    sys.executable,
                    CLIENT_SCRIPT,
                    "--server_url",
                    SERVER_URL,
                    "--tempo",
                    str(tempo),
                    "--generation_length",
                    str(gen_len_frames),
                    # Input
                    "--midi-file-input",
                    str(input_file),
                    # Output
                    "--output-dir",
                    str(output_subdir),
                    # Sampling
                    "--temperature",
                    str(sampling["temperature"]),
                    "--top_k",
                    str(sampling["top_k"]),
                    "--top_p",
                    str(sampling["top_p"]),
                    "--no-midi-output",  # Headless
                    # Headless/Automation (handled by gen_length limit)
                ]

                # Injection / Prompt
                if prompt_len > 0:
                    cmd.extend(["--injection-file", str(input_file)])
                    cmd.extend(["--injection-length", str(prompt_len)])

                # Log command (debug)
                # print(" ".join(cmd))

                start_time = time.perf_counter()

                # Run Client
                # Capture stdout/stderr to avoid pollution? Or let it show?
                # Tqdm is active, so capturing is better.
                result = subprocess.run(cmd, capture_output=True, text=True)

                duration = time.perf_counter() - start_time

                if result.returncode != 0:
                    print(f"Error in {sample_name}: {result.stderr}")
                    status = "failed"
                else:
                    status = "success"

                # --- Move/Rename MIDI File ---
                # Client usually saves as {input_filename}.mid in output_dir
                # We want to move it to midi_files/{sample_name}.mid

                target_midi_path = None
                generated_files_list = []

                if status == "success":
                    # Look for likely midi output
                    # The client saves "001.mid" (input name)
                    expected_src_midi = output_subdir / input_file.name

                    if expected_src_midi.exists():
                        target_midi_name = f"{sample_name}.mid"
                        target_midi_path = midi_output_dir / target_midi_name

                        # Copy or Move
                        # Copy or Move
                        shutil.copy2(expected_src_midi, target_midi_path)
                        generated_files_list.append(target_midi_name)
                    else:
                        # Fallback: check any mid file
                        mids = list(output_subdir.glob("*.mid"))
                        if mids:
                            # Taking the first one if exact match not found
                            src = mids[0]
                            target_midi_name = f"{sample_name}.mid"
                            target_midi_path = midi_output_dir / target_midi_name
                            shutil.copy2(src, target_midi_path)
                            generated_files_list.append(target_midi_name)

                # --- Record Metrics ---
                metrics.append(
                    {
                        "config": config_name,
                        "sample_idx": sample_idx,
                        "input_file": input_file.name,
                        "tempo": tempo,
                        "prompt_len": prompt_len,
                        "gen_len": gen_len,
                        "sampling": sampling,
                        "duration": duration,
                        "log_dir": str(output_subdir),
                        "midi_path": str(target_midi_path) if target_midi_path else None,
                        "status": status,
                    }
                )

            except Exception as e:
                print(f"FAILED {config_name} Sample {sample_idx}: {e}")
                import traceback

                traceback.print_exc()

    # Save Metrics
    with open(run_output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Batch run complete. Results in {run_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--test-mode", action="store_true", help="Run only 1 combination for verification")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples per configuration")
    args = parser.parse_args()

    run_batch_test(args)
