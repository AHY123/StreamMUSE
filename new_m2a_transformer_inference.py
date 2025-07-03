from models.new_m2a_transformer import NewM2ATransformer, EOS_TOKEN, PAD_TOKEN
from preprocess.preprocess_midi2pt_dataset import preprocess_midi, DURATION_TEMPLATES
import torch
import pretty_midi
import os
import argparse
from typing import Literal, List
from schema.model_schema import NewM2ATransformerSchema # 引入模型 schema
import time

# Assume DURATION_TEMPLATES is accessible from StreamMUSE.preprocess.preprocess_midi2pt_dataset

def decode_output(outputs: List[torch.Tensor], save_path: str, tempo: float = 120.0, prompt: bool = True):
    """
    Decodes model output tensors into a MIDI file for NewM2ATransformer's sampling output.

    Args:
        outputs: A list of torch.Tensor, where each tensor is of shape [Batch, subseq_len].
                 The list represents the sequence of interleaved (acc, mel, acc, mel...) frames.
                 For global_sampling_from_scratch, it's (mel, acc, mel, acc...)
                 For global_sampling (with prompt), it's (a0, m0, a1, m1, ...) and then (generated a_i, gt m_i, ...)
        save_path: Path to save the generated MIDI file.
        tempo: MIDI tempo.
        prompt: Boolean, if True, the first part of output is considered prompt,
                and its type (melody/accompaniment) is inferred from `program` token.
                If False, the sequence order in `outputs` directly dictates type.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4

    # Using program 0 for melody (Piano), program 1 for accompaniment (Guitar) based on preprocess logic
    # In preprocess: x_processed[:, :, :, 0] = 0 (for mel), y_processed[:, :, :, 0] = 1 (for acc)
    # The program token is already embedded in content[i].item()
    instrument_map: dict[Literal[0, 1], pretty_midi.Instrument] = {
        0: pretty_midi.Instrument(program=0, name="Melody_Piano"), # Program 0 for melody
        1: pretty_midi.Instrument(program=24, name="Accompaniment_Guitar"), # Program 1 for accompaniment
    }
    # Add instruments to MIDI object if they don't exist
    if 0 not in [inst.program for inst in midi.instruments]:
        midi.instruments.append(instrument_map[0])
    if 24 not in [inst.program for inst in midi.instruments]:
        midi.instruments.append(instrument_map[1])


    # Total number of frames (each element in outputs is one frame, either acc or mel)
    # Each frame has its own `start_time`
    current_time_offset_frames = 0
    
    for frame_idx, data_subsequence in enumerate(outputs):
        # data_subsequence is [B, L], we take the first batch item
        content = data_subsequence.squeeze(0) # [subseq_len]

        current_start_time = current_time_offset_frames * time_step_length

        for i in range(0, len(content), 2): # Each note is (program_token, pitch_duration_token)
            program_token = int(content[i].item())

            if program_token == EOS_TOKEN:
                break # End of subsequence
            if program_token == PAD_TOKEN:
                continue # Skip padding tokens

            if i + 1 >= len(content):
                print(f"Warning: Incomplete note (program token without pitch_duration) "
                      f"at frame_idx {frame_idx}, subseq_idx {i}.")
                break

            pitch_duration_token = int(content[i + 1].item())

            # Decode program (0 or 1)
            # In preprocess: x_processed[:, :, :, 0] = 0 (for mel), y_processed[:, :, :, 0] = 1 (for acc)
            # So, program 0 is melody, program 1 is accompaniment.
            if program_token == 0 : # Melody
                midi_program = instrument_map[0].program
            elif program_token == 1: # Accompaniment
                midi_program = instrument_map[1].program
            else:
                print(f"Warning: Invalid program token {program_token} at frame_idx {frame_idx}, subseq_idx {i}. Skipping note.")
                continue


            # Decode pitch and duration from pitch_duration_token
            # In preprocess: pitch_duration = pitch + duration * 128 + 2
            # So, pitch_duration_token - 2 gives original pitch + duration * 128
            decoded_pitch_duration = pitch_duration_token - 2
            
            pitch = decoded_pitch_duration % 128
            duration_idx = decoded_pitch_duration // 128

            if not (0 <= pitch < 128):
                print(f"Warning: Invalid decoded pitch {pitch} at frame_idx {frame_idx}, subseq_idx {i}. Skipping note.")
                continue
            if not (0 <= duration_idx < len(DURATION_TEMPLATES)):
                print(f"Warning: Invalid decoded duration index {duration_idx} at frame_idx {frame_idx}, subseq_idx {i}. Skipping note.")
                continue

            note_duration_seconds = DURATION_TEMPLATES[duration_idx] * time_step_length
            end_time = current_start_time + note_duration_seconds

            # Get the correct instrument object from the MIDI object
            target_inst = None
            for inst_in_midi in midi.instruments:
                if inst_in_midi.program == midi_program:
                    target_inst = inst_in_midi
                    break
            
            if target_inst is None:
                # This should ideally not happen if instrument_map is correctly populated and added to midi.instruments
                print(f"Error: Instrument with program {midi_program} not found in MIDI object. Skipping note.")
                continue

            target_inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=current_start_time, end=end_time))
        
        current_time_offset_frames += 1 # Move to the next global time step

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)


def decompress(model: NewM2ATransformer, byte_arr_mel: List[List[int]], byte_arr_acc: List[List[int]]):
    """
    Preprocesses raw byte array data into model-compatible tensors using NewM2ATransformer's preprocess.

    Args:
        model: An instance of NewM2ATransformer.
        byte_arr_mel: Raw melody data as a list of lists of integers.
        byte_arr_acc: Raw accompaniment data as a list of lists of integers.

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor]: Preprocessed melody and accompaniment tensors.
    """
    # Ensure byte_arr_mel and byte_arr_acc are properly converted to torch.LongTensor
    # And have the expected shape for preprocess: [batch, seq, subseq]
    # Assuming byte_arr_mel[0] and byte_arr_acc[0] are already [seq, subseq]
    x_mel_raw_tensor = torch.tensor(byte_arr_mel, dtype=torch.long).unsqueeze(0).cuda() # Add batch dim
    x_acc_raw_tensor = torch.tensor(byte_arr_acc, dtype=torch.long).unsqueeze(0).cuda() # Add batch dim

    # NewM2ATransformer's preprocess expects pitch_shift as a 1D tensor [batch_size]
    # Here, batch_size is 1.
    pitch_shift_tensor = torch.zeros(x_mel_raw_tensor.shape[0], dtype=torch.long).cuda() # Changed from int8 to long

    # Call the new preprocess method
    # It returns (processed_x_mel, processed_x_acc)
    processed_x_mel, processed_x_acc = model.preprocess(x_mel_raw_tensor, pitch_shift=pitch_shift_tensor, y=x_acc_raw_tensor)
    return processed_x_mel, processed_x_acc


def continuation(
    model: NewM2ATransformer,
    midi_path: str,
    prompt_length: int = 100,
    generation_length: int = 384,
    temperature: float = 1.0,
    n_samples: int = 1,
    gt_mel: bool = True,
):
    """
    Performs continuation (generation) using the NewM2ATransformer model.

    Args:
        model: An instance of NewM2ATransformer.
        midi_path: Path to the input melody MIDI file.
        prompt_length: Number of time steps to use as prompt from the input MIDI.
        generation_length: Total number of time steps to generate (including prompt length).
        temperature: Sampling temperature.
        n_samples: Number of samples to generate.
        gt_mel: If True, uses ground truth melody for the generated part; otherwise, generates melody too.
                Note: In NewM2ATransformer, global_sampling still takes x_mel_gt, but if not provided (None),
                it generates both acc and mel.
    """
    if not os.path.isfile(midi_path):
        print(f"Error: {midi_path} is not a valid file.")
        return

    # Use preprocess_midi to get raw byte arrays
    byte_arr_mel_raw = preprocess_midi(midi_path, 4) # This returns a list of byte arrays
    byte_arr_acc_raw = preprocess_midi(midi_path.replace("mel", "acc"), 4)

    if byte_arr_mel_raw is None or not byte_arr_mel_raw:
        print(f"Error: preprocess_midi returned None or empty for mel file: {midi_path}")
        return
    if byte_arr_acc_raw is None or not byte_arr_acc_raw:
        print(f"Error: preprocess_midi returned None or empty for acc file: {midi_path.replace('mel', 'acc')}")
        return

    # Decompress (preprocess) the first item from the list of byte arrays
    x_mel_processed, x_acc_processed = decompress(model, byte_arr_mel_raw[0], byte_arr_acc_raw[0])

    # Ensure processed data is on CUDA
    x_mel_processed = x_mel_processed.cuda()
    x_acc_processed = x_acc_processed.cuda()

    B, S, L = x_mel_processed.shape # B=1, S=sequence length, L=subsequence length (e.g., 8)

    # Determine the actual starting timestep if prompt_length == 0
    first_timestep = 1
    if prompt_length == 0:
        # In this case, we use the first valid frame of x_mel_processed as the start for generation
        valid_frames = (x_mel_processed != EOS_TOKEN) & (x_mel_processed != PAD_TOKEN)
        # Find the first frame where there's any valid token
        frame_has_valid_token = valid_frames.any(dim=2).squeeze(0) # [S]
        first_valid_idx = torch.nonzero(frame_has_valid_token, as_tuple=False)
        if first_valid_idx.numel() > 0:
            first_timestep = first_valid_idx[0].item()
        else:
            print(f"Warning: No valid tokens found in {midi_path} melody for prompt_length=0. Starting from index 0.")
            first_timestep = 0 # Fallback

    # Slice data for prompt and ground truth melody (for continuation)
    # The NewM2ATransformer's global_sampling expects x as interleaved prompt, and x_mel_gt separately
    
    # Prompt melody (for global_sampling)
    prompt_mel = x_mel_processed[:, :prompt_length]
    # Prompt accompaniment (for global_sampling)
    prompt_acc = x_acc_processed[:, :prompt_length]

    # Ground truth melody for continuation (if gt_mel is True)
    # This is the melody that the model will "follow" for the generated part.
    # It starts from where the prompt ends.
    melody_for_generation_gt = x_mel_processed[:, first_timestep + prompt_length:]


    # --- Decode Prompt (if prompt_length > 0) ---
    if prompt_length > 0:
        # Create a combined prompt sequence for visualization only
        # The model's `global_sampling` will take `prompt_acc` and `prompt_mel` separately
        # But for `decode_output`, we need an interleaved sequence.
        combined_prompt_for_decode = []
        for i in range(prompt_length):
            # Append accompaniment then melody for decoding if they exist
            if i < prompt_acc.shape[1]:
                combined_prompt_for_decode.append(prompt_acc[:, i, :])
            if i < prompt_mel.shape[1]: # Only append melody if it exists
                combined_prompt_for_decode.append(prompt_mel[:, i, :])
        
        # Ensure we don't try to decode an empty list
        if combined_prompt_for_decode:
            decode_output(
                combined_prompt_for_decode,
                f"temp/{model.save_name}/{os.path.basename(midi_path)}_promptlen{prompt_length}.mid",
                tempo=90.0,
                prompt=True
            )
    
    # --- Decode original melody for reference (if not using GT melody for generation) ---
    if not gt_mel:
        # Only decode original melody if we're not using it as GT for generation
        decode_output(
            [x_mel_processed[:, i, :] for i in range(x_mel_processed.shape[1])],
            f"temp/{model.save_name}/{os.path.basename(midi_path)}_originalmelody.mid",
            tempo=90.0,
            prompt=False # Not a prompt, it's just the original melody
        )

    # --- Model Sampling ---
    with torch.no_grad():
        # Repeat prompt and GT melody for n_samples
        prompt_acc_repeated = prompt_acc.repeat(n_samples, 1, 1)
        prompt_mel_repeated = prompt_mel.repeat(n_samples, 1, 1)
        melody_for_generation_gt_repeated = melody_for_generation_gt.repeat(n_samples, 1, 1)

        start_time = time.time()
        
        # NewM2ATransformer's global_sampling takes initial_accompaniment (x) and optional x_mel_gt
        # And global_sampling_from_scratch takes x_mel
        if prompt_length == 0:
            # If no prompt, generate from scratch using the full x_mel_processed as initial melody
            # In this mode, `x_mel_gt` will be the entire input melody sequence from which acc is generated
            # The model will generate acc for each frame of x_mel_gt
            output_list = model.global_sampling_from_scratch(
                x_mel=melody_for_generation_gt_repeated, # Use the melody part that needs generation
                temperature=temperature,
                max_seq_len=generation_length # This will be the number of acc/mel pairs generated
            )
        else:
            # If there's a prompt, use global_sampling
            # prompt_acc_repeated and prompt_mel_repeated form the initial history (h)
            # melody_for_generation_gt_repeated is the future melody to follow
            output_list = model.global_sampling(
                x=prompt_acc_repeated, # This is the initial *accompaniment* part of the prompt
                x_mel_gt=melody_for_generation_gt_repeated , # Ground truth melody for future frames
                temperature=temperature,
                max_seq_len=generation_length - prompt_length # Generate the remaining length
            )
        end_time = time.time()
        print(f"Generation time: {end_time - start_time:.2f} seconds")

    # --- Decode Generated Samples ---
    for i in range(n_samples):
        # Each element in `output_list` is a tensor of shape [n_samples, subseq_len]
        # We need to extract the i-th sample from each tensor
        sample_output_subsequences = [output_tensor[i : i + 1, :] for output_tensor in output_list]

        # Combine prompt (if any) with generated output for decoding
        final_output_for_decode: List[torch.Tensor] = []
        if prompt_length > 0:
            # Append prompt (interleaved acc, mel)
            for j in range(prompt_length):
                if j < prompt_acc.shape[1]:
                    final_output_for_decode.append(prompt_acc[i : i+1, j, :]) # Get i-th sample from prompt acc
                if j < prompt_mel.shape[1]:
                    final_output_for_decode.append(prompt_mel[i : i+1, j, :]) # Get i-th sample from prompt mel

        # Append the generated sequence
        # The `output_list` from `global_sampling` (or _from_scratch) already contains interleaved acc/mel or just generated acc
        # The sampling functions return a list where:
        #   For global_sampling_from_scratch: [gen_acc_0, mel_0, gen_acc_1, mel_1, ...]
        #   For global_sampling: [prompt_a0, prompt_m0, ..., gen_a_k, gt_m_k, ...]
        final_output_for_decode.extend(sample_output_subsequences)
        
        decode_output(
            final_output_for_decode,
            f"temp/{model.save_name}/prompt{prompt_length}/{os.path.basename(midi_path)}_temp{temperature}_sample{i}.mid",
            tempo=90.0,
            prompt=True # Treat the entire sequence as potentially mixed (prompt + generated)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate accompaniment for MIDI files using NewM2ATransformer.")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.ckpt file).")
    parser.add_argument("--prompt_len", type=int, default=75, help="Number of frames to use as prompt. 0 for generation from scratch.")
    parser.add_argument("--n_samples", type=int, default=2, help="Number of samples to generate for each MIDI.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation.")
    parser.add_argument("--generation_length", type=int, default=384, help="Total number of frames to generate (including prompt).")
    parser.add_argument("--gt_mel", action="store_true", help="If set, use ground truth melody for the generated part. Otherwise, model generates both.")

    args = parser.parse_args()

    model_path = args.model_path

    # Define a dummy schema for loading the model.
    # In a real scenario, you might load this from a config file or infer from checkpoint.
    # Adjust these values to match your trained model's config.
    # For example, if "small" implies hidden_size=256, num_layers=4, etc.
    if "small" in model_path:
        model_config = NewM2ATransformerSchema(
            large=False,
            hidden_size=256,
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            local_model_num_layers=2,
            local_model_num_attention_heads=4,
            local_model_intermediate_size=512,
            frame_shift=4, # This should match the frame_shift used during training
        )
    else: # Assume large
        model_config = NewM2ATransformerSchema(hidden_size=768, num_layers=12, num_attention_heads=8, intermediate_size=2048,local_model_num_attention_heads=8,local_model_num_layers=6, local_model_intermediate_size=2048)

    # Load the NewM2ATransformer model from checkpoint using the new schema
    model = NewM2ATransformer.load_from_checkpoint(model_path, model_schema=model_config)
    
    model.save_name = os.path.basename(model_path).replace(".ckpt", "") # Clean up filename for path
    model.cuda()
    model.eval() # Set model to evaluation mode

    print(f"Loaded model: {model.save_name}")
    print(f"Generating samples (N={args.n_samples}, Temp={args.temperature}, Prompt_Len={args.prompt_len}, GT_Mel={args.gt_mel})")

    input_mel_dir = './input/mel'
    if not os.path.isdir(input_mel_dir):
        print(f"Error: Input melody directory '{input_mel_dir}' not found. Please create it and put MIDI files inside.")
    else:
        for midi_filename in os.listdir(input_mel_dir):
            if midi_filename.endswith('.mid'):
                midi_full_path = os.path.join(input_mel_dir, midi_filename)
                print(f"\nProcessing {midi_full_path}...")
                continuation(
                    model,
                    midi_full_path,
                    temperature=args.temperature,
                    generation_length=args.generation_length,
                    n_samples=args.n_samples,
                    prompt_length=args.prompt_len,
                    gt_mel=args.gt_mel
                )