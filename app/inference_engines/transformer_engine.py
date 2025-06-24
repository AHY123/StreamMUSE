import torch
import os
import numpy as np

# We need to import the model definition and special tokens from your training script.
from m2a_transformer import RoFormerSymbolicTransformer, PAD_TOKEN, EOS_TOKEN

# DURATION_TEMPLATES is used to decode the model's duration token output.
# This should ideally be in a shared settings/config file.
DURATION_TEMPLATES = np.array([
    0.0, 0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 
    0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 
    1.25, 1.375, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0
])

class TransformerInferenceEngine:
    """
    An inference engine that uses the RoFormerSymbolicTransformer for real-time generation.
    """
    def __init__(self, checkpoint_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        print(f"\rLoading model from {checkpoint_path}...")
        self.device = torch.device(device)

        # Determine model size from checkpoint name, similar to m2a_transformer_inference.py
        is_large = 'large' in os.path.basename(checkpoint_path)
        self.model = RoFormerSymbolicTransformer.load_from_checkpoint(
            checkpoint_path, large=is_large, strict=False
        )
        self.model.to(self.device)
        self.model.eval()
        print("\rTransformer model loaded successfully.")

        # --- Real-time State Initialization ---
        # The history of summary vectors (h). We start with the global SOS token.
        self.h_history = self.model.global_sos.view(1, 1, -1).to(self.device)
        
        # --- Configuration ---
        self.subseq_len = 64  # The expected number of tokens per frame.
        self.temperature = 1.0  # Sampling temperature.

    def _notes_to_frame(self, note_events: list) -> torch.Tensor:
        """Converts a list of note events into a tokenized frame tensor."""
        # For real-time input, we assume a fixed duration for each note press.
        # Index 4 corresponds to a duration of 0.125s (a 16th note at 120bpm).
        duration_index = 4 
        tokens = []

        # The model expects frames as a flat list of [program, pitch_duration, ...].
        # For melody, the program token is 0.
        for event in note_events:
            pitch = event['pitch']
            # Encode pitch and duration into a single token, as done in training.
            pitch_duration_token = pitch + duration_index * 128 + 2
            tokens.extend([0, pitch_duration_token])

        # Pad the frame to the required subsequence length.
        frame = tokens[:self.subseq_len]
        frame += [PAD_TOKEN] * (self.subseq_len - len(frame))
        
        return torch.tensor(frame, dtype=torch.long, device=self.device).unsqueeze(0)

    def _frame_to_notes(self, frame_tensor: torch.Tensor) -> list:
        """Converts a generated frame of tokens back to note events."""
        notes = []
        frame = frame_tensor.squeeze(0).tolist()

        # Process the frame in (program, pitch_duration) pairs.
        for i in range(0, len(frame), 2):
            program_token = frame[i]
            if program_token in (EOS_TOKEN, PAD_TOKEN):
                break
            if i + 1 >= len(frame):
                break
            
            pitch_duration_token = frame[i+1]
            if pitch_duration_token in (EOS_TOKEN, PAD_TOKEN):
                break

            # Decode the pitch_duration token back into pitch and duration index.
            pitch_duration = pitch_duration_token - 2
            pitch = pitch_duration % 128
            # duration_idx = pitch_duration // 128 # We don't need duration for now.
            
            notes.append({'pitch': pitch})
        return notes

    def _frame_to_notes(self, frame_tensor: torch.Tensor) -> list:
        """Converts a generated frame of tokens back to note events with duration."""
        notes = []
        frame = frame_tensor.squeeze(0).tolist()

        # Process the frame in (program, pitch_duration) pairs.
        for i in range(0, len(frame), 2):
            program_token = frame[i]
            if program_token in (EOS_TOKEN, PAD_TOKEN):
                break
            if i + 1 >= len(frame):
                break
            
            pitch_duration_token = frame[i+1]
            if pitch_duration_token in (EOS_TOKEN, PAD_TOKEN):
                break

            # Decode the pitch_duration token back into pitch and duration index.
            pitch_duration = pitch_duration_token - 2
            pitch = pitch_duration % 128
            duration_idx = pitch_duration // 128
            
            # Convert duration index to seconds using the template.
            duration_sec = DURATION_TEMPLATES[duration_idx] if duration_idx < len(DURATION_TEMPLATES) else 0.0
            
            notes.append({'pitch': pitch, 'duration': duration_sec})
        return notes

    @torch.no_grad()
    def generate_accompaniment(self, user_note_events: list, current_tick_count: int) -> list:
        """
        The main real-time generation method. This is a stateful, step-by-step
        version of the `global_sampling` function from the training script.
        """
        # 1. Convert user input to a melody frame and get its summary vector.
        melody_frame = self._notes_to_frame(user_note_events)
        token_type_ids_mel = torch.zeros((1, 1, self.subseq_len + 1), dtype=torch.long, device=self.device)
        h_mel, _ = self.model.local_encode(melody_frame.unsqueeze(1), token_type_ids_mel)

        # 2. Append the melody summary to our ongoing history.
        #    FIX: Add a dimension to the 2D h_mel tensor to make it 3D for concatenation.
        self.h_history = torch.cat([self.h_history, h_mel.unsqueeze(1)], dim=1)

        # 3. Run the global model to predict the summary for the next accompaniment frame.
        causal_mask = self.model.buffered_future_mask(self.h_history)
        h_out_sequence = self.model.model(self.h_history, attention_mask=causal_mask, interleave_pos=True)[0]
        h_acc_pred = h_out_sequence[:, -1, :]

        # 4. Generate the accompaniment frame tokens from the predicted summary.
        acc_frame = self.model.global_sampling(h_acc_pred, h_mel, max_subseq_len=self.subseq_len, temperature=self.temperature)

        # --- FIX: Pad the generated frame to the required length ---
        # The generated frame can be shorter than subseq_len if EOS is sampled.
        # We must pad it to the expected length before encoding.
        actual_len = acc_frame.shape[1]
        if actual_len < self.subseq_len:
            padding = torch.full(
                (1, self.subseq_len - actual_len), 
                PAD_TOKEN, 
                dtype=torch.long, 
                device=self.device
            )
            acc_frame = torch.cat([acc_frame, padding], dim=1)
        # --- End of FIX ---

        # 5. Get the summary of the frame we just generated to keep the history consistent.
        token_type_ids_acc = torch.ones((1, 1, self.subseq_len + 1), dtype=torch.long, device=self.device)
        h_acc, _ = self.model.local_encode(acc_frame.unsqueeze(1), token_type_ids_acc)
        
        self.h_history = torch.cat([self.h_history, h_acc.unsqueeze(1)], dim=1)

        # 6. Decode the generated frame into playable notes with DURATION.
        accompaniment_notes = self._frame_to_notes(acc_frame)

        # 7. Schedule the notes to be played in the near future.
        scheduled_tick = current_tick_count + 2
        scheduled_events = [
            {'pitch': note['pitch'], 'duration': note['duration'], 'play_at_tick': scheduled_tick}
            for note in accompaniment_notes
        ]
        
        return scheduled_events