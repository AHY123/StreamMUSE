import torch
import torch.nn.functional as F
from ...tokenizer.fns.tokenizer import FnsTokenizer
from typing import Optional

class FnsM2AProcessor:
    def __init__(self, tokenizer: FnsTokenizer, sub_seq_len: int):
        self.tokenizer = tokenizer
        self.sub_seq_len = sub_seq_len
        self.frame_none_id = self.tokenizer.vocab["Frame_None"]
        self.pad_token_id = self.tokenizer.pad_token_id

    def preprocess_with_all_program(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Segments token_ids into multiple frames based on 'Frame_None' tokens.
        Each frame is then processed, and padded/truncated to a (fixed_sub_seq_len) shape.
        If a frame segment, after removing Frame_None and Pad tokens, is empty,
        it will still be represented by a full padding frame of fixed_sub_seq_len.

        Args:
            token_ids (torch.Tensor): Input token IDs, with shape (batch_size, total_tokens).

        Returns:
            torch.Tensor: Processed token IDs, with shape (batch_size, seq_len, fixed_sub_seq_len).
                          'seq_len' will be the maximum number of frames across all sequences in the batch.
        """
        assert token_ids.ndim == 2, f"token_ids.ndim must be 2, but got the shape {token_ids.shape}"

        batch_size, _ = token_ids.shape
        device = token_ids.device

        frame_none_id = self.frame_none_id
        pad_token_id = self.pad_token_id
        target_sub_seq_len = self.sub_seq_len

        processed_batches: list[list[torch.Tensor]] = []
        max_seq_len_in_batch = 0  # Stores the maximum number of frames in the current batch

        for i in range(batch_size):
            single_sequence = token_ids[i]
            frames_in_sequence: list[torch.Tensor] = []

            frame_start_indices = (single_sequence == frame_none_id).nonzero(as_tuple=True)[0]

            if frame_start_indices.numel() == 0:
                non_pad_tokens = single_sequence[single_sequence != pad_token_id]

                if non_pad_tokens.numel() > 0:
                    if non_pad_tokens.shape[0] > target_sub_seq_len:
                        frames_in_sequence.append(non_pad_tokens[:target_sub_seq_len].to(device))
                    else:
                        padded_frame = F.pad(non_pad_tokens, (0, target_sub_seq_len - non_pad_tokens.shape[0]), value=pad_token_id)
                        frames_in_sequence.append(padded_frame.to(device))
                else:
                    full_pad_frame = torch.full((target_sub_seq_len,), fill_value=pad_token_id, dtype=torch.long, device=device)
                    frames_in_sequence.append(full_pad_frame)
            else:
                current_frame_indices = frame_start_indices.tolist()

                for j in range(len(current_frame_indices)):
                    start_idx = current_frame_indices[j]
                    end_idx = current_frame_indices[j + 1] if j + 1 < len(current_frame_indices) else single_sequence.shape[0]
                    current_frame_segment = single_sequence[start_idx:end_idx]
                    current_frame_content = current_frame_segment[(current_frame_segment != pad_token_id) & (current_frame_segment != frame_none_id)]

                    if current_frame_content.numel() > 0:
                        if current_frame_content.shape[0] > target_sub_seq_len:
                            frames_in_sequence.append(current_frame_content[:target_sub_seq_len].to(device))
                        else:
                            padded_frame = F.pad(current_frame_content, (0, target_sub_seq_len - current_frame_content.shape[0]), value=pad_token_id)
                            frames_in_sequence.append(padded_frame.to(device))
                    else:
                        full_pad_frame = torch.full((target_sub_seq_len,), fill_value=pad_token_id, dtype=torch.long, device=device)
                        frames_in_sequence.append(full_pad_frame)

            if len(frames_in_sequence) > max_seq_len_in_batch:
                max_seq_len_in_batch = len(frames_in_sequence)

            processed_batches.append(frames_in_sequence)

        output_tensor = torch.full((batch_size, max_seq_len_in_batch, target_sub_seq_len), fill_value=pad_token_id, dtype=torch.long, device=device)

        for i, frames_for_this_seq in enumerate(processed_batches):
            for j, frame_tensor in enumerate(frames_for_this_seq):
                output_tensor[i, j, :] = frame_tensor

        return output_tensor

    def preprocess_with_program_id(self, token_ids: torch.Tensor, program_token_id: int) -> torch.Tensor:
        """
        Segments token_ids into multiple frames based on 'Frame_None' tokens.
        Each frame is then processed: filtered by program_token_id, and padded/truncated
        to a (fixed_sub_seq_len) shape.
        """
        assert token_ids.ndim == 2, f"token_ids.ndim must be 2, but got the shape {token_ids.shape}"

        batch_size, _ = token_ids.shape
        device = token_ids.device

        frame_none_id = self.frame_none_id
        pad_token_id = self.pad_token_id
        target_sub_seq_len = self.sub_seq_len

        output_sequences: list[torch.Tensor] = []

        for i in range(batch_size):
            single_sequence = token_ids[i]
            frame_start_indices = (single_sequence == frame_none_id).nonzero(as_tuple=True)[0]
            frames_for_current_sequence: list[torch.Tensor] = []

            if frame_start_indices.numel() == 0:
                filtered_sequence_tokens = []
                non_pad_tokens = single_sequence[single_sequence != pad_token_id]

                for k in range(0, non_pad_tokens.numel(), 3):
                    if k + 2 < non_pad_tokens.numel():
                        current_program_id = non_pad_tokens[k].item()
                        if current_program_id == program_token_id:
                            filtered_sequence_tokens.extend(non_pad_tokens[k:k+3])

                if len(filtered_sequence_tokens) > 0:
                    filtered_tensor = torch.tensor(filtered_sequence_tokens, dtype=torch.long, device=device)
                    if filtered_tensor.shape[0] > target_sub_seq_len:
                        frames_for_current_sequence.append(filtered_tensor[:target_sub_seq_len])
                    else:
                        padded_frame = F.pad(filtered_tensor, (0, target_sub_seq_len - filtered_tensor.shape[0]), value=pad_token_id)
                        frames_for_current_sequence.append(padded_frame)
                else:
                    full_pad_frame = torch.full((target_sub_seq_len,), fill_value=pad_token_id, dtype=torch.long, device=device)
                    frames_for_current_sequence.append(full_pad_frame)
            else:
                current_frame_indices = frame_start_indices.tolist()

                for j in range(len(current_frame_indices)):
                    start_idx = current_frame_indices[j]
                    end_idx = current_frame_indices[j + 1] if j + 1 < len(current_frame_indices) else single_sequence.shape[0]
                    current_frame_segment = single_sequence[start_idx:end_idx]
                    current_frame_non_padded = current_frame_segment[(current_frame_segment != pad_token_id) & (current_frame_segment != frame_none_id)]

                    filtered_frame_tokens = []
                    for k in range(0, current_frame_non_padded.numel(), 3):
                        if k + 2 < current_frame_non_padded.numel():
                            current_program_id = current_frame_non_padded[k].item()
                            if current_program_id == program_token_id:
                                filtered_frame_tokens.extend(current_frame_non_padded[k:k+3])

                    if len(filtered_frame_tokens) > 0:
                        filtered_tensor_frame = torch.tensor(filtered_frame_tokens, dtype=torch.long, device=device)
                        if filtered_tensor_frame.shape[0] > target_sub_seq_len:
                            frames_for_current_sequence.append(filtered_tensor_frame[:target_sub_seq_len])
                        else:
                            padded_frame = F.pad(filtered_tensor_frame, (0, target_sub_seq_len - filtered_tensor_frame.shape[0]), value=pad_token_id)
                            frames_for_current_sequence.append(padded_frame)
                    else:
                        full_pad_frame = torch.full((target_sub_seq_len,), fill_value=pad_token_id, dtype=torch.long, device=device)
                        frames_for_current_sequence.append(full_pad_frame)

            if frames_for_current_sequence:
                output_sequences.append(torch.stack(frames_for_current_sequence))
            else:
                output_sequences.append(torch.empty(0, target_sub_seq_len, dtype=torch.long, device=device))

        # This part needs to handle padding sequences to the same number of frames.
        max_frames = max(seq.shape[0] for seq in output_sequences) if output_sequences else 0
        padded_output = torch.full((batch_size, max_frames, target_sub_seq_len), fill_value=pad_token_id, dtype=torch.long, device=device)
        for i, seq in enumerate(output_sequences):
            if seq.numel() > 0:
                padded_output[i, :seq.shape[0], :] = seq

        return padded_output

    def postprocess_to_all_program(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor of token IDs back to a sequence of tokens, including 'Frame_None' tokens.
        """
        assert token_ids.ndim == 3, f"token_ids.ndim must be 3, but got the shape {token_ids.shape}"
        batch_size, _, _ = token_ids.shape
        device = token_ids.device
        pad_token_id = self.pad_token_id
        
        output_sequences = []
        for i in range(batch_size):
            frames = token_ids[i]
            # Remove padding from each frame and flatten
            content_tokens = [frame[frame != pad_token_id] for frame in frames]
            # Interleave with frame_none_id
            full_sequence = []
            for frame_content in content_tokens:
                if frame_content.numel() > 0:
                    full_sequence.append(frame_content)
                full_sequence.append(torch.tensor([self.frame_none_id], dtype=torch.long, device=device))
            
            if full_sequence:
                # Remove the last frame_none_id if it's extraneous
                if full_sequence[-1].item() == self.frame_none_id and len(full_sequence) > 1:
                     full_sequence.pop()
                output_sequences.append(torch.cat(full_sequence))
            else:
                output_sequences.append(torch.tensor([], dtype=torch.long, device=device))

        # Pad all sequences in the batch to the same length
        return torch.nn.utils.rnn.pad_sequence(output_sequences, batch_first=True, padding_value=pad_token_id)

    def postprocess_with_program_id(self, token_ids: torch.Tensor, program_token_id: int) -> torch.Tensor:
        """
        Converts a tensor of token IDs back to a sequence of tokens, filtering by program_token_id.
        """
        # This function's logic was flawed in the original. It was filtering for the program_id
        # but the input `token_ids` from the model would not contain program_ids.
        # A more logical implementation is to just flatten the output, similar to postprocess_to_all_program.
        # If specific filtering is needed, it should be applied to the final flat sequence.
        # Here, we provide a corrected flattening logic.
        
        assert token_ids.ndim == 3, f"token_ids.ndim must be 3, but got the shape {token_ids.shape}"
        batch_size, _, _ = token_ids.shape
        device = token_ids.device
        pad_token_id = self.pad_token_id

        output_sequences = []
        for i in range(batch_size):
            # Flatten the frames, removing padding
            non_pad_tokens = token_ids[i][token_ids[i] != pad_token_id]
            output_sequences.append(non_pad_tokens)

        # Pad all sequences in the batch to the same length
        return torch.nn.utils.rnn.pad_sequence(output_sequences, batch_first=True, padding_value=pad_token_id)
