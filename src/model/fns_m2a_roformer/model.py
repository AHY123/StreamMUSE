import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..pl_base_model.model import PlBaseModel
from .config import FnsM2ARoformerConfig
from .model_io import FnsM2ARoformerInput
from ...tokenizer.Fns.tokenizer import FnsTokenizer
from symusic import Score
import hydra


# Indicator: 0
# pitch+duration*2: 3200 (25*128)
N_NORMAL_TOKENS = 3202
N_TOKENS = N_NORMAL_TOKENS + 3
SOS_TOKEN = N_NORMAL_TOKENS
EOS_TOKEN = N_NORMAL_TOKENS + 1
PAD_TOKEN = N_NORMAL_TOKENS + 2


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


class FnsM2ATransformer(PlBaseModel):
    def __init__(self, config: FnsM2ARoformerConfig):
        super().__init__(config)
        local_decoder_config = config.local_decoder_network_config
        local_encoder_config = config.local_encoder_network_config
        global_network_config = config.global_network_config
        tokenizer_config = config.tokenizer_config

        # tokenization params
        self.tokenizer = FnsTokenizer(tokenizer_config.config)
        self.sub_seq_len = config.sub_seq_len
        self.frame_none_id = self.tokenizer.vocab["Frame_None"]
        self.mel_program_id = self.tokenizer.vocab["Program_0"]
        self.acc_program_id = self.tokenizer.vocab["Program_1"]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.TOKEN_NUM_USED = self.tokenizer.pitch_num * self.tokenizer.duration_num

        # network params
        self.local_encoder = hydra.utils.instantiate(local_encoder_config)
        self.model = hydra.utils.instantiate(global_network_config)
        self.local_decoder = hydra.utils.instantiate(local_decoder_config)
        encoder_hidden_size = config.local_encoder_network_config.config.hidden_size
        decoder_hidden_size = config.local_decoder_network_config.config.hidden_size

        # embbedig params , token_used -> hidden_size
        self.local_embedding = nn.Embedding(self.TOKEN_NUM_USED, encoder_hidden_size)
        self.token_type_embeddings = nn.Embedding(2, encoder_hidden_size)
        with torch.no_grad():
            self.token_type_embeddings.weight.mul_(2.0)
        self.final_decoder = nn.Linear(decoder_hidden_size, self.TOKEN_NUM_USED)
        self.global_sos = nn.Parameter(torch.randn(encoder_hidden_size))
        self._future_mask = torch.empty(0)

    def local_encode(self, x, token_type_ids):
        batch_size, seq_len, subseq_len = x.shape
        x = x.view(-1, subseq_len)

        # prepend SOS:
        x = torch.cat([torch.full((x.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=x.device), x], dim=-1)  # now [B*seq_len, subseq_len+1]

        mask = x != PAD_TOKEN  # [B*seq_len, subseq_len+1]
        word_emb = self.local_embedding(x)  # → [B*seq_len, subseq_len+1, H]

        type_emb = self.token_type_embeddings(token_type_ids)  # [B*seq_len, subseq_len+1, H]
        type_emb = type_emb.view(batch_size * seq_len, word_emb.shape[1], -1)

        emb = word_emb + type_emb
        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]

        return h[:, 0], emb[:, :-1]

    def local_decode(self, h, emb):
        batch_size, subseq_len, _ = emb.shape
        # Add h as the first token of emb
        h = h.view(batch_size, 1, -1)
        # print(emb.shape) #3840 8 512
        emb = torch.cat([h, emb[:, 1:]], dim=1)
        # Create an autoregressive mask
        h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
        return self.final_decoder(h)

    def local_sampling(self, h, max_subseq_len=32, temperature=1.0):
        batch_size, _ = h.shape
        y = torch.zeros((batch_size, 0), dtype=torch.long, device=h.device)
        emb = h[:, None, :]
        eos_triggered = torch.zeros(batch_size, dtype=torch.bool, device=h.device)

        for i in range(max_subseq_len):
            h_ = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
            if temperature == 0:
                p = F.one_hot(self.final_decoder(h_).argmax(dim=-1), N_TOKENS).float()
            else:
                p = F.softmax(self.final_decoder(h_[:, -1]) / temperature, dim=-1)
            y_next = torch.multinomial(p, 1)
            y_next[eos_triggered, :] = PAD_TOKEN
            eos_triggered = eos_triggered | (y_next.squeeze(1) == EOS_TOKEN)
            y = torch.cat([y, y_next], dim=1)
            if torch.all(eos_triggered):
                break

            # 5a) now append the embedding (always ACCOMPANIMENT), so token_type_ids = 1
            emb = torch.cat([emb, self.local_embedding(y_next) + self.token_type_embeddings(torch.ones_like(y_next))], dim=1)

        return y

    def global_sampling(self, x, x_mel_gt=None, max_seq_len=384, temperature=1.0):
        batch_size, seq_len, subseq_len = x.shape
        _, seq_len_gt, _ = x_mel_gt.shape
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()  # → [seq_len], 1 at even idx (acc), 0 at odd idx (mel)
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)
        h, _ = self.local_encode(x, token_type_ids)
        h_mel, _ = self.local_encode(
            x_mel_gt, torch.zeros(*x_mel_gt.shape[:-1], x_mel_gt.shape[-1] + 1, device=x_mel_gt.device, dtype=x_mel_gt.dtype)
        )
        h = h.view(batch_size, seq_len, -1)
        h_mel = h_mel.view(batch_size, seq_len_gt, -1)
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list a0,m0,a1,m1,a_to_be_2
        if x_mel_gt != None:
            # print('with gt!')
            for i in range(0, max_seq_len):
                if i % 10 == 0:
                    # print('Sampling', i, '/', max_seq_len)
                    ...
                if i % 2 == 0:
                    h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
                    y_next = self.local_sampling(h_out[:, -1], max_subseq_len=subseq_len, temperature=temperature)
                    y.append(y_next)
                    b, s, l = y_next.unsqueeze(1).shape
                    token_type_ids = torch.ones((b, s, l + 1), dtype=torch.long, device=y_next.device)
                    h = torch.cat([h, self.local_encode(y_next.unsqueeze(1), token_type_ids=token_type_ids)[0].unsqueeze(1)], dim=1)
                else:
                    # token_type_ids = torch.zeros((b, s, l+1), dtype=torch.long, device=y_next.device)
                    h_prev_mel = h_mel[:, i // 2, :].unsqueeze(1)  # [B, 1, H]
                    h = torch.cat([h, h_prev_mel], dim=1)  # [B, cur_len, H]
                    y.append(x_mel_gt[:, i // 2, :])
        else:
            for i in range(0, max_seq_len):
                # if i % 10 == 0:
                #     print('Sampling', i, '/', max_seq_len)
                h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
                y_next = self.local_sampling(h_out[:, -1], max_subseq_len=subseq_len, temperature=temperature)
                y.append(y_next)
                b, s, l = y_next.unsqueeze(1).shape
                if i % 2 == 0:
                    token_type_ids = torch.ones((b, s, l + 1), dtype=torch.long, device=y_next.device)
                else:
                    token_type_ids = torch.zeros((b, s, l + 1), dtype=torch.long, device=y_next.device)
                h = torch.cat([h, self.local_encode(y_next.unsqueeze(1), token_type_ids=token_type_ids)[0].unsqueeze(1)], dim=1)
        return y

    def global_sampling_from_scratch(self, x_mel: torch.LongTensor, temperature: float = 1.0, max_seq_len=384):
        B, S, L = x_mel.shape
        device = x_mel.device

        # Build program IDs = 0 for all melody tokens
        # token_type_ids = torch.zeros_like(x_mel, dtype=torch.long)  # [B, S, L]
        h_mel, _ = self.local_encode(x_mel, torch.zeros((B, S, L + 1), device=device, dtype=x_mel.dtype))
        h_mel = h_mel.view(B, S, self.hidden_size)  # [B, S, H]

        # Prepare SOS for global
        sos = self.global_sos.view(1, 1, -1).repeat(B, 1, 1)  # [B, 1, H]

        # Will store generated accompaniment frames
        y = []  # each entry: [B, L]

        # Start with just [SOS]
        h = sos  # [B, 1, H]
        for t in range(max_seq_len):
            # if t % 10 == 0:
            #         print('Sampling', t, '/', max_seq_len)
            if t > 0:
                # Append previous melody summary before generating new accompaniment
                h_prev_mel = h_mel[:, t - 1, :].unsqueeze(1)  # [B, 1, H]
                h = torch.cat([h, h_prev_mel], dim=1)  # [B, cur_len, H]
                y.append(x_mel[:, t - 1, :])

            h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
            y_next = self.local_sampling(h_out[:, -1], max_subseq_len=L, temperature=temperature)
            y.append(y_next)
            b, s, l = y_next.unsqueeze(1).shape
            token_type_ids = torch.ones((b, s, l + 1), dtype=torch.long, device=y_next.device)
            h = torch.cat([h, self.local_encode(y_next.unsqueeze(1), token_type_ids=token_type_ids)[0].unsqueeze(1)], dim=1)
        return y  # list of S tensors [B, L]

    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if self._future_mask.size(0) == 0 or (not self._future_mask.device == tensor.device) or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def forward(self, x):
        # x: [batch, seq, subseq]
        # Use local encoder to encode subsequences
        torch.cuda.memory._record_memory_history()  # tool for GPU memory
        batch_size, seq_len, subseq_len = x.shape  # 10*384*8
        assert seq_len % 2 == 0, "Expected even number of frames (2*S interleaved)."

        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()  # → [seq_len], 1 at even idx (acc), 0 at odd idx (mel)
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)
        h, emb = self.local_encode(x, token_type_ids)
        h = h.view(batch_size, seq_len, -1)  # 这是每一帧的summary

        # Prepend SOS token and remove the last token
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)

        # print(h.shape)
        h = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[
            0
        ]  ##all the sos of every timestep (considering other timestep)
        return self.local_decode(h, emb)

    def preprocess(
        self,
        x: torch.LongTensor,  # melody
        pitch_shift: torch.LongTensor,
        y: Optional[torch.LongTensor] = None,  # accompaniment
    ):
        batch_size, seq_length, subseq_length = x.shape
        x = x.long().view(batch_size, seq_length, subseq_length // 3, 3)
        x_processed = torch.zeros(batch_size, seq_length, subseq_length // 3, 2, dtype=torch.long, device=x.device)
        pad_indices = x[:, :, :, 1] == 255  # pitch is 255 that need to be pad
        eos_indices = x[:, :, :, 0] == 254  # program is 254
        is_not_drum = x[:, :, :, 0] != 127
        x_processed[:, :, :, 0] = 0  # program 不变
        # print(f"x:{x[:, :, :, 2].shape} pitch_shift: {pitch_shift[:,None].shape},is not drum: {is_not_drum.shape}")
        x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2]) * 128 + 2 + pitch_shift[:, None] * is_not_drum
        x_processed[pad_indices] = PAD_TOKEN
        x_processed[:, :, :, 0][eos_indices] = EOS_TOKEN

        if y == None:
            return x_processed.view(batch_size, seq_length, subseq_length // 3 * 2)
        else:
            batch_size_y, seq_length_y, subseq_length_y = y.shape
            y = y.long().view(batch_size_y, seq_length_y, subseq_length_y // 3, 3)
            y_processed = torch.zeros(batch_size_y, seq_length_y, subseq_length_y // 3, 2, dtype=torch.long, device=y.device)
            pad_indices_y = y[:, :, :, 1] == 255  # pitch is 255 that need to be pad
            eos_indices_y = y[:, :, :, 0] == 254  # program is 254
            is_not_drum_y = y[:, :, :, 0] != 127
            y_processed[:, :, :, 0] = 1  # program 不变
            y_processed[:, :, :, 1] = y[:, :, :, 1] + (y[:, :, :, 2]) * 128 + 2 + pitch_shift[:, None] * is_not_drum_y
            y_processed[pad_indices_y] = PAD_TOKEN
            y_processed[:, :, :, 0][eos_indices_y] = EOS_TOKEN

            return x_processed.view(batch_size, seq_length, subseq_length // 3 * 2), y_processed.view(
                batch_size_y, seq_length_y, subseq_length_y // 3 * 2
            )

    def loss(self, x_mel, x_acc, pitch_shift):
        # x_mel, x_acc = self.preprocess(x_mel, pitch_shift, y = x_acc)
        x_mel, x_acc = self.preprocess(x_mel, pitch_shift, y=x_acc)
        batch_size, seq_len, subseq_len = x_mel.shape  # 10*384*8
        stacked = torch.stack([x_acc, x_mel], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)

        x_target = x.clone()
        # build a mask: True at every odd timestep
        idx = torch.arange(seq_len * 2, device=x.device)
        mel_mask = (idx % 2 == 1).unsqueeze(0).unsqueeze(-1)  # [1, 2*S, 1]
        mel_mask = mel_mask.expand(batch_size, seq_len * 2, subseq_len)  # [B, 2*S, L]
        x_target[mel_mask] = PAD_TOKEN

        y = self(x)

        return F.cross_entropy(y.view(-1, N_TOKENS), x_target.view(-1), ignore_index=PAD_TOKEN)

    def training_step(self, batch: FnsM2ARoformerInput, batch_idx):
        batch = self._move_to_device(batch)
        x_mel, x_acc, pitch_shift = batch.mel_data, batch.acc_data, batch.pitch_shift
        loss = self.loss(x_mel, x_acc, pitch_shift)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # scheduler step
        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log("training/lr", scheduler.get_last_lr()[0], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: FnsM2ARoformerInput, batch_idx):
        batch = self._move_to_device(batch)
        x_mel, x_acc, pitch_shift = batch.mel_data, batch.acc_data, batch.pitch_shift
        loss = self.loss(x_mel, x_acc, pitch_shift)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def _move_to_device(self, batch: FnsM2ARoformerInput) -> FnsM2ARoformerInput:
        """
        Move the batch data to the appropriate device.
        Args:
            batch (M2AModelInputData): The input batch data.
        Returns:
            M2AModelInputData: The batch data moved to the appropriate device.
        """
        return FnsM2ARoformerInput(
            token_ids=batch.token_ids.to(self.device),
            pitch_shift=batch.pitch_shift.to(self.device),
        )

    def token_encode(self, score: Score) -> torch.Tensor:
        return self.tokenizer.encode(score)

    def token_decode(self, token_ids: torch.Tensor) -> Score:
        return self.tokenizer.decode(token_ids)

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

            # Find all indices of 'Frame_None' tokens.
            # 'Frame_None' indicates the start of a frame.
            frame_start_indices = (single_sequence == frame_none_id).nonzero(as_tuple=True)[0]

            if frame_start_indices.numel() == 0:
                # If no Frame_None tokens, treat the entire non-padded sequence as one frame
                # This frame will either contain actual tokens or be a full padding frame.
                non_pad_tokens = single_sequence[single_sequence != pad_token_id]

                if non_pad_tokens.numel() > 0:
                    # Truncate or pad to target_sub_seq_len
                    if non_pad_tokens.shape[0] > target_sub_seq_len:
                        frames_in_sequence.append(non_pad_tokens[:target_sub_seq_len].to(device))
                    else:
                        padded_frame = F.pad(non_pad_tokens, (0, target_sub_seq_len - non_pad_tokens.shape[0]), value=pad_token_id)
                        frames_in_sequence.append(padded_frame.to(device))
                else:
                    # If the sequence is empty or only contains pad tokens, add a full padding frame
                    full_pad_frame = torch.full((target_sub_seq_len,), fill_value=pad_token_id, dtype=torch.long, device=device)
                    frames_in_sequence.append(full_pad_frame)
            else:
                current_frame_indices = frame_start_indices.tolist()

                for j in range(len(current_frame_indices)):
                    start_idx = current_frame_indices[j]
                    end_idx = current_frame_indices[j + 1] if j + 1 < len(current_frame_indices) else single_sequence.shape[0]

                    current_frame_segment = single_sequence[start_idx:end_idx]

                    # Remove trailing PAD_TOKENs and the frame_none_id itself from the current segment
                    current_frame_content = current_frame_segment[(current_frame_segment != pad_token_id) & (current_frame_segment != frame_none_id)]

                    if current_frame_content.numel() > 0:
                        # Truncate or pad current_frame_content to target_sub_seq_len
                        if current_frame_content.shape[0] > target_sub_seq_len:
                            # Truncate if too long
                            frames_in_sequence.append(current_frame_content[:target_sub_seq_len].to(device))
                        else:
                            # Pad if too short (or already correct length)
                            padded_frame = F.pad(current_frame_content, (0, target_sub_seq_len - current_frame_content.shape[0]), value=pad_token_id)
                            frames_in_sequence.append(padded_frame.to(device))
                    else:
                        # If the frame segment is empty after removing special tokens, append a full padding frame
                        full_pad_frame = torch.full((target_sub_seq_len,), fill_value=pad_token_id, dtype=torch.long, device=device)
                        frames_in_sequence.append(full_pad_frame)

            # Update the maximum number of frames in the current batch
            if len(frames_in_sequence) > max_seq_len_in_batch:
                max_seq_len_in_batch = len(frames_in_sequence)

            processed_batches.append(frames_in_sequence)

        # Pad the sequences of frames to the final tensor shape:
        # (batch_size, max_seq_len_in_batch, target_sub_seq_len)
        output_tensor = torch.full((batch_size, max_seq_len_in_batch, target_sub_seq_len), fill_value=pad_token_id, dtype=torch.long, device=device)

        for i, frames_for_this_seq in enumerate(processed_batches):
            for j, frame_tensor in enumerate(frames_for_this_seq):
                # 'frame_tensor' here is already padded or truncated to target_sub_seq_len
                output_tensor[i, j, :] = frame_tensor

        return output_tensor

    def preprocess_with_program_id(self, token_ids: torch.Tensor, program_token_id: int) -> torch.Tensor:
        """
        Segments token_ids into multiple frames based on 'Frame_None' tokens.
        Each frame is then processed: filtered by program_token_id, and padded/truncated
        to a (fixed_sub_seq_len) shape.

        Crucially, if a frame, after filtering, has no relevant tokens, it will
        still be represented by a full padding frame of fixed_sub_seq_len.

        Args:
            token_ids (torch.Tensor): Input token IDs, with shape (batch_size, total_tokens).
            program_token_id (int): The program ID to filter for.

        Returns:
            list[torch.Tensor]: A list of processed token IDs, where each tensor in the list
                                has shape (num_frames_in_sequence, fixed_sub_seq_len).
                                Each tensor now explicitly includes fully padded frames for
                                segments that contained no matching program_id tokens.
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
                # If no Frame_None tokens, treat the entire non-padded sequence as one logical "frame"
                filtered_sequence_tokens = []
                non_pad_tokens = single_sequence[single_sequence != pad_token_id]

                for k in range(0, non_pad_tokens.numel(), 3):
                    if k + 2 < non_pad_tokens.numel():  # Ensure enough tokens for a triplet
                        current_program_id = non_pad_tokens[k].item()
                        if current_program_id == program_token_id:
                            # Append program, pitch, and duration
                            filtered_sequence_tokens.append(non_pad_tokens[k])
                            filtered_sequence_tokens.append(non_pad_tokens[k + 1])
                            filtered_sequence_tokens.append(non_pad_tokens[k + 2])

                if len(filtered_sequence_tokens) > 0:
                    filtered_tensor = torch.stack(filtered_sequence_tokens).to(device)
                    # Truncate or pad to target_sub_seq_len
                    if filtered_tensor.shape[0] > target_sub_seq_len:
                        frames_for_current_sequence.append(filtered_tensor[:target_sub_seq_len])
                    else:
                        padded_frame = F.pad(filtered_tensor, (0, target_sub_seq_len - filtered_tensor.shape[0]), value=pad_token_id)
                        frames_for_current_sequence.append(padded_frame)
                else:
                    # If no tokens after filtering, append a full padding frame
                    full_pad_frame = torch.full((target_sub_seq_len,), fill_value=pad_token_id, dtype=torch.long, device=device)
                    frames_for_current_sequence.append(full_pad_frame)

            else:
                current_frame_indices = frame_start_indices.tolist()

                for j in range(len(current_frame_indices)):
                    start_idx = current_frame_indices[j]
                    end_idx = current_frame_indices[j + 1] if j + 1 < len(current_frame_indices) else single_sequence.shape[0]

                    current_frame_segment = single_sequence[start_idx:end_idx]

                    # Remove trailing PAD_TOKENs and frame_none_id for filtering purposes
                    current_frame_non_padded = current_frame_segment[
                        (current_frame_segment != pad_token_id) & (current_frame_segment != frame_none_id)
                    ]

                    filtered_frame_tokens = []
                    for k in range(0, current_frame_non_padded.numel(), 3):
                        if k + 2 < current_frame_non_padded.numel():  # Ensure enough tokens for a triplet
                            current_program_id = current_frame_non_padded[k].item()
                            if current_program_id == program_token_id:
                                # Append program, pitch, and duration
                                filtered_frame_tokens.append(current_frame_non_padded[k])
                                filtered_frame_tokens.append(current_frame_non_padded[k + 1])
                                filtered_frame_tokens.append(current_frame_non_padded[k + 2])

                    if len(filtered_frame_tokens) > 0:
                        filtered_tensor_frame = torch.stack(filtered_frame_tokens).to(device)
                        # Truncate or pad filtered_tensor_frame to target_sub_seq_len
                        if filtered_tensor_frame.shape[0] > target_sub_seq_len:
                            frames_for_current_sequence.append(filtered_tensor_frame[:target_sub_seq_len])
                        else:
                            padded_frame = F.pad(filtered_tensor_frame, (0, target_sub_seq_len - filtered_tensor_frame.shape[0]), value=pad_token_id)
                            frames_for_current_sequence.append(padded_frame)
                    else:
                        # If no tokens after filtering for this frame segment, append a full padding frame
                        full_pad_frame = torch.full((target_sub_seq_len,), fill_value=pad_token_id, dtype=torch.long, device=device)
                        frames_for_current_sequence.append(full_pad_frame)

            # Stack all collected frames (including full padding ones) for the current sequence
            if frames_for_current_sequence:
                output_sequences.append(torch.stack(frames_for_current_sequence))
            else:
                # This case handles an input sequence that was entirely empty or only contained pad_tokens
                # It results in no logical frames, even padding ones.
                # If you want even an empty input to result in one full_pad_frame,
                # you'd need to add `frames_for_current_sequence.append(full_pad_frame)` here.
                # For now, if no frames were generated at all, we append an empty tensor.
                output_sequences.append(torch.empty(0, target_sub_seq_len, dtype=torch.long, device=device))

        return torch.stack(output_sequences, dim=0)

    def postprocess(
        self,
    ): ...


if __name__ == "__main__":
    from src.model.specific_model.fns_m2a_roformer.config import FnsM2ARoformerConfig

    model_config = FnsM2ARoformerConfig()
    model = FnsM2ATransformer(model_config)
    # print(model.tokenizer.vocab)
    # print(model.tokenizer.pitch_num)
    # print(model.tokenizer.duration_num)
    x=model.tokenizer.encode("datasets/Seperated-POP909-Dataset/original/001.mid")[499:600]
    x= torch.tensor(x.ids)
    x= torch.stack([x,x],dim=0)
    print(model.preprocess_with_all_program(x).shape)
    print(model.preprocess_with_program_id(x, program_token_id=model.mel_program_id).shape)
    print(model.preprocess_with_program_id(x, program_token_id=model.acc_program_id).shape)
    # print(x.ids)
    # score =model.tokenizer.decode(x.ids)
    # score.dump_midi("z.mid")
