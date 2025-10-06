#!/usr/bin/env python3
"""
Create a small subset of the real training data to test scaling up gradually.
"""

import torch

# Load real training data
real_mel = torch.load('../data/reward_training_mel_cp4.pt', mmap=True)
real_acc = torch.load('../data/reward_training_acc_cp4.pt', mmap=True)
real_mel_len = torch.load('../data/reward_training_mel_cp4.length.pt', mmap=True)
real_acc_len = torch.load('../data/reward_training_acc_cp4.length.pt', mmap=True)
real_mel_pitch = torch.load('../data/reward_training_mel_cp4.pitch_shift_range.pt', mmap=True)
real_acc_pitch = torch.load('../data/reward_training_acc_cp4.pitch_shift_range.pt', mmap=True)

print(f'Loaded {len(real_mel_len)} songs')

# Take first 20 songs to create a small subset
num_subset_songs = 20

subset_mel_len = real_mel_len[:num_subset_songs]
subset_acc_len = real_acc_len[:num_subset_songs]
subset_mel_pitch = real_mel_pitch[:num_subset_songs]
subset_acc_pitch = real_acc_pitch[:num_subset_songs]

# Calculate the frames needed for first 20 songs
mel_frames_needed = subset_mel_len.sum().item()
acc_frames_needed = subset_acc_len.sum().item()

print(f'Subset needs {mel_frames_needed} mel frames, {acc_frames_needed} acc frames')

# Extract the data for first 20 songs
subset_mel_data = real_mel[:mel_frames_needed]
subset_acc_data = real_acc[:acc_frames_needed]

print(f'Extracted mel: {subset_mel_data.shape}, acc: {subset_acc_data.shape}')

# Save subset
torch.save(subset_mel_data, 'debug_data/subset20_mel.pt')
torch.save(subset_acc_data, 'debug_data/subset20_acc.pt')
torch.save(subset_mel_len, 'debug_data/subset20_mel.length.pt')
torch.save(subset_acc_len, 'debug_data/subset20_acc.length.pt')
torch.save(subset_mel_pitch, 'debug_data/subset20_mel.pitch_shift_range.pt')
torch.save(subset_acc_pitch, 'debug_data/subset20_acc.pitch_shift_range.pt')

print('Saved 20-song subset to debug_data/subset20_*')
print(f'This creates {num_subset_songs * 2} training samples (20 real + 20 fake)')