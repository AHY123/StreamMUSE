# Simple Reward Models

This directory contains a simplified implementation of discriminative reward models for StreamMUSE, designed to be standalone and easy to debug.

## Files

- `discriminative_model.py` - The reward model architecture (hierarchical like main model)
- `data_loader.py` - Data loading and preprocessing 
- `train_discriminative.py` - Simple training script
- `README.md` - This file

## Architecture

The discriminative model follows the same hierarchical structure as the main StreamMUSE model:

1. **Input**: Interleaved melody-accompaniment sequences `[batch, 2*seq_len, 12]`
2. **Preprocessing**: Same tokenization as main model `[12] -> [8]` 
3. **Local Encoder**: Process individual polyphonic frames
4. **Global Encoder**: Process sequence of frame representations
5. **Classification Head**: Binary real/fake prediction

## Usage

```bash
# Train the model
python train_discriminative.py \
    --melody_path /path/to/reward_training_mel_cp4.pt \
    --acc_path /path/to/reward_training_acc_cp4.pt \
    --output_dir ./checkpoints \
    --batch_size 8 \
    --epochs 50 \
    --target_length 384
```

## Data Format

The script expects:
- `reward_training_mel_cp4.pt` - Melody sequences `[num_sequences, seq_len, 12]`
- `reward_training_acc_cp4.pt` - Accompaniment sequences `[num_sequences, seq_len, 12]`  
- `reward_training_mel_cp4.length.pt` - Sequence lengths for melody
- `reward_training_acc_cp4.length.pt` - Sequence lengths for accompaniment

## Training Strategy

- **Real pairs**: melody[i] + accompaniment[i] (same song)
- **Fake pairs**: melody[i] + accompaniment[j] (different songs)
- **Ratio**: 1:1 real to fake samples
- **Interleaving**: `[acc_0, mel_0, acc_1, mel_1, ...]`
- **Loss**: Binary cross-entropy

## Key Design Decisions

1. **Hierarchical Architecture**: Reuses the local+global encoder pattern from main model
2. **Same Preprocessing**: Identical tokenization to ensure compatibility
3. **Token Type Embeddings**: Distinguishes melody vs accompaniment frames
4. **Interleaved Input**: Follows main model's expectation of alternating frames
5. **Simple Training Loop**: Standalone script without complex infrastructure

This approach should avoid the token indexing issues we encountered by following the exact same data flow as the working main model.