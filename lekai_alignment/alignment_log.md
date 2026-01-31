# Lekai Alignment Log

## ... (Previous Runs)

## Run 006: Acc Prompt Injection
- **Issue**: User reported 1-beat gap after prompt.
- **Root Cause**:
    - My engine was injecting only 2 Bar tokens at Measure boundaries.
    - `inference_v2` logic implies BOTH `part0` (Mel) and `part1` (Acc) sequences contain Bar tokens.
    - In Stagger Mode (interleaved), this means we need `Acc_Bar` (2 tokens) AND `Mel_Bar` (2 tokens) to bridge the measure.
    - With only 2 tokens, the model was essentially seeing `Acc_Bar` but missing `Mel_Bar`, leading it to predict `Mel_Bar` (or silence) instead of `Acc` Notes.
- **Fix (Run 006_gap_fix)**:
    - Updated `generate_accompaniment` and `inject_context_beat` to inject **4 Bar Tokens** (255, 255, 255, 255) at Measure Boundaries in Stagger Mode.
    - Order: `Mel[t-1]` -> `Bar x4` -> `Acc[t]`.
- **Status**: Running `run_006_gap_fix`.
