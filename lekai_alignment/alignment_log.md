# Lekai Alignment Log

## ... (Previous Runs)

## Run 006: Acc Prompt Injection
- **Issue**: User reported 1-beat gap after prompt.
    - **Fix (Run 006_gap_fix)**: Inject 4 Bar tokens at boundaries.
- **Issue 2**: User reported "Abnormally short notes" generated after prompt (prompt has long notes).
    - **Root Cause**: `inject_context_beat` was NOT updating `self._active_acc_pitches`. This meant the decoder (`pianoroll_to_events`) thought NO notes were active at the start of generation. If the model generated "Sustain" tokens (continuing prompt notes), the decoder ignored them or failed to link them, effectively cutting the notes off.
    - **Fix (Run 006_active_pitch_fix)**: Added logic to `inject_context_beat` to scan history and populate `self._active_acc_pitches` with notes that sustain past the injected beat.
    - **Status**: Running `run_006_active_pitch_fix`.
