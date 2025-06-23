class BaselineEngine:
    """
    A placeholder inference engine that simulates model behavior.
    It maintains a history and generates a simple, rule-based accompaniment.
    """
    def __init__(self, max_history=32):
        self.history = []
        self.max_history = max_history
        print("\rInitialized Baseline Inference Engine.")

    def generate_accompaniment(self, user_note_event):
        """
        Takes a user note event, updates history, and returns an accompaniment.
        
        Args:
            user_note_event (dict): The event from the input handler.
                                    e.g., {'type': 'note_on', 'pitch': 60, 'key': 'a'}

        Returns:
            list: A list of generated MIDI note numbers. e.g., [72]
        """
        # A real engine would do complex tokenization here.
        user_note = user_note_event["pitch"]
        self.history.append(user_note)

        # --- Placeholder Inference Logic ---
        # A real model would take the entire `self.history` as input.
        # We'll just generate a note an octave (+12 semitones) higher.
        accompaniment_note = user_note + 12
        # ---------------------------------

        # The model should be aware of its own output for the next turn.
        self.history.append(accompaniment_note)

        # Enforce the sliding context window.
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return [accompaniment_note]