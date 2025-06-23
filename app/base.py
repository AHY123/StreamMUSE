from abc import ABC, abstractmethod

class InputHandler(ABC):
    @abstractmethod
    def read(self):
        """Read input and return a note or event."""
        pass