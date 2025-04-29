from abc import ABC, abstractmethod
import numpy as np


class VectorLoader(ABC):
    @abstractmethod
    def load(self) -> (np.array, list[int]):
        """Load vectors into an (n×d) array and return the list of IDs."""
        pass