import hashlib
import random
from collections import deque
from typing import Tuple, Union


class ExperienceReplay:
    def __init__(self, capacity: Union[int, float]):
        self.data = deque(maxlen=int(capacity))
        self.registry = {}

    def add(self, experience) -> None:
        state = experience[0]
        hstate = hashlib.md5(str(state).encode("utf-8")).hexdigest()
        if self.registry.get(hstate) is None:
            self.data.append(experience)
            self.registry[hstate] = 1

    def sample(self, batch_size: int) -> Tuple:
        return random.sample(self.data, k=batch_size)

    def __len__(self) -> int:
        return len(self.data)

