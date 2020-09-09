import random
import numpy as np
from typing import Tuple, List


class Game:
    size = 4

    def __str__(self) -> str:
        return str(self.state)

    def __init__(self):
        self.score = 0
        self.state = np.zeros((self.size, self.size), dtype=np.int32)
        self.fill_cell()

    def is_game_over(self) -> bool:
        if len(np.argwhere(self.state == 0)) < 1:

            # look for adjacent equal numbers in rows
            for i in range(self.size):
                for j in range(self.size - 1):
                    if self.state[i, j] == self.state[i, j + 1]:
                        return False

            # look for adjacent equal numbers in cols
            for j in range(self.size):
                for i in range(self.size - 1):
                    if self.state[i, j] == self.state[i + 1, j]:
                        return False

            return True

        return False

    def get_random_empty_coordinate(self) -> np.array:
        ids = np.argwhere(self.state == 0)
        i = random.choice(ids)
        return i[0], i[1]

    def fill_cell(self) -> None:
        x, y = self.get_random_empty_coordinate()
        if random.random() < 0.9:
            self.state[x, y] = 2
        else:
            self.state[x, y] = 4

    def left(self) -> Tuple[bool, int, int]:
        skip = False
        moved = False
        reward = 0
        nonzero_before = np.count_nonzero(self.state)

        for i in range(self.size):
            row = self.state[i, :].copy()
            self.state[i, :] = np.concatenate((row[row != 0], row[row == 0]))
            if not moved:
                moved = not np.allclose(row, self.state[i, :])
            for j in range(self.size - 1, 0, -1):
                if skip:
                    skip = False
                    continue
                if self.state[i, j - 1] == 0:
                    self.state[i, j - 1] = self.state[i, j]
                    self.state[i, j] = 0
                    moved = True
                elif self.state[i, j - 1] == self.state[i, j]:
                    self.state[i, j - 1] = self.state[i, j] + self.state[i, j - 1]
                    reward += self.state[i, j - 1]
                    self.state[i, j] = 0
                    skip = True
                    moved = True
            row = self.state[i, :].copy()
            self.state[i, :] = np.concatenate((row[row != 0], row[row == 0]))
        self.score += reward
        nonzero_after = np.count_nonzero(self.state)
        num_merges = nonzero_before - nonzero_after

        return moved, reward, num_merges

    def right(self) -> Tuple[bool, int, int]:
        skip = False
        moved = False
        reward = 0
        nonzero_before = np.count_nonzero(self.state)

        for i in range(self.size):
            row = self.state[i, :].copy()
            self.state[i, :] = np.concatenate((row[row == 0], row[row != 0]))
            if not moved:
                moved = not np.allclose(row, self.state[i, :])

            for j in range(0, self.size - 1, 1):
                if skip:
                    skip = False
                    continue
                if self.state[i, j + 1] == 0:
                    self.state[i, j + 1] = self.state[i, j]
                    self.state[i, j] = 0
                    moved = True
                elif self.state[i, j + 1] == self.state[i, j]:
                    self.state[i, j + 1] = self.state[i, j] + self.state[i, j + 1]
                    reward += self.state[i, j + 1]
                    self.state[i, j] = 0
                    skip = True
                    moved = True
            row = self.state[i, :].copy()
            self.state[i, :] = np.concatenate((row[row == 0], row[row != 0]))

        self.score += reward
        nonzero_after = np.count_nonzero(self.state)
        num_merges = nonzero_before - nonzero_after
        return moved, reward, num_merges

    def up(self) -> Tuple[bool, int, int]:
        skip = False
        moved = False
        reward = 0
        nonzero_before = np.count_nonzero(self.state)

        for j in range(self.size):
            col = self.state[:, j].copy()
            self.state[:, j] = np.concatenate((col[col != 0], col[col == 0]))
            if not moved:
                moved = not np.allclose(col, self.state[:, j])

            for i in range(self.size - 1, 0, -1):
                if skip:
                    skip = False
                    continue
                if self.state[i - 1, j] == 0:
                    self.state[i - 1, j] = self.state[i, j]
                    self.state[i, j] = 0
                    moved = True
                elif self.state[i - 1, j] == self.state[i, j]:
                    self.state[i - 1, j] = self.state[i, j] + self.state[i - 1, j]
                    reward += self.state[i - 1, j]
                    self.state[i, j] = 0
                    skip = True
                    moved = True
            col = self.state[:, j].copy()
            self.state[:, j] = np.concatenate((col[col != 0], col[col == 0]))

        self.score += reward
        nonzero_after = np.count_nonzero(self.state)
        num_merges = nonzero_before - nonzero_after
        return moved, reward, num_merges

    def down(self) -> Tuple[bool, int, int]:
        skip = False
        moved = False
        reward = 0
        nonzero_before = np.count_nonzero(self.state)

        for j in range(self.size):
            col = self.state[:, j].copy()
            self.state[:, j] = np.concatenate((col[col == 0], col[col != 0]))
            if not moved:
                moved = not np.allclose(col, self.state[:, j])

            for i in range(0, self.size - 1, 1):
                if skip:
                    skip = False
                    continue
                if self.state[i + 1, j] == 0:
                    self.state[i + 1, j] = self.state[i, j]
                    self.state[i, j] = 0
                    moved = True
                elif self.state[i + 1, j] == self.state[i, j]:
                    self.state[i + 1, j] = self.state[i, j] + self.state[i + 1, j]
                    reward += self.state[i + 1, j]
                    self.state[i, j] = 0
                    skip = True
                    moved = True
            col = self.state[:, j].copy()
            self.state[:, j] = np.concatenate((col[col == 0], col[col != 0]))

        self.score += reward
        nonzero_after = np.count_nonzero(self.state)
        num_merges = nonzero_before - nonzero_after
        return moved, reward, num_merges

    def move(self, direction: str) -> Tuple[bool, int, int]:
        if direction == "UP":
            return self.up()
        elif direction == "DOWN":
            return self.down()
        elif direction == "RIGHT":
            return self.right()
        elif direction == "LEFT":
            return self.left()

