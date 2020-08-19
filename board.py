import random
import numpy as np
from typing import Tuple


class Board:
    size = 4

    def __str__(self) -> str:
        return str(self.board)

    def __init__(self):
        self.score = 0
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.fill_cell()

    def is_game_over(self) -> bool:
        if len(np.argwhere(self.board == 0)) < 1:

            # look for adjacent equal numbers in rows
            for i in range(self.size):
                for j in range(self.size - 1):
                    if self.board[i, j] == self.board[i, j + 1]:
                        return False

            # look for adjacent equal numbers in cols
            for j in range(self.size):
                for i in range(self.size - 1):
                    if self.board[i, j] == self.board[i + 1, j]:
                        return False

            return True

        return False

    def get_random_empty_coordinate(self) -> np.array:
        ids = np.argwhere(self.board == 0)
        i = random.choice(ids)
        return i[0], i[1]

    def fill_cell(self) -> None:
        x, y = self.get_random_empty_coordinate()
        self.board[x, y] = 2

    def left(self) -> bool:
        for i in range(self.size):
            row = self.board[i, :].copy()
            self.board[i, :] = np.concatenate((row[row != 0], row[row == 0]))
            moved = not np.allclose(row, self.board[i, :])
            for j in range(self.size - 1, 0, -1):
                if self.board[i, j - 1] == 0:
                    self.board[i, j - 1] = self.board[i, j]
                    self.board[i, j] = 0
                    moved = True
                elif self.board[i, j - 1] == self.board[i, j]:
                    self.board[i, j - 1] = self.board[i, j] + self.board[i, j - 1]
                    self.score += self.board[i, j - 1]
                    self.board[i, j] = 0
                    moved = True
        return moved

    def right(self) -> bool:
        moved = False
        for i in range(self.size):
            row = self.board[i, :].copy()
            self.board[i, :] = np.concatenate((row[row == 0], row[row != 0]))
            moved = not np.allclose(row, self.board[i, :])
            for j in range(0, self.size - 1, 1):
                if self.board[i, j + 1] == 0:
                    self.board[i, j + 1] = self.board[i, j]
                    self.board[i, j] = 0
                    moved = True
                elif self.board[i, j + 1] == self.board[i, j]:
                    self.board[i, j + 1] = self.board[i, j] + self.board[i, j + 1]
                    self.score += self.board[i, j + 1]
                    self.board[i, j] = 0
                    moved = True
        return moved

    def up(self) -> bool:
        for j in range(self.size):
            col = self.board[:, j].copy()
            self.board[:, j] = np.concatenate((col[col != 0], col[col == 0]))
            moved = not np.allclose(col, self.board[:, j])

            for i in range(self.size - 1, 0, -1):
                if self.board[i - 1, j] == 0:
                    self.board[i - 1, j] = self.board[i, j]
                    self.board[i, j] = 0
                    moved = True
                elif self.board[i - 1, j] == self.board[i, j]:
                    self.board[i - 1, j] = self.board[i, j] + self.board[i - 1, j]
                    self.score += self.board[i - 1, j]
                    self.board[i, j] = 0
                    moved = True
        return moved

    def down(self) -> bool:
        for j in range(self.size):
            col = self.board[:, j].copy()
            self.board[:, j] = np.concatenate((col[col == 0], col[col != 0]))
            moved = not np.allclose(col, self.board[:, j])

            for i in range(0, self.size - 1, 1):
                if self.board[i + 1, j] == 0:
                    self.board[i + 1, j] = self.board[i, j]
                    self.board[i, j] = 0
                    moved = True
                elif self.board[i + 1, j] == self.board[i, j]:
                    self.board[i + 1, j] = self.board[i, j] + self.board[i + 1, j]
                    self.score += self.board[i + 1, j]
                    self.board[i, j] = 0
                    moved = True
        return moved

    def move(self, direction: str) -> bool:
        if direction == "UP":
            return self.up()
        elif direction == "DOWN":
            return self.down()
        elif direction == "RIGHT":
            return self.right()
        elif direction == "LEFT":
            return self.left()


