import random
from tqdm import tqdm

from board import Board

if __name__ == "__main__":
    b = Board()
    moves = ["UP", "DOWN", "RIGHT", "LEFT"]
    num_moves = 0
    while not b.is_game_over():
        moved = b.move(random.choice(moves))

        if moved and not b.is_game_over():
            num_moves += 1
            b.fill_cell()

    print("GAME OVER!")
    print("score: ", b.score)
    print("num moves: ", num_moves)
    print(b.board)
