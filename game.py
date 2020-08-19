import random
from board import Board

if __name__ == "__main__":
    b = Board()
    moves = ["UP", "DOWN", "RIGHT", "LEFT"]
    for _ in range(1000):
        moved = b.move(random.choice(moves))

        if moved and not b.is_game_over():
            b.fill_cell()
            continue

        if b.is_game_over():
            print("GAME OVER!")
            print("score: ", b.score)
            print(b.board)
            break
