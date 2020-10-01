import random
import math
import numpy as np
from tqdm import tqdm
from game import Game

from policy_network import PolicyNetwork
from experience_replay import ExperienceReplay
from logger import Logger


ACTIONS = {0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}
NUM_ACTIONS = len(ACTIONS)

NUM_GAMES = 30000
OBSERVE = 1000
MAX_TILE = 2048

epsilon = 0.1
min_epsilon = 1e-2
gamma_epsilon = 0.999
gamma_reward = 0.99

replay = ExperienceReplay(capacity=1e6)
logger = Logger()

online = PolicyNetwork(batch_size=32)
target = PolicyNetwork(batch_size=32)


def preprocess(a: np.array) -> np.array:
    a = np.where(a <= 0, 1, a)
    a = np.log2(a) / np.log2(MAX_TILE)
    return a


if __name__ == "__main__":

    best_score = 0
    best_board = None
    best_iteration = 0

    bar = tqdm(range(NUM_GAMES), desc=f"best score: {best_score}, last score: 0")
    for i in bar:

        g = Game()

        num_moves = 0
        num_random_moves = 0

        if i % 10:
            target.model.set_weights(online.model.get_weights())

        if i > OBSERVE:
            epsilon = epsilon * gamma_epsilon if epsilon > min_epsilon else min_epsilon

        moved = True
        while not g.is_game_over():

            state = g.state
            s = preprocess(state)
            s = np.expand_dims(state, axis=0)  # batch axis
            s = np.expand_dims(s, axis=3)  # channel axis

            if moved:  # all actions are OK
                mask = np.ones(NUM_ACTIONS)

            if i < OBSERVE or random.random() < epsilon:
                action = np.argmax(np.random.randn(NUM_ACTIONS) * mask)
                num_random_moves += 1
            else:
                policy = online.predict(s)
                action = np.argmax(policy * mask)

            moved, reward, num_merges = g.move(ACTIONS[action])
            num_moves += 1

            if not moved:
                mask[action] = 0

            if moved and not g.is_game_over():
                g.fill_cell()
                next_state = preprocess(g.state)
                next_state = np.expand_dims(next_state, axis=2)  # channel axis

                action_onehot = np.zeros(NUM_ACTIONS)
                action_onehot[action] = 1

                state = np.expand_dims(state, axis=2)

                if reward > 0:
                    extra_bonus = 0
                    if np.max(state) == state[0, 0]:
                        extra_bonus += math.log2(2 ** 20)
                        if np.argmax(np.sum(state, axis=1)):
                            extra_bonus += math.log2(2 ** 20)
                    reward = math.log2(reward) + num_merges + extra_bonus
                replay.add((state, action_onehot, reward, next_state))

            if g.is_game_over():
                logger.log("stats/score", g.score, i)
                logger.log("stats/num_moves", num_moves, i)
                logger.log("stats/max_tile", np.max(g.state), i)
                logger.log("stats/best_score", best_score, i)

                logger.log("settings/epsilon", epsilon, i)
                logger.log("settings/num_random_moves", num_random_moves, i)
                logger.log(
                    "settings/perc_random_moves", num_random_moves / num_moves, i
                )
                logger.log("settings/experience", len(replay), i)

                reward = 0
                replay.add((state, action_onehot, reward, np.zeros(state.shape)))

            if i > OBSERVE:

                batch = replay.sample(batch_size=32)

                states = []
                actions = []
                rewards = []
                next_states = []
                for e, b in enumerate(batch):
                    states.append(b[0])
                    actions.append(b[1])
                    rewards.append(b[2])
                    next_states.append(b[3])
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)

                future_action = np.argmax(online.predict(next_states), axis=1)
                future_rewards = target.predict(next_states)[future_action]
                rewards = np.reshape(rewards, (-1, 1)) + future_rewards
                online.update(states, actions, rewards)

        if g.score >= best_score:
            best_score = g.score
            best_board = g.state
            best_iteration = i

            print("best score", best_score, "at game", best_iteration)
            print(best_board)

            online.save()
        
        bar.set_description(f"best score: {best_score}, last score: {g.score}")
        bar.refresh()

    net.save()
