import numpy as np
import multiprocessing
import sys
import os
import time
import datetime
import csv  # Added for CSV logging
from copy import deepcopy
import glob
import random


sys.path.append(os.getcwd())
from trainable_agent import TrainableAgent
from agents.student_agent import StudentAgent as BaseStudent


def execute_move(board, move, player):
    (r_src, c_src), (r_dest, c_dest) = move
    board[r_dest, c_dest] = player
    if abs(r_dest - r_src) > 1 or abs(c_dest - c_src) > 1:
        board[r_src, c_src] = 0

    r_min, r_max = max(0, r_dest - 1), min(7, r_dest + 2)
    c_min, c_max = max(0, c_dest - 1), min(7, c_dest + 2)
    opponent = 3 - player
    window = board[r_min:r_max, c_min:c_max]
    mask = window == opponent
    window[mask] = player


def check_endgame(board):
    p1 = np.count_nonzero(board == 1)
    p2 = np.count_nonzero(board == 2)
    empty = np.count_nonzero(board == 0)
    if p1 == 0 or p2 == 0 or empty == 0:
        return True, p1, p2
    return False, p1, p2


GENERATIONS = 1000
POPULATION_SIZE = 12
LOG_FILE = "training_log.txt"
CSV_FILE = "training_history.csv"  # File for your CSV logs
BEST_WEIGHTS_FILE = "best_weights_so_far.txt"
BOUNDS = [
    (160.0, 190.0),  # Material (Focus around 177)
    (120.0, 160.0),  # Quads (Focus around 140)
    (0.0, 15.0),  # Mobility (Focus around 6.8)
    (40.0, 70.0),  # Corners (Focus around 53)
    (15.0, 35.0),  # Edges (Focus around 22)
    (-35.0, -15.0),  # Center (Focus around -22)
]
CURRENT_KING_WEIGHTS = [179.39, 144.18, 5.42, 53.68, 24.02, -16.37]


def log(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")


def save_best_weights(weights, score, gen):
    with open(BEST_WEIGHTS_FILE, "w") as f:
        f.write(f"# Gen {gen} (Score vs King: {score:.2f})\n")
        f.write(f"WEIGHTS = {list(weights)}\n")


def log_csv(gen, score, weights):
    """Logs the best individual of the generation to CSV."""
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(CSV_FILE).st_size == 0:
            writer.writerow(["Gen", "BestScore", "BestWeights"])
        writer.writerow([gen, score, str(list(weights))])


def load_board_files():
    """
    Loads all .csv files from the ./boards/ directory.
    """
    board_files = glob.glob(os.path.join("boards", "*.csv"))
    boards = []

    if not board_files:
        print("WARNING: No boards found in ./boards/ folder! Using default.")
        b1 = np.zeros((7, 7), dtype=int)
        b1[0, 0] = b1[6, 6] = 1
        b1[0, 6] = b1[6, 0] = 2
        boards.append(b1)
        return boards

    print(f"Loading {len(board_files)} maps for training...")
    for fpath in board_files:
        try:
            b_data = []
            with open(fpath, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    b_data.append([int(x) for x in row])
            boards.append(np.array(b_data))
        except Exception as e:
            print(f"Failed to load {fpath}: {e}")

    return boards


def play_match(args):
    w_p1, w_p2, board_start = args
    agent1 = TrainableAgent(w_p1)
    agent2 = TrainableAgent(w_p2)

    board = board_start.copy()
    max_turns = 80  # --- UPDATED: Increased from 60 to 80 ---
    no_capture = 0

    turn_order = [1, 2]

    for _ in range(max_turns):
        current_p = turn_order[0]
        opponent_p = turn_order[1]

        agent = agent1 if current_p == 1 else agent2

        try:
            move = agent.step(board.copy(), current_p, opponent_p)
        except Exception:
            return -2.0 if current_p == 1 else 2.0

        if move:
            execute_move(board, move, current_p)
            no_capture = 0
        else:
            no_capture += 1

        if no_capture > 4 or check_endgame(board)[0]:
            break

        turn_order.reverse()

    end, p1_s, p2_s = check_endgame(board)
    diff = p1_s - p2_s

    if diff > 0:
        return 1.0 + (diff / 50.0)
    if diff < 0:
        return -1.0 + (diff / 50.0)
    return 0


def evaluate_individual(gene, king_weights, boards):
    score = 0
    b = random.choice(boards)
    score += play_match((gene, king_weights, b))
    res = play_match((king_weights, gene, b))
    score += -res  # Invert because Gene is P2

    return score


def evaluation_wrapper(args):
    return evaluate_individual(*args)


def mutate(gene, scale=1.0):
    new_gene = gene.copy()
    idx = np.random.randint(0, len(gene))
    change = np.random.uniform(-10.0 * scale, 10.0 * scale)
    min_v, max_v = BOUNDS[idx]
    new_gene[idx] = max(min_v, min(max_v, new_gene[idx] + change))
    return new_gene


if __name__ == "__main__":
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Gen", "BestScore", "BestWeights"])

    boards = load_board_files()

    population = [CURRENT_KING_WEIGHTS]
    for _ in range(POPULATION_SIZE - 1):
        population.append(mutate(CURRENT_KING_WEIGHTS, scale=0.5))

    log(f"Starting King of the Hill Training. King: {CURRENT_KING_WEIGHTS}")

    for g in range(GENERATIONS):
        tasks = [(gene, CURRENT_KING_WEIGHTS, boards) for gene in population]

        scores = []
        with multiprocessing.Pool(processes=8) as pool:
            for sc in pool.imap(evaluation_wrapper, tasks):
                scores.append(sc)
                print(f"Gen {g+1} > Eval: {sc:.2f}", end="\r")

        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        best_score, best_gene = ranked[0]

        print(f"\nGen {g+1} Best Score: {best_score:.2f}")
        log_csv(g + 1, best_score, best_gene)

        if best_score > 0.5:
            log(f"*** NEW KING CROWNED *** Score {best_score:.2f}")
            log(f"Old King: {CURRENT_KING_WEIGHTS}")
            CURRENT_KING_WEIGHTS = best_gene
            log(f"New King: {CURRENT_KING_WEIGHTS}")
            save_best_weights(CURRENT_KING_WEIGHTS, best_score, g)

        next_gen = [best_gene]
        next_gen.append(CURRENT_KING_WEIGHTS)
        while len(next_gen) < POPULATION_SIZE:
            parent = ranked[np.random.randint(0, 4)][1]
            next_gen.append(mutate(parent, scale=0.5))

        population = next_gen
