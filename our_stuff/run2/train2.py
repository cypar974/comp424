import numpy as np
import multiprocessing
import sys
import os
import time
import glob
import csv
import random
import traceback
import datetime

sys.path.append(os.getcwd())

from agents.student_agent import StudentAgent as BaselineAgent
from trainable_agent import TrainableAgent
from helpers import execute_move, check_endgame

# --- CONFIGURATION ---
GENERATIONS = 50
POPULATION_SIZE = 8
GAMES_PER_MATCH = 2
LOG_FILE = "training_log.txt"
CSV_FILE = "training_history.csv"
BEST_WEIGHTS_FILE = "best_weights_so_far.txt"

# Search Bounds
BOUNDS = [
    (160.0, 220.0),  # Material (Keep same)
    (100.0, 200.0),  # Quads -> INCREASE CAP (It loves quads, let it go higher)
    (0.0, 30.0),  # Mobility
    (10.0, 40.0),  # Corners -> Slight increase
    (1.0, 15.0),  # Edges
    (-50.0, 0.0),  # Center -> DECREASE FLOOR (Let it penalize center heavily)
]


# --- LOGGING HELPER ---
def log(message):
    """Print to console and append to log file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")


def save_best_weights(weights, score, gen):
    """Save the best weights immediately to a file."""
    with open(BEST_WEIGHTS_FILE, "w") as f:
        f.write(f"# Best Weights from Gen {gen} (Score: {score:.2f})\n")
        f.write(f"WEIGHTS = {list(weights)}\n")
    log(f"SAVED checkpoint to {BEST_WEIGHTS_FILE}")


# --- FAST AGENT WRAPPER ---
class FastBaselineAgent(BaselineAgent):
    def step(self, chess_board, my_pos, adv_pos):
        # SPEEDUP: Depth 1 for fast training
        val, best_move = self.alphabeta(
            chess_board,
            depth=1,  # <--- Optimized to Depth 1
            alpha=float("-inf"),
            beta=float("inf"),
            is_maximizing=True,
            player=my_pos,
            opponent=adv_pos,
            start_time=time.time(),
            time_limit=100000,
            allow_null=False,
        )
        return best_move


def load_board_files(boards_folder="boards"):
    boards = []
    board_files = glob.glob(os.path.join(boards_folder, "*.csv"))

    if not board_files:
        log(f"WARNING: No boards found in '{boards_folder}'. Using default.")
        b1 = np.zeros((7, 7), dtype=int)
        b1[0, 0] = b1[6, 6] = 1
        b1[0, 6] = b1[6, 0] = 2
        b1[6, 0] = 2
        boards.append(b1)
        return boards

    log(f"Loading {len(board_files)} boards from '{boards_folder}'...")
    for board_file in board_files:
        try:
            board = np.loadtxt(board_file, dtype=int, delimiter=",")
            if board.shape != (7, 7):
                continue
            boards.append(board)
        except Exception as e:
            log(f"  Error loading {board_file}: {e}")
    return boards


def play_match_on_board(args):
    """Play one game with Margin of Victory Scoring."""
    w_p1, w_p2, board = args

    # Initialize Agents
    try:
        agent1 = TrainableAgent(w_p1) if w_p1 else FastBaselineAgent()
        agent2 = TrainableAgent(w_p2) if w_p2 else FastBaselineAgent()
    except Exception:
        return 0

    board = board.copy()
    max_turns = 80
    no_capture_counter = 0

    for _ in range(max_turns):
        # --- P1 Move ---
        try:
            m1 = agent1.step(board.copy(), 1, 2)
        except Exception:
            return -1  # Crash = Loss

        if m1:
            execute_move(board, m1, 1)
            no_capture_counter = 0
        else:
            no_capture_counter += 1

        if no_capture_counter > 4 or check_endgame(board)[0]:
            break

        # --- P2 Move ---
        try:
            m2 = agent2.step(board.copy(), 2, 1)
        except Exception:
            return 1  # Opponent Crash = Win

        if m2:
            execute_move(board, m2, 2)
            no_capture_counter = 0
        else:
            no_capture_counter += 1

        if no_capture_counter > 4 or check_endgame(board)[0]:
            break

    # --- NEW SCORING LOGIC ---
    end, p1_s, p2_s = check_endgame(board)
    diff = p1_s - p2_s

    # If P1 Wins: Score = 1.0 + (Margin / 100)
    # Example: Win by 40 pieces = 1.40 score
    if diff > 0:
        return 1.0 + (diff / 100.0)

    # If P1 Loses: Score = -1.0 + (Margin / 100)
    # Example: Lose by 5 pieces = -1.05 score
    elif diff < 0:
        return -1.0 + (diff / 100.0)

    return 0  # Draw


def evaluate_individual(gene, all_boards):
    total_score = 0
    games_played = 0

    # SAMPLING: Speed up by picking 3 random boards per candidate
    active_boards = all_boards
    if len(all_boards) > 3:
        active_boards = random.sample(all_boards, 3)

    for board in active_boards:
        # Game 1: Gene is P1
        res1 = play_match_on_board((gene, None, board))
        total_score += res1
        games_played += 1

        # Game 2: Gene is P2 (Flip result)
        res2 = play_match_on_board((None, gene, board))
        total_score += -res2
        games_played += 1

    return total_score / games_played if games_played > 0 else 0


def evaluation_wrapper(args):
    return evaluate_individual(*args)


def smart_mutate(gene, generation, max_generations):
    new_gene = gene.copy()
    progress = generation / max_generations
    strength = 15.0 if progress < 0.3 else (10.0 if progress < 0.7 else 5.0)

    num_mutations = np.random.randint(1, 3)
    for _ in range(num_mutations):
        idx = np.random.randint(0, len(gene))
        change = np.random.uniform(-strength, strength)
        min_val, max_val = BOUNDS[idx]
        new_gene[idx] = max(min_val, min(max_val, new_gene[idx] + change))
    return new_gene


def crossover(p1, p2):
    return [p1[i] if np.random.random() < 0.5 else p2[i] for i in range(len(p1))]


if __name__ == "__main__":
    # Clear logs
    with open(LOG_FILE, "w") as f:
        f.write("--- Training Started ---\n")
    with open(CSV_FILE, "w") as f:
        f.write("Gen,BestScore,BestWeights\n")

    all_boards = load_board_files("boards")
    log(f"Training on {len(all_boards)} boards.")

    # Seed

    best_known = [191.16, 124.67, 13.55, 12.44, 9.65, -10.0]
    population = [best_known]
    for _ in range(POPULATION_SIZE - 1):
        population.append(smart_mutate(best_known, 0, GENERATIONS))

    best_gene_all_time = best_known
    best_score_all_time = -999

    log(f"Starting Training: {GENERATIONS} Gens, Pop {POPULATION_SIZE}")

    for g in range(GENERATIONS):
        log(f"--- Generation {g+1}/{GENERATIONS} ---")

        # Prepare Tasks
        tasks = [(gene, all_boards) for gene in population]
        scores = []

        # Run with Progress Feedback
        start_gen = time.time()
        with multiprocessing.Pool(processes=min(6, len(population))) as pool:
            # Use imap to get results as they finish
            for i, score in enumerate(pool.imap(evaluation_wrapper, tasks)):
                scores.append(score)
                # Verbose feedback per candidate
                print(
                    f"  > Candidate {i+1}/{POPULATION_SIZE}: Score {score:.2f}",
                    end="\r",
                )

        print("")  # Newline after progress bar
        log(f"Gen {g+1} Finished in {time.time() - start_gen:.1f}s")

        # Rank
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        best_score, best_gene = ranked[0]

        log(
            f"Gen {g+1} Best: {best_score:.2f} | Weights: {[round(x,1) for x in best_gene]}"
        )

        # Checkpoint if new best
        if best_score > best_score_all_time:
            best_score_all_time = best_score
            best_gene_all_time = best_gene
            log("   *** NEW BEST FOUND! Saving Checkpoint... ***")
            save_best_weights(best_gene, best_score, g + 1)

        # Save History to CSV
        with open(CSV_FILE, "a") as f:
            writer = csv.writer(f)
            writer.writerow([g + 1, best_score, best_gene])

        # Evolution (Elitism + Mutation + Crossover)
        next_gen = [x[1] for x in ranked[:3]]  # Keep top 3
        while len(next_gen) < POPULATION_SIZE:
            if np.random.random() < 0.6:
                parent = ranked[np.random.randint(0, 3)][1]
                next_gen.append(smart_mutate(parent, g, GENERATIONS))
            else:
                p1 = ranked[np.random.randint(0, 3)][1]
                p2 = ranked[np.random.randint(0, 3)][1]
                next_gen.append(crossover(p1, p2))

        population = next_gen

    log("\n=== FINAL WEIGHTS ===")
    log(str(best_gene_all_time))
