import numpy as np
import multiprocessing
import sys
import os
import time

# Fix path to ensure imports work from root
sys.path.append(os.getcwd())

from agents.student_agent import StudentAgent as BaselineAgent
from trainable_agent import TrainableAgent
from helpers import execute_move, check_endgame

# --- CONFIGURATION (OPTIMIZED FOR STABILITY) ---
GENERATIONS = 15
POPULATION_SIZE = 6  # Reduced to 10 to prevent CPU hang
GAMES_PER_MATCH = 2
BOARD_SIZE = 7

# Search Bounds for each weight: [Min, Max]
# [Material, Quads, Mobility, Pos_Corner, Pos_Edge, Pos_Center]
BOUNDS = [
    (150.0, 250.0),  # Material: Focus on high aggression
    (80.0, 150.0),  # Quads: Keep this high
    (0.0, 30.0),  # Mobility: Fine tune
    (10.0, 40.0),  # Corners
    (1.0, 15.0),  # Edges
    (-10.0, 5.0),  # Center: Seems to prefer negative/low
]


def play_match(args):
    """
    Simulates a match between two agent configurations.
    Returns: 1 if P1 wins, -1 if P2 wins, 0 if Tie.
    """
    w_p1, w_p2 = args

    # If weights provided, use Trainable, else use Baseline
    agent1 = TrainableAgent(w_p1) if w_p1 else BaselineAgent()
    agent2 = TrainableAgent(w_p2) if w_p2 else BaselineAgent()

    # Setup Board
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    board[0, 0] = 1
    board[BOARD_SIZE - 1, BOARD_SIZE - 1] = 1
    board[0, BOARD_SIZE - 1] = 2
    board[BOARD_SIZE - 1, 0] = 2

    turn = 1
    no_capture_counter = 0

    # Fast Game Loop
    while True:
        # P1 moves
        m1 = agent1.step(board.copy(), 1, 2)
        if m1:
            execute_move(board, m1, 1)
            no_capture_counter = 0
        else:
            no_capture_counter += 1

        if check_endgame(board)[0]:
            break
        if no_capture_counter > 10:
            break  # Tie-break for stalled games

        # P2 moves
        m2 = agent2.step(board.copy(), 2, 1)
        if m2:
            execute_move(board, m2, 2)
            no_capture_counter = 0
        else:
            no_capture_counter += 1

        if check_endgame(board)[0]:
            break
        if no_capture_counter > 10:
            break

    # Result
    end, p1_s, p2_s = check_endgame(board)
    if p1_s > p2_s:
        return 1
    elif p2_s > p1_s:
        return -1
    return 0


def evaluate_generation(population):
    tasks = []
    # Create match-ups: Candidate vs Baseline
    for gene in population:
        tasks.append((gene, None))  # Candidate is P1
        tasks.append((None, gene))  # Candidate is P2

    print("  Playing matches:", end=" ", flush=True)

    match_results = []

    # FIX: Force 6 processes to avoid E-core hanging
    # Using 'imap' to print dots as they finish
    with multiprocessing.Pool(processes=6) as pool:
        for result in pool.imap(play_match, tasks):
            match_results.append(result)
            print(".", end="", flush=True)

    print(" Done!")  # Newline after dots

    scores = []
    for i in range(len(population)):
        # Win as P1 (+1), Win as P2 (-(-1) = +1)
        score = match_results[i * 2] + (-match_results[i * 2 + 1])
        scores.append(score)
    return scores


def mutate(gene):
    new_gene = list(gene)
    idx = np.random.randint(0, len(gene))

    # CHANGE THIS: Increase mutation range from 10.0 to 20.0 or 25.0
    # This forces larger, more exploratory jumps in the weights.
    change = np.random.uniform(-25.0, 25.0)  # <--- CHANGE THIS LINE

    new_gene[idx] += change
    return new_gene


if __name__ == "__main__":
    # Initialize Population
    population = []
    for _ in range(POPULATION_SIZE):
        gene = [np.random.uniform(b[0], b[1]) for b in BOUNDS]
        population.append(gene)

    print(f"--- Starting Training ({GENERATIONS} Gens) ---")
    print(f"--- Population Size: {POPULATION_SIZE} ---")

    best_gene_all_time = None
    best_score_all_time = -999

    for g in range(GENERATIONS):
        scores = evaluate_generation(population)

        # Sort by score
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        best_score, best_gene = ranked[0]

        print(
            f"Gen {g+1}: Best Score {best_score}/2 | Top Weights: {[round(x,1) for x in best_gene]}"
        )

        if best_score > best_score_all_time:
            best_score_all_time = best_score
            best_gene_all_time = best_gene

        # Evolution: Keep top 2, mutate rest (adjusted for smaller population)
        next_gen = [x[1] for x in ranked[:4]]
        while len(next_gen) < POPULATION_SIZE:
            parent = ranked[np.random.randint(0, 2)][1]
            next_gen.append(mutate(parent))
        population = next_gen

    print("\n--- TRAINING COMPLETE ---")
    print("Paste these weights into your 'fast_evaluate':")
    if best_gene_all_time:
        print(f"Material: {round(best_gene_all_time[0], 2)}")
        print(f"Quads:    {round(best_gene_all_time[1], 2)}")
        print(f"Mobility: {round(best_gene_all_time[2], 2)}")
        print(f"Corners:  {round(best_gene_all_time[3], 2)}")
        print(f"Edges:    {round(best_gene_all_time[4], 2)}")
        print(f"Center:   {round(best_gene_all_time[5], 2)}")
