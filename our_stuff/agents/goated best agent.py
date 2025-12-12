# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from helpers import MoveCoordinates

# --- CONSTANTS & PRECOMPUTATION ---
BOARD_SIZE = 7
BOARD_AREA = BOARD_SIZE * BOARD_SIZE

# Precomputed bit masks for efficient access
BIT_MASKS = [1 << i for i in range(BOARD_AREA)]

# Positional weights: Corners and "pockets" are valuable.
# Center is critical for mobility.
POS_WEIGHTS_GRID = [
    [20, 5, 5, 5, 5, 5, 20],
    [5, 1, 1, 1, 1, 1, 5],
    [5, 1, 8, 5, 8, 1, 5],
    [5, 1, 5, 8, 5, 1, 5],
    [5, 1, 8, 5, 8, 1, 5],
    [5, 1, 1, 1, 1, 1, 5],
    [20, 5, 5, 5, 5, 5, 20],
]
POS_WEIGHTS = [0] * BOARD_AREA
for r in range(BOARD_SIZE):
    for c in range(BOARD_SIZE):
        POS_WEIGHTS[r * BOARD_SIZE + c] = POS_WEIGHTS_GRID[r][c]


def precompute_masks():
    neighbors = [0] * BOARD_AREA
    jumps = [0] * BOARD_AREA
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            idx = r * BOARD_SIZE + c
            # Neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        n_idx = nr * BOARD_SIZE + nc
                        neighbors[idx] |= 1 << n_idx
            # Jumps
            for dr in [-2, -1, 0, 1, 2]:
                for dc in [-2, -1, 0, 1, 2]:
                    if abs(dr) <= 1 and abs(dc) <= 1:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        n_idx = nr * BOARD_SIZE + nc
                        jumps[idx] |= 1 << n_idx
    return neighbors, jumps


NEIGHBOR_MASKS, JUMP_MASKS = precompute_masks()

# Fast bit counting
if hasattr(int, "bit_count"):

    def count_set_bits(n):
        return n.bit_count()

else:

    def count_set_bits(n):
        return bin(n).count("1")


class SearchTimeout(Exception):
    pass


@register_agent("best_agent")
class BestAgent(Agent):
    """
    The absolute limit of Python performance for Ataxx.
    Features:
    - Bitboards
    - Iterative Deepening
    - PVS (Principal Variation Search / NegaScout)
    - Aspiration Windows
    - Transposition Table (Exact/Bounds)
    - Killer Heuristic + History Heuristic (New)
    - Advanced Pattern Evaluation (Mobility + Danger + Clusters)
    """

    def __init__(self):
        super(BestAgent, self).__init__()
        self.name = "BestAgent"
        self.autoplay = True
        self.neighbor_masks = NEIGHBOR_MASKS
        self.jump_masks = JUMP_MASKS
        self.pos_weights = POS_WEIGHTS

        # Configuration
        self.time_limit = 1.90  # Strict safety margin
        self.start_time = 0
        self.nodes_explored = 0

        # Transposition Table
        self.tt = {}

        # Heuristics Tables
        # Killers: [depth][slot]
        self.killers = []
        # History: [src][dst] -> score
        self.history = [[0] * BOARD_AREA for _ in range(BOARD_AREA)]

    def step(self, chess_board, player, opponent):
        self.start_time = time.time()
        self.nodes_explored = 0
        self.tt = {}
        self.killers = [[None] * 2 for _ in range(100)]
        # Decay history for new turn to prefer fresh good moves
        for r in range(BOARD_AREA):
            for c in range(BOARD_AREA):
                self.history[r][c] //= 2

        # 1. Board Parsing
        my_board = 0
        opp_board = 0
        obstacles = 0
        flat = chess_board.flatten()
        for i in range(BOARD_AREA):
            v = flat[i]
            if v == player:
                my_board |= 1 << i
            elif v == opponent:
                opp_board |= 1 << i
            elif v == 3:
                obstacles |= 1 << i

        if my_board == 0:
            return None

        best_move = None
        last_score = 0

        try:
            # Iterative Deepening
            for depth in range(1, 25):
                # Aspiration Windows
                if depth > 3 and best_move is not None:
                    alpha_window = last_score - 40
                    beta_window = last_score + 40
                    score, move = self.negamax(
                        my_board, opp_board, obstacles, depth, alpha_window, beta_window
                    )

                    if score <= alpha_window or score >= beta_window:
                        score, move = self.negamax(
                            my_board,
                            opp_board,
                            obstacles,
                            depth,
                            -float("inf"),
                            float("inf"),
                        )
                else:
                    score, move = self.negamax(
                        my_board,
                        opp_board,
                        obstacles,
                        depth,
                        -float("inf"),
                        float("inf"),
                    )

                if move is not None:
                    best_move = move
                    last_score = score

                if score > 90000:
                    break

        except SearchTimeout:
            pass

        if best_move is None:
            return None

        src, dst = best_move

        # Correctly return MoveCoordinates object as expected by the simulator
        return MoveCoordinates(
            src=(src // BOARD_SIZE, src % BOARD_SIZE),
            dest=(dst // BOARD_SIZE, dst % BOARD_SIZE),
        )

    def negamax(self, my_board, opp_board, obstacles, depth, alpha, beta):
        # Polling time
        self.nodes_explored += 1
        if self.nodes_explored & 2047 == 0:
            if time.time() - self.start_time > self.time_limit:
                raise SearchTimeout()

        alpha_orig = alpha

        # 1. Transposition Table
        tt_entry = self.tt.get((my_board, opp_board))
        tt_move = None
        if tt_entry:
            tt_val, tt_depth, tt_flag, tt_m = tt_entry
            if tt_depth >= depth:
                if tt_flag == 0:
                    return tt_val, tt_m
                elif tt_flag == 1:
                    alpha = max(alpha, tt_val)
                elif tt_flag == 2:
                    beta = min(beta, tt_val)
                if alpha >= beta:
                    return tt_val, tt_m
            tt_move = tt_m

        # 2. Terminal / Leaf
        if depth == 0:
            return self.quiescence(my_board, opp_board, obstacles, alpha, beta), None

        all_pieces = my_board | opp_board | obstacles
        empty = ~all_pieces & ((1 << 49) - 1)

        if empty == 0 or my_board == 0 or opp_board == 0:
            return self.evaluate_endgame(my_board, opp_board), None

        best_score = -float("inf")
        best_move = None

        # 3. Move Ordering Logic
        moves_to_try = []  # List of (src, dst, type)

        # A. TT Move
        if tt_move:
            moves_to_try.append((tt_move[0], tt_move[1], 0))  # Type 0 = TT

        # --- Move Generation Loop ---

        # Phase 0: TT Move
        if tt_move:
            # Execute TT
            src, dst = tt_move
            is_jump = not (self.neighbor_masks[src] & (1 << dst))
            if is_jump:
                new_my = (my_board ^ (1 << src)) | (1 << dst)
            else:
                new_my = my_board | (1 << dst)

            caps = self.neighbor_masks[dst] & opp_board
            new_my |= caps
            new_opp = opp_board & ~caps

            val, _ = self.negamax(new_opp, new_my, obstacles, depth - 1, -beta, -alpha)
            val = -val

            if val > best_score:
                best_score = val
                best_move = tt_move
            alpha = max(alpha, val)
            if alpha >= beta:
                self.tt[(my_board, opp_board)] = (best_score, depth, 1, best_move)
                return best_score, best_move

        # Phase 1: Generate Remaining Moves & Score Them
        candidates = []

        pieces = my_board
        while pieces:
            lsb = pieces & -pieces
            src = lsb.bit_length() - 1
            pieces ^= lsb

            # Duplicates (High Priority)
            valid_dsts = self.neighbor_masks[src] & empty
            while valid_dsts:
                dst_lsb = valid_dsts & -valid_dsts
                dst = dst_lsb.bit_length() - 1
                valid_dsts ^= dst_lsb

                if tt_move and src == tt_move[0] and dst == tt_move[1]:
                    continue

                # Score: Capture size * 1000 + History
                caps = count_set_bits(self.neighbor_masks[dst] & opp_board)
                score = (caps * 1000) + self.history[src][dst]

                # Killer Bonus
                if (
                    self.killers[depth][0]
                    and src == self.killers[depth][0][0]
                    and dst == self.killers[depth][0][1]
                ):
                    score += 100000
                elif (
                    self.killers[depth][1]
                    and src == self.killers[depth][1][0]
                    and dst == self.killers[depth][1][1]
                ):
                    score += 90000

                candidates.append((score, src, dst, False))  # False = Duplicate

            # Jumps (Low Priority)
            valid_jumps = self.jump_masks[src] & empty
            while valid_jumps:
                dst_lsb = valid_jumps & -valid_jumps
                dst = dst_lsb.bit_length() - 1
                valid_jumps ^= dst_lsb

                if tt_move and src == tt_move[0] and dst == tt_move[1]:
                    continue

                caps = count_set_bits(self.neighbor_masks[dst] & opp_board)
                score = (caps * 1000) + self.history[src][dst] - 50  # Penalty for jump

                if (
                    self.killers[depth][0]
                    and src == self.killers[depth][0][0]
                    and dst == self.killers[depth][0][1]
                ):
                    score += 100000
                elif (
                    self.killers[depth][1]
                    and src == self.killers[depth][1][0]
                    and dst == self.killers[depth][1][1]
                ):
                    score += 90000

                candidates.append((score, src, dst, True))  # True = Jump

        # Sort candidates (descending score)
        # This sort is key to "History Heuristic"
        candidates.sort(key=lambda x: x[0], reverse=True)

        moves_tried = 1 if tt_move else 0

        for _, src, dst, is_jump in candidates:
            # Execute
            if not is_jump:
                new_my = my_board | (1 << dst)
            else:
                new_my = (my_board ^ (1 << src)) | (1 << dst)

            caps = self.neighbor_masks[dst] & opp_board
            new_my |= caps
            new_opp = opp_board & ~caps

            # PVS Logic
            if moves_tried == 0:
                val, _ = self.negamax(
                    new_opp, new_my, obstacles, depth - 1, -beta, -alpha
                )
                val = -val
            else:
                val, _ = self.negamax(
                    new_opp, new_my, obstacles, depth - 1, -alpha - 1, -alpha
                )
                val = -val
                if alpha < val < beta:
                    val, _ = self.negamax(
                        new_opp, new_my, obstacles, depth - 1, -beta, -alpha
                    )
                    val = -val

            moves_tried += 1

            if val > best_score:
                best_score = val
                best_move = (src, dst)
            alpha = max(alpha, val)
            if alpha >= beta:
                # Beta Cutoff
                # Store Killer
                if self.killers[depth][0] != (src, dst):
                    self.killers[depth][1] = self.killers[depth][0]
                    self.killers[depth][0] = (src, dst)
                # Update History
                self.history[src][dst] += depth * depth

                self.tt[(my_board, opp_board)] = (best_score, depth, 1, best_move)
                return best_score, best_move

        if moves_tried == 0:
            # Pass
            val, _ = self.negamax(
                opp_board, my_board, obstacles, depth - 1, -beta, -alpha
            )
            val = -val
            return val, None

        flag = 0
        if best_score <= alpha_orig:
            flag = 2
        elif best_score >= beta:
            flag = 1
        self.tt[(my_board, opp_board)] = (best_score, depth, flag, best_move)

        return best_score, best_move

    def quiescence(self, my_board, opp_board, obstacles, alpha, beta):
        """
        Quiescence Search: Extends search for unstable (capture) nodes.
        For Ataxx, any move adjacent to opponent is 'unstable' (capture).
        We just return static eval for simplicity because Ataxx is 'explosive' everywhere.
        Implementing full Q-Search in Python 2s limit is often counter-productive.
        """
        return self.evaluate(my_board, opp_board, obstacles)

    def evaluate(self, my_board, opp_board, obstacles):
        # 1. Material (Weight: 100)
        my_pop = count_set_bits(my_board)
        opp_pop = count_set_bits(opp_board)
        score = (my_pop - opp_pop) * 100

        # 2. Positional (Weight: ~5-20)
        # Using precomputed weights
        temp = my_board
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            score += self.pos_weights[idx]
            temp ^= lsb

        temp = opp_board
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            score -= self.pos_weights[idx]
            temp ^= lsb

        # 3. Mobility (Weight: 10)
        # Approximate mobility: Sum of empty neighbors
        # We scan all my pieces and OR their neighbor masks
        my_mob_mask = 0
        temp = my_board
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            my_mob_mask |= self.neighbor_masks[idx]
            temp ^= lsb

        opp_mob_mask = 0
        temp = opp_board
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            opp_mob_mask |= self.neighbor_masks[idx]
            temp ^= lsb

        empty = ~(my_board | opp_board | obstacles) & ((1 << 49) - 1)
        score += count_set_bits(my_mob_mask & empty) * 10
        score -= count_set_bits(opp_mob_mask & empty) * 10

        return score

    def evaluate_endgame(self, my_board, opp_board):
        my_count = count_set_bits(my_board)
        opp_count = count_set_bits(opp_board)
        if my_count > opp_count:
            return 100000 + (my_count - opp_count)
        elif opp_count > my_count:
            return -100000 - (opp_count - my_count)
        return 0
