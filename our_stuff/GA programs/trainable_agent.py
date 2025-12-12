import numpy as np
import sys
import os
import time  # <--- Added import

# Ensure we can import from the 'agents' folder and the current root
sys.path.append(os.getcwd())

from agents.student_agent import StudentAgent
from helpers import get_valid_moves


class TrainableAgent(StudentAgent):
    def __init__(self, weights=None):
        super().__init__()
        # Weights Configuration:
        # [0] Material
        # [1] Quads
        # [2] Mobility
        # [3] Positional: Corners
        # [4] Positional: Edges
        # [5] Positional: Center
        self.weights = weights if weights else [100.0, 50.0, 0.0, 10.0, 2.0, 0.0]
        self.positional_weights = None

    def step(self, chess_board, my_pos, adv_pos):
        # Training Optimization: Fixed depth of 2 for speed
        depth = 2
        alpha = float("-inf")
        beta = float("inf")

        # FIX: We pass time.time() as start_time, and a huge time_limit
        # to ensure it never times out during training.
        try:
            val, best_move = self.alphabeta(
                chess_board,
                depth,
                alpha,
                beta,
                True,
                my_pos,
                adv_pos,
                time.time(),
                100000,
                allow_null=False,
            )
            return best_move
        except TimeoutError:
            # Fallback if something goes wrong (shouldn't happen with 100000s limit)
            valid = get_valid_moves(chess_board, my_pos)
            return valid[0] if valid else None

    def fast_evaluate(self, board, player, opponent):
        # Unpack the genome (weights)
        w_mat, w_quad, w_mob, w_corn, w_edge, w_cent = self.weights

        # 1. Material
        p_count = np.count_nonzero(board == player)
        o_count = np.count_nonzero(board == opponent)
        score_material = (p_count - o_count) * w_mat

        # 2. Structure (Quads)
        b_tl = board[:-1, :-1]
        b_tr = board[:-1, 1:]
        b_bl = board[1:, :-1]
        b_br = board[1:, 1:]

        p_quads = np.sum(
            (b_tl == player) & (b_tr == player) & (b_bl == player) & (b_br == player)
        )
        o_quads = np.sum(
            (b_tl == opponent)
            & (b_tr == opponent)
            & (b_bl == opponent)
            & (b_br == opponent)
        )
        score_structure = (p_quads - o_quads) * w_quad

        # 3. Mobility
        score_mobility = 0
        if w_mob > 0.1:
            p_moves = len(get_valid_moves(board, player))
            o_moves = len(get_valid_moves(board, opponent))
            score_mobility = (p_moves - o_moves) * w_mob

        # 4. Positional
        if self.positional_weights is None:
            rows, cols = board.shape
            pw = np.full((rows, cols), w_cent)

            # Edges
            pw[0, :] = w_edge
            pw[rows - 1, :] = w_edge
            pw[:, 0] = w_edge
            pw[:, cols - 1] = w_edge

            # Corners
            pw[0, 0] = pw[0, cols - 1] = pw[rows - 1, 0] = pw[rows - 1, cols - 1] = (
                w_corn
            )

            self.positional_weights = pw

        p_pos = np.sum(self.positional_weights[board == player])
        o_pos = np.sum(self.positional_weights[board == opponent])
        score_position = p_pos - o_pos

        return score_material + score_structure + score_mobility + score_position
