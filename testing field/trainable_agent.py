import numpy as np
import sys
import os
import time

sys.path.append(os.getcwd())
from agents.student_agent import StudentAgent


class TrainableAgent(StudentAgent):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights if weights else [173.0, 146.0, 13.0, 29.0, 7.0, -8.0]
        self.positional_weights_cache = None

    def step(self, chess_board, my_pos, adv_pos):
        depth = 3  # Fixed depth for consistent training comparison
        alpha = float("-inf")
        beta = float("inf")

        try:
            val, best_move = self.alphabeta(
                chess_board,
                depth,
                alpha,
                beta,
                True,
                my_pos,
                adv_pos,
                time.perf_counter(),
                100000.0,  # Huge time buffer
            )
            return best_move
        except Exception as e:
            valid = self.get_valid_moves_fast(chess_board, my_pos)
            return valid[0] if valid else None

    def fast_evaluate(self, board, player, opponent):
        """
        Overwrite the hardcoded weights of student_agent.py with our Genome.
        """
        w_mat, w_quad, w_mob, w_corn, w_edge, w_cent = self.weights
        p_count = np.count_nonzero(board == player)
        o_count = np.count_nonzero(board == opponent)
        score_material = (p_count - o_count) * w_mat
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
        p_moves = len(self.get_valid_moves_fast(board, player))
        o_moves = len(self.get_valid_moves_fast(board, opponent))
        score_mobility = (p_moves - o_moves) * w_mob
        if self.positional_weights_cache is None:
            rows, cols = board.shape
            pw = np.full((rows, cols), w_cent)
            pw[0, :] = w_edge
            pw[rows - 1, :] = w_edge
            pw[:, 0] = w_edge
            pw[:, cols - 1] = w_edge
            pw[0, 0] = pw[0, cols - 1] = pw[rows - 1, 0] = pw[rows - 1, cols - 1] = (
                w_corn
            )
            self.positional_weights_cache = pw

        p_pos = np.sum(self.positional_weights_cache[board == player])
        o_pos = np.sum(self.positional_weights_cache[board == opponent])
        score_position = p_pos - o_pos

        return score_material + score_structure + score_mobility + score_position
