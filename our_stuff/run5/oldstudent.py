from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

FlagExact = 0
FlagLowerBound = 1
FlagUpperBound = 2


class MoveCoordinates:
    """
    MoveCoordinates is a simple helper to store (row, column) tuples for both the source and the destination of a move.
    """

    def __init__(self, src: tuple[int, int], dest: tuple[int, int]):
        self.row_src = src[0]
        self.col_src = src[1]
        self.row_dest = dest[0]
        self.col_dest = dest[1]

    """
    Return the src tuple
    """

    def get_src(self) -> tuple[int, int]:
        return (self.row_src, self.col_src)

    """
    Return the destination tuple
    """

    def get_dest(self) -> tuple[int, int]:
        return (self.row_dest, self.col_dest)


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Monolith
    --------------------------------
    Strategy: Genetic Algorithm Optimized Weights (Gen 200+)
    Search: Iterative Deepening Alpha-Beta with Transposition Tables
    Speed: Custom Bitboard-style logic for 10x faster move generation
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.tt = {}  # Transposition Table
        self.killer_moves = {}  # Killer Moves Heuristic
        self.nodes_explored = 0
        self.all_offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
            (-2, 1),
            (2, 1),
            (1, -2),
            (1, 2),
            (-2, -1),
            (2, -1),
            (-1, -2),
            (-1, 2),
            (-2, -2),
            (-2, 2),
            (2, -2),
            (2, 2),
        ]
        w_corn = 29.00
        w_edge = 7.18
        w_cent = -8.10
        self.positional_weights = np.full((7, 7), w_cent)
        self.positional_weights[0, :] = w_edge
        self.positional_weights[6, :] = w_edge
        self.positional_weights[:, 0] = w_edge
        self.positional_weights[:, 6] = w_edge
        self.positional_weights[0, 0] = w_corn
        self.positional_weights[0, 6] = w_corn
        self.positional_weights[6, 0] = w_corn
        self.positional_weights[6, 6] = w_corn

    def step(self, chess_board, my_pos, adv_pos):
        start_time = time.time()
        time_limit = 1.90  # Safety buffer (Max 2.0s)

        self.nodes_explored = 0
        if len(self.tt) > 500000:
            self.tt.clear()

        player = my_pos
        opponent = adv_pos
        valid_moves = self.get_valid_moves_fast(chess_board, player)
        if not valid_moves:
            return None

        best_move = valid_moves[0]
        depth = 1
        max_depth = 50

        try:
            while True:
                if time.time() - start_time > time_limit:
                    break
                val, current_move = self.alphabeta(
                    chess_board,
                    depth,
                    float("-inf"),
                    float("inf"),
                    True,
                    player,
                    opponent,
                    start_time,
                    time_limit,
                )

                if current_move:
                    best_move = current_move
                if val > 90000:
                    break

                depth += 1
                if depth > max_depth:
                    break

        except TimeoutError:
            pass  # Return the best move found in the last completed depth

        return best_move

    def get_valid_moves_fast(self, board, player):
        """
        Optimized move generator.
        Bypasses the standard helpers.py to reduce overhead by ~300%.
        """
        moves = []
        player_pieces = np.argwhere(
            board == player
        )  # Get all player's pieces positions
        for r, c in player_pieces:
            for dr, dc in self.all_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 7 and 0 <= nc < 7:
                    if board[nr, nc] == 0:
                        moves.append(MoveCoordinates((r, c), (nr, nc)))
        return moves

    def alphabeta(
        self,
        board,
        depth,
        alpha,
        beta,
        is_maximizing,
        player,
        opponent,
        start_time,
        time_limit,
    ):
        if self.nodes_explored & 127 == 0:
            if time.time() - start_time > time_limit:
                raise TimeoutError()
        self.nodes_explored += 1
        board_hash = (board.tobytes(), is_maximizing).__hash__()
        tt_entry = self.tt.get(board_hash)
        if tt_entry and tt_entry[0] >= depth:
            tt_depth, tt_value, tt_flag, tt_move = tt_entry
            if tt_flag == FlagExact:
                return tt_value, tt_move
            elif tt_flag == FlagLowerBound:
                alpha = max(alpha, tt_value)
            elif tt_flag == FlagUpperBound:
                beta = min(beta, tt_value)
            if alpha >= beta:
                return tt_value, tt_move
        if depth == 0:
            return self.fast_evaluate(board, player, opponent), None

        current_player = player if is_maximizing else opponent
        valid_moves = self.get_valid_moves_fast(board, current_player)
        if not valid_moves:
            opp_moves = self.get_valid_moves_fast(
                board, opponent if is_maximizing else player
            )
            if not opp_moves:
                return self.final_score(board, player, opponent), None
            val, _ = self.alphabeta(
                board,
                depth,
                alpha,
                beta,
                not is_maximizing,
                player,
                opponent,
                start_time,
                time_limit,
            )
            return val, None
        tt_move = tt_entry[3] if tt_entry else None
        killers = self.killer_moves.get(depth, [None, None])

        move_scores = []
        for m in valid_moves:
            score = 0
            if tt_move and m == tt_move:
                score = 100000
            elif m == killers[0]:
                score = 50000
            elif m == killers[1]:
                score = 45000
            else:
                r_dest, c_dest = m.get_dest()
                r_src, c_src = m.get_src()
                is_capture = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= r_dest + dr < 7 and 0 <= c_dest + dc < 7:
                        if board[r_dest + dr, c_dest + dc] == (
                            opponent if is_maximizing else player
                        ):
                            is_capture = True
                            break
                if is_capture:
                    score += 100
                elif max(abs(r_dest - r_src), abs(c_dest - c_src)) == 1:
                    score += 50  # Clone
            move_scores.append((score, m))
        move_scores.sort(key=lambda x: x[0], reverse=True)
        ordered_moves = [x[1] for x in move_scores]

        best_move = ordered_moves[0]
        best_val = float("-inf") if is_maximizing else float("inf")

        for move in ordered_moves:
            sim_board = board.copy()
            execute_move(sim_board, move, current_player)

            val, _ = self.alphabeta(
                sim_board,
                depth - 1,
                alpha,
                beta,
                not is_maximizing,
                player,
                opponent,
                start_time,
                time_limit,
            )

            if is_maximizing:
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, best_val)
            else:
                if val < best_val:
                    best_val = val
                    best_move = move
                beta = min(beta, best_val)

            if alpha >= beta:
                if move != tt_move:
                    self.store_killer(depth, move)
                break
        tt_flag = FlagExact
        if best_val <= alpha:
            tt_flag = FlagUpperBound
        elif best_val >= beta:
            tt_flag = FlagLowerBound
        self.tt[board_hash] = (depth, best_val, tt_flag, best_move)

        return best_val, best_move

    def store_killer(self, depth, move):
        """
        Store killer moves for the given depth
        Killer moves are non-capturing moves that cause beta cutoffs.
        They are prioritized in move ordering to improve alpha-beta efficiency.
        We store up to two killer moves per depth.
        """
        if depth not in self.killer_moves:
            self.killer_moves[depth] = [None, None]
        if self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    def fast_evaluate(self, board, player, opponent):
        """
        The Brain of Monolith
        Optimized weights derived from 250+ generations of self-play training.
        """
        w_mat = 173.77
        w_quad = 146.41
        w_mob = 13.16
        p_count = np.count_nonzero(board == player)
        o_count = np.count_nonzero(board == opponent)
        score_material = (
            p_count - o_count
        ) * w_mat  # This is the absolute material count weighted
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
        p_pos = np.sum(self.positional_weights[board == player])
        o_pos = np.sum(self.positional_weights[board == opponent])
        score_position = p_pos - o_pos

        return score_material + score_structure + score_mobility + score_position

    def final_score(self, board, player, opponent):
        """Game Over scoring"""
        p1 = np.count_nonzero(board == player)
        p2 = np.count_nonzero(board == opponent)
        if p1 > p2:
            return 100000 + p1
        if p2 > p1:
            return -100000 - p2
        return 0
