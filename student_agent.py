from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, check_endgame, get_valid_moves

# --------------------------------------------
# Cyprien Armand - Austin Wang - Emily Zhang
# 261200373 - 260349977 - 261177806
# Student Agent for COMP 424 Final Project
# --------------------------------------------

# Constants for Transposition Table
FlagExact = 0
FlagLowerBound = 1
FlagUpperBound = 2


class MoveCoordinates:
    """
    Small wrapper that stores a move in the format the engine expects.

    The engine requires an object with named source and destination fields,
    so this class adapts the internal (src, dest) tuple representation.

    We created this because the original Move class was very slow due to
    extensive validation and copying. This lightweight class improves performance
    by minimizing overhead.
    """

    def __init__(self, src: tuple[int, int], dest: tuple[int, int]):
        self.row_src = src[0]
        self.col_src = src[1]
        self.row_dest = dest[0]
        self.col_dest = dest[1]

    def get_src(self) -> tuple[int, int]:
        return (self.row_src, self.col_src)

    def get_dest(self) -> tuple[int, int]:
        return (self.row_dest, self.col_dest)


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Alpha-beta agent with a custom evaluation function.

    The agent searches using iterative deepening alpha-beta with a strict
    time limit. It keeps a transposition table (TT) so repeated positions can
    be reused without searching again.

    Move ordering uses the TT move, killer moves, and a basic capture/clone
    heuristic with a history table to make pruning more effective.

    The evaluation combines material and positional weights.
    These weights were tuned using a genetic algorithm on self-play games.
    Moves are applied directly on the NumPy board
    and then wrapped in MoveCoordinates for the game engine.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.tt = {}  # Transposition table for cached positions
        self.killer_moves = {}  # Killer moves stored per depth
        self.history = np.zeros((7, 7), dtype=int)  # History scores for move ordering
        self.nodes_explored = 0  # Counter for time checks

        # Cached constants for speed
        # All possible move offsets (clone: distance 1, jump: distance 2)
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

        self.positional_weights = np.array(
            [
                [53.70, 24.00, 24.00, 24.00, 24.00, 24.00, 53.70],
                [24.00, -16.38, -16.38, -16.38, -16.38, -16.38, 24.00],
                [24.00, -16.38, -16.38, -16.38, -16.38, -16.38, 24.00],
                [24.00, -16.38, -16.38, -16.38, -16.38, -16.38, 24.00],
                [24.00, -16.38, -16.38, -16.38, -16.38, -16.38, 24.00],
                [24.00, -16.38, -16.38, -16.38, -16.38, -16.38, 24.00],
                [53.70, 24.00, 24.00, 24.00, 24.00, 24.00, 53.70],
            ],
            dtype=float,
        )

    def step(self, full_board, my_pos, adv_pos):
        """
        Select a move by running iterative deepening alpha-beta search.

        The search increases depth progressively until either the time limit
        is reached or the evaluation suggests a forced win. The best move
        from the deepest completed iteration is returned.
        """
        start_time = time.perf_counter()
        time_limit = 1.92  # Safety buffer (Max 2.0s)

        self.nodes_explored = 0
        self.history.fill(0)

        # Periodic TT cleanup to prevent memory issues
        if len(self.tt) > 500000:
            self.tt.clear()

        player = my_pos
        opponent = adv_pos

        valid_moves = self.get_valid_moves(full_board, player)
        if not valid_moves:
            return None

        best_move = valid_moves[0]
        depth = 1
        max_depth = 50

        try:
            while True:
                if time.perf_counter() - start_time > time_limit:
                    break

                val, current_move = self.alphabeta(
                    full_board,
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

                # Mate detection: If we found a winning path, stop searching
                if val > 90000:
                    break

                depth += 1
                if depth > max_depth:
                    break

        except TimeoutError:
            pass

        return MoveCoordinates(best_move[0], best_move[1])

    def get_valid_moves(self, board, player):
        """
        Return all legal moves for a given player.
        Rewritten for speed by avoiding function call overhead
        and using NumPy operations.
        """
        moves = []
        player_pieces = np.argwhere(board == player)

        for r, c in player_pieces:
            for dr, dc in self.all_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 7 and 0 <= nc < 7:
                    if board[nr, nc] == 0:
                        moves.append(((r, c), (nr, nc)))
        return moves

    def apply_move_local(self, board, move, player):
        """
        Apply a move directly to the board.

        Places a piece at the destination, removes the source piece for
        jump moves, and flips any opponent pieces in the 3x3 neighborhood
        around the destination. Walls (value 3) are never flipped.
        """
        (r_src, c_src), (r_dest, c_dest) = move

        board[r_dest, c_dest] = player

        # Distinguish clone vs jump
        if abs(r_dest - r_src) > 1 or abs(c_dest - c_src) > 1:
            board[r_src, c_src] = 0

        r_min = max(0, r_dest - 1)
        r_max = min(7, r_dest + 2)
        c_min = max(0, c_dest - 1)
        c_max = min(7, c_dest + 2)

        window = board[r_min:r_max, c_min:c_max]

        # Determine opponent ID.
        # If player=1, opp=2. If player=2, opp=1.
        opponent = 3 - player

        mask = window == opponent
        window[mask] = player

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
        """
        Performs alpha-beta search with a transposition table and move ordering.

        Time control is enforced by checking the clock every 128
        nodes. Previously evaluated positions are reused through the
        transposition table. Move ordering uses the TT move, killer moves,
        and a capture/clone heuristic with a history score.
        """
        alpha0 = alpha
        beta0 = beta

        # Check time only every 127 nodes to save CPU cycles.
        if self.nodes_explored & 127 == 0:
            if time.perf_counter() - start_time > time_limit:
                raise TimeoutError()
        self.nodes_explored += 1

        # Transposition table lookup for this (board, side-to-move) pair.
        board_hash = (board.tobytes(), is_maximizing).__hash__()
        tt_entry = self.tt.get(board_hash)

        # Use TT entry if it exists and is deep enough
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
        valid_moves = self.get_valid_moves(board, current_player)

        # If the current side has no moves, either the game is over or we pass.
        if not valid_moves:
            opp_moves = self.get_valid_moves(
                board, opponent if is_maximizing else player
            )
            if not opp_moves:
                # Neither side can move: true terminal position.
                return self.final_score(board, player, opponent), None
            # Pass turn: same board, roles swapped, shallower depth.
            val, _ = self.alphabeta(
                board,
                depth - 1,
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

        # Define the scoring logic as a local helper
        def move_scorer(m):
            if tt_move and m == tt_move:
                return 100000
            elif m == killers[0]:
                return 50000
            elif m == killers[1]:
                return 45000

            # Heuristic calculation
            (r_src, c_src), (r_dest, c_dest) = m
            score = 0

            # Base Score: Capture (100) vs Clone (50)
            is_capture = False
            # Check neighbors for capture
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r_dest + dr, c_dest + dc
                if 0 <= nr < 7 and 0 <= nc < 7:
                    if board[nr, nc] == (opponent if is_maximizing else player):
                        is_capture = True
                        break

            if is_capture:
                score = 100
            elif max(abs(r_dest - r_src), abs(c_dest - c_src)) == 1:
                score = 50

            score += self.history[r_dest, c_dest]
            return score

        # Sorting the list in-place using the key function.
        # This avoids creating a list of tuples and a second list of moves.
        valid_moves.sort(key=move_scorer, reverse=True)

        best_move = valid_moves[0]
        best_val = float("-inf") if is_maximizing else float("inf")

        # Iterate over the now-sorted valid_moves directly
        for move in valid_moves:
            sim_board = board.copy()
            self.apply_move_local(sim_board, move, current_player)

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
                # Alpha-beta cutoff
                if move != tt_move:
                    self.store_killer(depth, move)
                    (r_src, c_src), (r_dest, c_dest) = move
                    # Depth-squared bias means deep cutoffs have more impact.
                    self.history[r_dest, c_dest] += depth**2
                break

        # Chooses how to store this node in the transposition table.
        # Use original alpha0 and beta0 values to make sure not everything becomes FlagUpperBound
        tt_flag = FlagExact
        if best_val <= alpha0:
            tt_flag = FlagUpperBound
        elif best_val >= beta0:
            tt_flag = FlagLowerBound
        self.tt[board_hash] = (depth, best_val, tt_flag, best_move)

        return best_val, best_move

    def store_killer(self, depth, move):
        """Record a move that caused a beta cutoff at a given depth."""
        if depth not in self.killer_moves:
            self.killer_moves[depth] = [None, None]
        if self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    def fast_evaluate(self, board, player, opponent):
        """
        Compute a heuristic score for the board using 2 features.

        The weights for each feature were obtained from offline self-play
        training using a genetic algorithm (around 350 generations). The
        final evaluation combines 2 signals:

        Material is the piece count difference.
        Positional score comes from a weight map that likes corners/edges.

        The final score is a weighted sum of these features. Higher scores
        mean the position is better for the current player.
        """

        w_mat = 173.77

        p_count = np.count_nonzero(board == player)
        o_count = np.count_nonzero(board == opponent)
        score_material = (p_count - o_count) * w_mat

        p_pos = np.sum(self.positional_weights[board == player])
        o_pos = np.sum(self.positional_weights[board == opponent])
        score_position = p_pos - o_pos

        return score_material + score_position

    def final_score(self, board, player, opponent):
        """Game over scoring for terminal states based on piece counts."""
        p1 = np.count_nonzero(board == player)
        p2 = np.count_nonzero(board == opponent)
        if p1 > p2:
            return 100000 + p1
        if p2 > p1:
            return -100000 - p2
        return 0
