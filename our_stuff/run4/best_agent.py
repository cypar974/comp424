from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from helpers import execute_move, MoveCoordinates

# Constants
FlagExact = 0
FlagLowerBound = 1
FlagUpperBound = 2


@register_agent("best_agent")
class BestAgent(Agent):
    def __init__(self):
        super(BestAgent, self).__init__()
        self.name = "TheArchitect_Clean"
        self.autoplay = True
        self.tt = {}
        self.killer_moves = {}
        self.nodes_explored = 0
        self.positional_weights = None

        # --- CACHED CONSTANTS FOR SPEED ---
        self.directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        self.jumps = [
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
        self.all_offsets = self.directions + self.jumps

    def step(self, chess_board, my_pos, adv_pos):
        start_time = time.time()
        time_limit = 1.90

        self.nodes_explored = 0
        # Manage TT Size
        if len(self.tt) > 500000:
            self.tt.clear()

        player = my_pos
        opponent = adv_pos

        # Use FAST move generation
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

                # REMOVED: Aspiration Windows (Too unstable for now)
                # REMOVED: Null Move Pruning (Risky in Ataxx)

                # Standard Search
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

                # Win instantly if we found a winning line
                if val > 90000:
                    break

                depth += 1
                if depth > max_depth:
                    break

        except TimeoutError:
            pass

        return best_move

    def get_valid_moves_fast(self, board, player):
        """Internal optimized move generator (Bypasses helpers.py)"""
        moves = []
        rows, cols = board.shape
        # Numpy argwhere is much faster than nested loops
        player_pieces = np.argwhere(board == player)

        for r, c in player_pieces:
            for dr, dc in self.all_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
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
        # Check time every 1024 nodes (Bitwise AND is fast)
        if self.nodes_explored & 1023 == 0:
            if time.time() - start_time > time_limit:
                raise TimeoutError()
        self.nodes_explored += 1

        # TT Lookup
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

        # Base Case
        if depth == 0:
            return self.fast_evaluate(board, player, opponent), None

        current_player = player if is_maximizing else opponent
        valid_moves = self.get_valid_moves_fast(board, current_player)

        # Terminal Node Check (No moves)
        if not valid_moves:
            opp_moves = self.get_valid_moves_fast(
                board, opponent if is_maximizing else player
            )
            if not opp_moves:
                return self.final_score(board, player, opponent), None
            # Pass turn (Same player moves again if opponent has no moves? No, other player moves)
            # Standard Ataxx: If I have no moves, I pass.
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

        # Move Ordering
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
                # Optimized Heuristic: Capture > Clone
                r_dest, c_dest = m.get_dest()
                r_src, c_src = m.get_src()

                # Check neighbors for captures (Fast check)
                is_capture = False
                # Just check 4 cardinal directions for speed
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
        if depth not in self.killer_moves:
            self.killer_moves[depth] = [None, None]
        if self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    def fast_evaluate(self, board, player, opponent):
        # --- THE "ARCHITECT" WEIGHTS (Learned Overnight) ---
        w_mat = 173.77
        w_quad = 146.41
        w_mob = 13.16
        w_corn = 29.00
        w_edge = 7.18
        w_cent = -8.10

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

        score_mobility = 0
        if w_mob > 5.0:
            p_moves = len(self.get_valid_moves_fast(board, player))
            o_moves = len(self.get_valid_moves_fast(board, opponent))
            score_mobility = (p_moves - o_moves) * w_mob

        if self.positional_weights is None:
            rows, cols = board.shape
            pw = np.full((rows, cols), w_cent)
            pw[0, :] = w_edge
            pw[rows - 1, :] = w_edge
            pw[:, 0] = w_edge
            pw[:, cols - 1] = w_edge
            pw[0, 0] = pw[0, cols - 1] = pw[rows - 1, 0] = pw[rows - 1, cols - 1] = (
                w_corn
            )
            self.positional_weights = pw

        p_pos = np.sum(self.positional_weights[board == player])
        o_pos = np.sum(self.positional_weights[board == opponent])
        score_position = p_pos - o_pos

        return score_material + score_structure + score_mobility + score_position

    def final_score(self, board, player, opponent):
        p1 = np.count_nonzero(board == player)
        p2 = np.count_nonzero(board == opponent)
        if p1 > p2:
            return 100000 + p1
        if p2 > p1:
            return -100000 - p2
        return 0
