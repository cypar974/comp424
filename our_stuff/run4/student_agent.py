from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

FlagExact = 0
FlagLowerBound = 1
FlagUpperBound = 2


@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.tt = {}
        self.killer_moves = {}
        self.nodes_explored = 0
        self.positional_weights = None

    def step(self, chess_board, my_pos, adv_pos):
        start_time = time.time()
        time_limit = 1.90

        self.nodes_explored = 0
        self.killer_moves = {}
        if len(self.tt) > 200000:
            self.tt.clear()

        player = my_pos
        opponent = adv_pos

        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None

        best_move = valid_moves[0]

        depth = 1
        max_depth = 50
        val = 0

        try:
            while True:
                if time.time() - start_time > time_limit:
                    break
                alpha = float("-inf")
                beta = float("inf")

                if depth > 2:
                    alpha = val - 60  # Window size of roughly 6 pieces
                    beta = val + 60

                val, current_move = self.alphabeta(
                    chess_board,
                    depth,
                    alpha,
                    beta,
                    True,
                    player,
                    opponent,
                    start_time,
                    time_limit,
                    allow_null=True,  # Enable Null Move at root? No, usually inside recursion.
                )

                if val <= alpha or val >= beta:
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
                        allow_null=True,
                    )

                if current_move:
                    best_move = current_move

                if val >= 90000:
                    break

                depth += 1
                if depth > max_depth:
                    break

        except TimeoutError:
            pass

        return best_move

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
        allow_null=True,
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
        if allow_null and depth >= 3 and not self.is_endgame(board):
            R = 2
            null_val, _ = self.alphabeta(
                board,
                depth - 1 - R,
                -beta,
                -beta + 1,
                not is_maximizing,
                player,
                opponent,
                start_time,
                time_limit,
                allow_null=False,
            )
            null_val = -null_val

            if is_maximizing:
                if null_val >= beta:
                    return beta, None
            else:
                if null_val <= alpha:
                    return alpha, None
        if depth <= 2:
            static_eval = self.fast_evaluate(board, player, opponent)
            futility_margin = 150 * depth

            if is_maximizing:
                if static_eval - futility_margin > beta:
                    return static_eval, None
            else:
                if static_eval + futility_margin < alpha:
                    return static_eval, None
        if depth == 0:
            return self.fast_evaluate(board, player, opponent), None

        current_player = player if is_maximizing else opponent
        valid_moves = get_valid_moves(board, current_player)

        if not valid_moves:
            opp_moves = get_valid_moves(board, opponent if is_maximizing else player)
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
                allow_null=False,
            )
            return val, None
        tt_move = tt_entry[3] if tt_entry else None
        killers = self.killer_moves.get(depth, [None, None])
        ordered_moves = self.order_moves_optimized(
            board, valid_moves, tt_move, killers, current_player
        )

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
                allow_null=True,
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

    def order_moves_optimized(self, board, moves, tt_move, killers, player):
        score_list = []
        rows, cols = board.shape
        opponent = 3 - player

        for move in moves:
            score = 0
            if tt_move and move == tt_move:
                score = 100000
            elif move == killers[0]:
                score = 50000
            elif move == killers[1]:
                score = 45000
            else:
                r_dest, c_dest = move.get_dest()
                r_src, c_src = move.get_src()
                if max(abs(r_dest - r_src), abs(c_dest - c_src)) == 1:
                    score += 50
                if 0 <= r_dest < rows and 0 <= c_dest < cols:
                    has_opp_neighbor = False
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if 0 <= r_dest + dr < rows and 0 <= c_dest + dc < cols:
                                if board[r_dest + dr, c_dest + dc] == opponent:
                                    has_opp_neighbor = True
                                    break
                        if has_opp_neighbor:
                            break
                    if has_opp_neighbor:
                        score += 100

            score_list.append((score, move))

        score_list.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in score_list]

    def is_endgame(self, board):
        return np.count_nonzero(board) > (board.size * 0.8)

    def fast_evaluate(self, board, player, opponent):
        w_mat = 191.2
        w_quad = 126.6
        w_mob = 20.0
        w_corn = 21.4
        w_edge = 15.0
        w_cent = -10.0
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
        if w_mob > 0:
            p_moves = len(get_valid_moves(board, player))
            o_moves = len(get_valid_moves(board, opponent))
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

    def get_positional_weights(self, board):
        rows, cols = board.shape
        weights = np.zeros((rows, cols))
        weights[0, 0] = weights[0, cols - 1] = weights[rows - 1, 0] = weights[
            rows - 1, cols - 1
        ] = 5
        weights[0, :] += 2
        weights[rows - 1, :] += 2
        weights[:, 0] += 2
        weights[:, cols - 1] += 2
        return weights
