from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from helpers import get_valid_moves


@register_agent("internet_agent")
class InternetAgent(Agent):
    """
    Reverse Engineered Agent from the provided JS code (Hard Level).
    Strategy: Greedy with a specific heuristic.
    1. Base: +1 if Clone, +0 if Jump.
    2. Penalty (Hard Mode): -1 for every friendly neighbor around the SOURCE tile.
       (Prefers moving isolated pieces).
    3. Bonus: +1 for every opponent piece captured at the DESTINATION.
    """

    def __init__(self):
        super().__init__()
        self.name = "InternetAgent_Hard"
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos):
        valid_moves = get_valid_moves(chess_board, my_pos)
        if not valid_moves:
            return None

        best_score = -99999
        best_moves = []

        rows, cols = chess_board.shape

        # Directions for neighbor checking (including diagonals)
        # The JS code iterates x-1 to x+1, y-1 to y+1
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for move in valid_moves:
            r_src, c_src = move.get_src()
            r_dest, c_dest = move.get_dest()

            # --- 1. BASE SCORE ---
            # JS Logic: if distance == 2 (Jump), score = 0. Else (Clone) score = 1.
            is_jump = max(abs(r_src - r_dest), abs(c_src - c_dest)) > 1
            score = 0 if is_jump else 1

            # --- 2. HARD MODE PENALTY (Neighbors of SOURCE) ---
            # JS Logic: Iterate neighbors of source. If neighbor == my_color, score--
            friendly_neighbors = 0
            for dr, dc in directions:
                nr, nc = r_src + dr, c_src + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if chess_board[nr, nc] == my_pos:
                        friendly_neighbors += 1

            score -= friendly_neighbors

            # --- 3. CAPTURE BONUS (Neighbors of DESTINATION) ---
            # JS Logic: Iterate neighbors of dest. If neighbor == opp_color, score++
            opponent_captures = 0
            for dr, dc in directions:
                nr, nc = r_dest + dr, c_dest + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if chess_board[nr, nc] == adv_pos:
                        opponent_captures += 1

            score += opponent_captures

            # --- FIND MAX ---
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        # Tie-breaking: Pick a random move from the best ones
        if not best_moves:
            return None

        return best_moves[np.random.randint(len(best_moves))]
