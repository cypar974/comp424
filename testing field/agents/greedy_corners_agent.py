# greedy_corners_agent.py
#
#

from agents.agent import Agent
from store import register_agent
from helpers import get_valid_moves, execute_move
import copy
import random
import numpy as np

@register_agent("greedy_corners_agent")
class StudentAgent(Agent):
    """
    A simple Ataxx agent using a greedy heuristic: maximize piece difference, corner control, and minimize opponent mobility.
    """

    def __init__(self):
        super().__init__()
        self.name = "greedy_corners_agent"

    def step(self, board, color, opponent):
        """
        Choose a move based on a simple Ataxx heuristic.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
        - opponent: Integer representing the opponent's color.

        Returns:
        - MoveCoordinates: The chosen move.
        """
        # Get all legal moves for the current player
        legal_moves = get_valid_moves(board, color)

        if not legal_moves:
            return None  # No valid moves available, pass turn

        # Apply heuristic: maximize piece difference, corner control, and minimize opponent mobility
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)
            # evaluate by piece difference, corner bonus, and opponent mobility
            move_score = self.evaluate_board(simulated_board, color, opponent)

            if move_score > best_score:
                best_score = move_score
                best_move = move

        # Return the best move found (or random fallback)
        return best_move or random.choice(legal_moves)

    def evaluate_board(self, board, color, opponent):
        """
        Evaluate the board state based on multiple factors.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
        - player_score: Score of the current player.
        - opponent_score: Score of the opponent.

        Returns:
        - int: The evaluated score of the board.
        """
        # piece difference
        player_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        score_diff = player_count - opp_count
        # corner control bonus
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        corner_bonus = sum(1 for (i, j) in corners if board[i, j] == color) * 5
        # penalize opponent mobility
        opp_moves = len(get_valid_moves(board, opponent))
        mobility_penalty = -opp_moves
        return score_diff + corner_bonus + mobility_penalty
