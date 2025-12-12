# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import random
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves
from collections import defaultdict

"""
A basic MCTS implementation with a choose-best-leaf expansion heuristic.
This agent beats random agent 100% of the time but loses to greedy corners 100% of the time
"""

# Get the opposite player given the current one
PLAYER_SWAP_DICT = {1: 2, 2: 1}

MAX_DEPTH = 2


def evaluate_action(self, board, action, color):
    """
    Evaluate the board state based on multiple factors.

    Parameters:
    - board: 2D numpy array representing the game board. - before the action occurs
    - action: the MoveCoords to execute
    - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).

    Returns:
    - int: The evaluated score of the board.
    """
    opponent = PLAYER_SWAP_DICT[color]
    # Execute action
    next_state = deepcopy(board)
    execute_move(next_state, action, color)

    # piece difference
    player_count = np.count_nonzero(next_state == color)
    opp_count = np.count_nonzero(next_state == opponent)
    score_diff = player_count - opp_count

    # corner control bonus
    n = next_state.shape[0]
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    corner_bonus = sum(1 for (i, j) in corners if next_state[i, j] == color) * 5

    # penalize opponent mobility
    opp_moves = len(get_valid_moves(next_state, opponent))
    mobility_penalty = -opp_moves
    return score_diff + corner_bonus + mobility_penalty


def beta_sample(alpha, beta):
    """
    Generate a sample from a Beta distribution using numpy.
    """
    x = np.random.gamma(alpha, 1)
    y = np.random.gamma(beta, 1)
    return x / (x + y)


# This implementation is inspired by https://ai-boson.github.io/mcts/
class MCTSNode:
    def __init__(
        self, state, agent_player, current_player, parent=None, parent_action=None
    ):
        self.state = state
        self.wins = 0
        self.total = 0
        self.parent = parent
        self.agent_player = agent_player
        self.current_player = current_player  # Which player's turn it is in this node
        self.parent_action = parent_action  # The action carried out by the parent node to reach the current state
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[0] = 0
        self._number_of_visits = 0
        # Let's sort the untried actions by point differential
        self._untried_actions = [
            (
                move,
                evaluate_action(
                    self.state, board=self.state, action=move, color=self.current_player
                ),
            )
            for move in get_valid_moves(state, current_player)
        ]
        self._untried_actions.sort(key=lambda x: x[1])  # Sort by the evaluation score
        self.children = []

    def expand(self):
        # Calculate scores for each action
        scores = [action[1] for action in self._untried_actions]

        # Normalize scores to [0, 1]
        min_score = np.min(scores)
        max_score = np.max(scores)
        normalized_scores = (scores - min_score) / (max_score - min_score + 1e-8)

        # Calculate alpha and beta for each action based on normalized scores
        alpha = 1 + normalized_scores  # Successes
        beta_params = 1 + 1 - normalized_scores  # Failures (1 - normalized score)

        # Sample from the Beta distribution for each action
        sampled_values = [beta_sample(a, b) for a, b in zip(alpha, beta_params)]
        selected_index = np.argmax(
            sampled_values
        )  # Select the action with the highest sampled value

        # Pop the selected action
        action = self._untried_actions.pop(selected_index)
        next_state = deepcopy(self.state)
        execute_move(next_state, action[0], self.current_player)

        child_node = MCTSNode(
            next_state,
            agent_player=self.agent_player,
            current_player=PLAYER_SWAP_DICT[self.current_player],
            parent=self,
            parent_action=action[0],
        )
        self.children.append(child_node)

        return child_node

    def is_terminal_node(self):
        is_endgame, p0_score, p1_score = check_endgame(self.state)
        return is_endgame

    def q(self):
        wins = self._results[1]
        losses = self._results[-1]
        return wins - losses

    def n(self):
        return self._number_of_visits

    def rollout(self):
        # This function returns the reward as a dictionary which maps from the player number to the reward for that player

        current_rollout_state = deepcopy(self.state)
        current_rollout_player = self.current_player

        rollout_depth = 0
        is_endgame, p0_score, p1_score = check_endgame(current_rollout_state)
        initial_score_differential = p0_score - p1_score

        while True:
            # Check if rollout has reached a terminal state
            is_endgame, p0_score, p1_score = check_endgame(current_rollout_state)
            possible_moves = get_valid_moves(
                current_rollout_state, current_rollout_player
            )
            if is_endgame or (len(possible_moves) < 1):
                if p0_score > p1_score:
                    return {1: 1, 2: 0}
                else:  # Treat ties as a win for P2
                    return {1: 0, 2: 1}

            # If we hit the maximum depth limit, let's use disc count differential as a heuristic
            if rollout_depth > MAX_DEPTH:
                score_differential = p0_score - p1_score
                if score_differential > initial_score_differential:
                    return {1: 1, 2: 0}
                else:
                    return {1: 0, 2: 1}

            action = self.rollout_policy(possible_moves)

            execute_move(current_rollout_state, action, current_rollout_player)
            current_rollout_player = PLAYER_SWAP_DICT[current_rollout_player]

    def backpropagate(self, result):
        # We expect the result to be a dictionary which maps from player number to the reward for that player
        self._number_of_visits += 1.0
        self._results[result[self.current_player]] += 1.0
        if self.parent:
            self.parent.backpropagate(result)

    # Check if all actions have been tried
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        # Wait does it pick the worst move TODO TODO TODO
        choices_weights = [
            (c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n()))
            for c in self.children
        ]
        if self.current_player == self.agent_player:
            return self.children[np.argmin(choices_weights)]
        else:
            return self.children[np.argmax(choices_weights)]

    # Rollout policy is to randomly select a move
    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():
            ## Let's improve the lookahead heuristic by being smart about which node we expand
            if current_node.is_fully_expanded():
                try:
                    current_node = current_node.best_child()
                except:
                    return current_node
            else:
                # TODO: Maybe we want to have some randomness here so we don't explore low-heuristic scorer values
                return current_node.expand()

        return current_node

    def best_action(self):
        simulation_no = 30
        rollout_no = 1

        for i in range(simulation_no):

            v = self._tree_policy()
            for j in range(rollout_no):
                reward = v.rollout()
                v.backpropagate(reward)

        best_child = self.best_child(c_param=0.0)
        return best_child.parent_action


@register_agent("mcts_agent")
class StudentAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "MCTSAgent"

    def step(self, chess_board, player, opponent):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (board_size, board_size)
          where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
          and 2 represents Player 2's discs (Brown).
        - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
        - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

        You should return a tuple (r,c), where (r,c) is the position where your agent
        wants to place the next disc. Use functions in helpers to determine valid moves
        and more helpful tools.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Simple MCTS-ish search:
        start_time = time.time()

        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            # No valid moves
            return None

        tree_root = MCTSNode(chess_board, player, player)
        best_move = tree_root.best_action()

        time_taken = time.time() - start_time

        print("MCTS Agent's turn took ", time_taken, "seconds.")

        return best_move
