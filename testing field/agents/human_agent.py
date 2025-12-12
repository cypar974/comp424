# Human Input agent
import sys

from agents.agent import Agent
from store import register_agent
from helpers import MoveCoordinates, check_move_validity


@register_agent("human_agent")
class HumanAgent(Agent):
    def __init__(self):
        super(HumanAgent, self).__init__()
        self.name = "HumanAgent"

    def step(self, chess_board, player, opponent):
        """
        Get human input for the position to place the disc

        Parameters
        ----------
        chess_board : numpy.ndarray of shape (board_size, board_size)
            The chess board with 0 representing an empty space, 1 for black (Player 1),
            and 2 for white (Player 2).
        player : int
            The current player (1 for black, 2 for white).
        opponent : int
            The opponent player (1 for black, 2 for white).

        Returns
        -------
        move_coords : MoveCoordinates instance
            The positions of the source and destination
        """
        text = input("Your move (src_row, src_column, dest_row, dest_column) or input q to quit: ")
        while len(text.split(",")) != 4 and "q" not in text.lower():
            print("Wrong Input Format! Input should be row,column.")
            text = input("Your move (src_row, src_column, dest_row, dest_column) or input q to quit: ")

        if "q" in text.lower():
            print("Game ended by user!")
            sys.exit(0)

        x_src, y_src, x_dest, y_dest = text.split(",")
        x_src, y_src, x_dest, y_dest = int(x_src.strip()), int(y_src.strip()), int(x_dest.strip()), int(y_dest.strip())
        move_coords = MoveCoordinates(src=(x_src, y_src), dest=(x_dest, y_dest))

        while not check_move_validity(chess_board, move_coords, player):
            print(
                "Invalid Move! Each (row,column) pair should be within the board and the destination position must be empty."
            )
            text = input("Your move (src_row, src_column, dest_row, dest_column) or input q to quit: ")
            while len(text.split(",")) != 4 and "q" not in text.lower():
                print("Wrong Input Format! Input should be row,column.")
                text = input("Your move (src_row, src_column, dest_row, dest_column) or input q to quit: ")

            if "q" in text.lower():
                print("Game ended by user!")
                sys.exit(0)

            x_src, y_src, x_dest, y_dest = text.split(",")
            x_src, y_src, x_dest, y_dest = int(x_src.strip()), int(y_src.strip()), int(x_dest.strip()), int(y_dest.strip())
            move_coords = MoveCoordinates(src=(x_src, y_src), dest=(x_dest, y_dest))

        return move_coords