from world import World, PLAYER_1_NAME, PLAYER_2_NAME
import argparse
from utils import all_logging_disabled
import logging
import numpy as np
import datetime
import os

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_1", type=str, default="random_agent")
    parser.add_argument("--player_2", type=str, default="random_agent")
    parser.add_argument("--board_path", type=str, default=None)
    parser.add_argument(
        "--board_roster_dir",
        type=str,
        default="boards/",
        help="In autoplay mode, the path to a directory containing all board files",
    )
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--display_delay", type=float, default=0.4)
    parser.add_argument("--display_save", action="store_true", default=False)
    parser.add_argument("--display_save_path", type=str, default="plots/")
    parser.add_argument("--autoplay", action="store_true", default=False)
    parser.add_argument("--autoplay_runs", type=int, default=100)
    args = parser.parse_args()
    return args


class Simulator:
    """
    Entry point of the game simulator.

    Parameters
    ----------
    args : argparse.Namespace
    """

    def __init__(self, args):
        self.args = args

        # if board_roster_dir was passed, add all file paths inside it to a list and save here
        if hasattr(self.args, "board_roster_dir") and self.args.board_roster_dir:
            roster_dir = self.args.board_roster_dir
            if isinstance(roster_dir, str) and os.path.isdir(roster_dir):
                self.board_options = [
                    os.path.join(roster_dir, fname)
                    for fname in os.listdir(roster_dir)
                    if fname.endswith(".csv") or fname.endswith(".board")
                ]
            else:
                self.board_options = [] # TODO: Or should these be None?
        else:
            self.board_options = []

    def reset(self, swap_players=False, board_fpath=None):
        """
        Reset the game

        Parameters
        ----------
        swap_players : bool
            if True, swap the players
        board_fpath : str
            if not None, set the board to the layout in the file stored at board_fpath
        """
        if board_fpath is None:
            board_fpath = self.args.board_path
        if swap_players:
            player_1, player_2 = self.args.player_2, self.args.player_1
        else:
            player_1, player_2 = self.args.player_1, self.args.player_2

        self.world = World(
            player_1=player_1,
            player_2=player_2,
            board_fpath=board_fpath,
            display_ui=self.args.display,
            display_delay=self.args.display_delay,
            display_save=self.args.display_save,
            display_save_path=self.args.display_save_path,
            autoplay=self.args.autoplay,
        )

    def run(self, swap_players=False, board_fpath=None):
        self.reset(swap_players=swap_players, board_fpath=board_fpath)
        is_end, p0_score, p1_score = self.world.step()
        while not is_end:
            is_end, p0_score, p1_score = self.world.step()
        logger.info(
            f"Run finished. {PLAYER_1_NAME} player, agent {self.args.player_1}: {p0_score}. {PLAYER_2_NAME}, agent {self.args.player_2}: {p1_score}"
        )
        return p0_score, p1_score, self.world.p0_time, self.world.p1_time

    def autoplay(self):
        """
        Run multiple simulations of the gameplay and aggregate win %
        """
        p1_win_count = 0
        p2_win_count = 0
        p1_times = []
        p2_times = []
        if self.args.display:
            logger.warning("Since running autoplay mode, display will be disabled")
        self.args.display = False
        with all_logging_disabled():
            for i in range(self.args.autoplay_runs):
                swap_players = i % 2 == 0
                board_fpath = self.board_options[ np.random.randint(len(self.board_options)) ] 
                p0_score, p1_score, p0_time, p1_time = self.run(
                    swap_players=swap_players, board_fpath=board_fpath
                )
                if swap_players:
                    p0_score, p1_score, p0_time, p1_time = (
                        p1_score,
                        p0_score,
                        p1_time,
                        p0_time,
                    )
                if p0_score > p1_score:
                    p1_win_count += 1
                elif p0_score < p1_score:
                    p2_win_count += 1
                else:  # Tie
                    p1_win_count += 0.5
                    p2_win_count += 0.5
                p1_times.extend(p0_time)
                p2_times.extend(p1_time)

        logger.info(
            f"Player 1, agent {self.args.player_1}, win percentage: {p1_win_count / self.args.autoplay_runs}. Maximum turn time was {np.round(np.max(p1_times),5)} seconds."
        )
        logger.info(
            f"Player 2, agent {self.args.player_2}, win percentage: {p2_win_count / self.args.autoplay_runs}. Maximum turn time was {np.round(np.max(p2_times),5)} seconds."
        )

        """
        The code in this comment will be part of the book-keeping that we use to score the end-of-term tournament. FYI. 
        Uncomment and use it if you find this book-keeping helpful.
        fname = (
            "tournament_results/"
            + self.world.player_1_name
            + "_vs_"
            + self.world.player_2_name
            + "_at_"
            + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            + ".csv"
        )
        with open(fname, "w") as fo:
            fo.write(f"P1Name,P2Name,NumRuns,P1WinPercent,P2WinPercent,P1RunTime,P2RunTime\n")
            fo.write(
                f"{self.world.player_1_name},{self.world.player_2_name},{self.args.autoplay_runs},{p1_win_count / self.args.autoplay_runs},{p2_win_count / self.args.autoplay_runs},{np.round(np.max(p1_times),5)},{np.round(np.max(p2_times),5)}\n"
            )
        """

if __name__ == "__main__":
    args = get_args()
    simulator = Simulator(args)
    if args.autoplay:
        simulator.autoplay()
    else:
        simulator.run()
