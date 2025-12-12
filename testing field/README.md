# Ataxx!

**Project Description & Template** : https://www.overleaf.com/read/khdfckprxqtn#a952b1

## Setup
To setup your Python environment, we highly suggest using virtualenv to keep your dependencies in order. Run:
```bash
python3.10 -m venv env_name
```
and then:
```
source env_name/bin/activate
```
to create and activate your virtual environment. Note: You will need to activate your virtual environment every time you start a new shell. It should appear as:
```
(env_name) <username@machine>:~/your/path/to/project$ 
```

Ensure that you have properly installed pip and python in your venv. To check this, you can run:
```
which pip
```
If you get something like:
```
/usr/bin/pip
```
rather than :
```
/path/to/your/venv/bin/pip
```
Then you may need to run:
```
python -m pip install --upgrade pip
```

To setup the game, clone this repository and install the dependencies. Be sure to have your virtual environment activated.
```bash
pip install -r requirements.txt
```

## Playing a game
The simulator requires you to specify which agents should compete against each other. As a starting point, we provide you with several agents that uses the game interface. Following these examples, your goal is to add your own agents using the techniques taught in class. 

For example, to play the game using two copies of the provided random agent (which takes a random action every turn), run the following:
```bash
python simulator.py --player_1 random_agent --player_2 random_agent
```

This will spawn a random game board of size $N \times N$, and run the two agents of class [RandomAgent](agents/random_agent.py). You will be able to see their moves in the console.

## Visualizing a game

To visualize the moves within a game, use the `--display` flag. You can set the delay (in seconds) using `--display_delay` argument to better visualize the steps the agents take to win a game.

```bash
python simulator.py --player_1 random_agent --player_2 random_agent --display
```

## Play on your own!

To take control of one side of the game and compete against the random agent yourself, use a [`human_agent`](agents/human_agent.py) to play the game.

```bash
python simulator.py --player_1 human_agent --player_2 random_agent --display
```

## Autoplaying multiple games
There is some randomness affecting the outcome of the game from the initial layout and agent logic. To fairly evaluate agents, we will run them against each other multiple times, alternating their roles as player_1 and player_2. All board will be of size 7 x 7. The aggregate win percentage will determine a fair winner. Use the `--autoplay` flag to run $n$ games sequentially, where $n$ can be set using `--autoplay_runs`. The default is 100, and will be used for the final player vs. player run.

```bash
python simulator.py --player_1 random_agent --player_2 random_agent --autoplay
```

During autoplay, boards are drawn randomly from `--board_roster_dir` for each iteration. You may test and develop on various setups by providing this board directory path to the command line. However, the defaults will be used during testnig and can be found in the `boards/` folder. Please ensure the timing limits are satisfied for every board in this size range. 

**Notes**

- Not all agents support autoplay (e.g. the human agent doesn't make sense this way). The variable `self.autoplay` in [Agent](agents/agent.py) can be set to `True` to allow the agent to be autoplayed. Typically this flag is set to false for a `human_agent`.
- UI display will be disabled in an autoplay.

## Develop your own general agent(s):

You need to write one agent and submit it for the class project, but you may develop additional agents during the development process to play against each other, gather data or similar. To write a general agent:

1. Modify **ONLY** the [`student_agent.py`](agents/student_agent.py) file in [`agents/`](agents/) directory, which extends the [`agents.Agent`](agents/agent.py) class.
2. Do not add any additional imports.
3. Implement the `step` function with your game logic. Make extensive use of the functions imported from helpers.py which should be the majority of what you need to interact with the game. Any further logic can be coded directly in your file as global or class variables, functions, etc. **Do not import world.py.**
4. Test your performance against the random_agent with:
```
python simulator.py --player_1 student_agent --player_2 random_agent --autoplay
```
5. Try playing against your own bot as a human. Consistently beating your own best-effort human play is a very good indicator of an A grade for the performance section.

## Advanced and optional: What if I want to create other agents and test them against eachother?
There can only be one file called `student_agent.py` which is already perfectly set up to interact with our evaluation code. You may create other agents during development for testing, which requires a few extra setup steps.

Suppose you want to create `second_agent.py`, a second strategy for your student agent:
1. Create the new file by starting from a copy of the provided student agent:
```
cp agents/student_agent.py agents/second_agent.py
```
2. Change the name in the decorator. Edit `@register_agent("student_agent")` to `@register_agent("second_agent")` and the class name from `StudentAgent` to `SecondAgent`. 
3. Import your new agent in the [`__init__.py`](agents/__init__.py) file in [`agents/`](agents/) directory, by adding the line `from .second_agent import SecondAgent`
4. Now you can run your two agents against each other in the simulator.py by running:
```
python simulator.py --player_1 student_agent --player_2 second_agent --display
```
    
Adapt these steps to any number of agents for your different ideas.

## Submission
To wrap up and submit, prepare the strongest player you developed by adding it to the `student_agent.py` file. This will be the only code file submitted for grading.

Here are a few last minute things to double-check, since your agent must follow some special rules to make it possible to run in auto-grading. **Failing to precisely follow the instructions below risks an automatic assignment of "poor" for the performance grade** as we don't have time to debug everyone's solution.

1. Check that you didn't modify anything outside of `student_agent.py`. You can use `git status` and `git diff` to accomplish this.
2. Ensure `student_agent.py` does not have any additional imports.
3. The `StudentAgent` class *must be* decorated with the exact name of `student_agent`. Do not add any comments or change that line at all, as we will be need it to run your agents in the tournament. A common mistake is to copy the contents from a differently named file, e.g.: `best_agent` or similar, without checking.
4. Check that the time limits are satisfied for all board sizes in the range 6-12, inclusive.
5. As a final test before submitting, make 100% sure the player you wish to be evaluated on runs correctly with the exact command we'll use in auto-grading ```python simulator.py --player_1 random_agent --player_2 student_agent --autoplay```

## Full API

```bash
python simulator.py -h       
usage: simulator.py [-h] [--player_1 PLAYER_1] [--player_2 PLAYER_2]
                    [--board_path BOARD_PATH] [--display]
                    [--display_delay DISPLAY_DELAY]

optional arguments:
  -h, --help            show this help message and exit
  --player_1 PLAYER_1
  --player_2 PLAYER_2
  --board_path BOARD_PATH
  --board_roster_dir BOARD_ROSTER_DIR
  --display
  --display_delay DISPLAY_DELAY
  --autoplay
  --autoplay_runs AUTOPLAY_RUNS
```

## GitHub Cloning Instructions
Because you would likely want to create your own private GitHub repository for this project, but are unable to due to this repository being public, I would suggest for you to follow the instructions below:
1. Clone the repository doing:
SSH cloning
```
git clone git@github.com:davaus80/COMP424-Fall2025.git
```
or HTTPS cloning:
```
git clone https://github.com/davaus80/COMP424-Fall2025.git
```
2. Create your own private repository on GitHub and add your partners following: https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository and https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository
3. Add this repository as a remote repository:
```
git remote add private <your-github-repo-link>
```
4. Push to your remote as:
```
git push private main
```
And then you should continue using this as you update your changes!

## Maximum Memory Measurements
Measure using:
```
/usr/bin/time -v python simulator.py --player_1 random_agent --player_2 <your-agent> --autoplay
```
and then you should see something like:
```
Command being timed: "python simulator.py --player_1 random_agent --player_2 random_agent --autoplay"
        User time (seconds): 8.77
        System time (seconds): 0.28
        Percent of CPU this job got: 35%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:25.60
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 61056
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 188
        Minor (reclaiming a frame) page faults: 23247
        Voluntary context switches: 1682
        Involuntary context switches: 687
        Swaps: 0
        File system inputs: 48512
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
```
where the max resident set size is the most RAM that you have used (in kilobytes).

## Issues? Bugs? Questions?

Feel free to open an issue in this repository, or contact us in Ed thread.

## About

This is a class project for COMP 424, McGill University, Fall 2025 (it was originally forked with the permission of Jackie Cheung and David Meger).

## License

[MIT](LICENSE)
