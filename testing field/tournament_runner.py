import subprocess
import concurrent.futures
import time
import re
import os
import sys
import random

# --- CONFIGURATION ---
TOTAL_GAMES = 40
WORKERS = 10
STUDENT = "cat"
OPPONENT = "student_agent"
DEBUG_MODE = True


def run_game(game_id):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    simulator_path = os.path.join(script_dir, "simulator.py")
    board_dir = os.path.join(script_dir, "boards")
    board_files = [
        f for f in os.listdir(board_dir) if f.endswith(".board") or f.endswith(".csv")
    ]
    random_board = os.path.join(board_dir, random.choice(board_files))

    if game_id % 2 == 0:
        p1, p2 = STUDENT, OPPONENT
        is_student_p1 = True
    else:
        p1, p2 = OPPONENT, STUDENT
        is_student_p1 = False

    cmd = [
        sys.executable,
        simulator_path,
        "--player_1",
        p1,
        "--player_2",
        p2,
        "--board_path",
        random_board,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        output = result.stderr + result.stdout

        match = re.search(
            r"Run finished.*?agent .*?: (\d+).*?agent .*?: (\d+)",
            output,
            re.DOTALL | re.IGNORECASE,
        )

        if match:
            score_p1 = int(match.group(1))
            score_p2 = int(match.group(2))

            if score_p1 > score_p2:
                return 1 if is_student_p1 else -1
            elif score_p2 > score_p1:
                return -1 if is_student_p1 else 1
            else:
                return 0
        else:
            if DEBUG_MODE:
                print(f"\n[DEBUG] Game {game_id} Parse Error!")
                print(f"Command: {' '.join(cmd)}")
                print(f"--- Simulator Output Start ---")
                print(output)
                print(f"--- Simulator Output End ---\n")
            return 0

    except Exception as e:
        if DEBUG_MODE:
            print(f"\n[DEBUG] Game {game_id} Exception: {e}")
        return 0


def main():
    print(f"🎮 Running {TOTAL_GAMES} games on {WORKERS} cores...")
    print(f"⚔️  Matchup: {STUDENT} vs {OPPONENT}")
    print("-" * 50)

    start = time.time()
    results = []
    completed_count = 0

    # Progress tracking variables
    last_update_time = time.time()
    games_started = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        # Submit all games first
        futures = [executor.submit(run_game, i) for i in range(TOTAL_GAMES)]
        games_started = len(futures)

        print(f"🚀 Started {games_started} games...")
        print()

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            completed_count = len(results)

            # Calculate progress
            progress = (completed_count / TOTAL_GAMES) * 100
            elapsed_time = time.time() - start

            # Estimate remaining time
            if completed_count > 0:
                avg_time_per_game = elapsed_time / completed_count
                remaining_games = TOTAL_GAMES - completed_count
                estimated_remaining = avg_time_per_game * remaining_games
            else:
                estimated_remaining = 0

            # Create progress bar
            bar_length = 30
            filled_length = int(bar_length * completed_count // TOTAL_GAMES)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)

            # Get result symbol
            if result == 1:
                result_symbol = "✅ WIN"
            elif result == -1:
                result_symbol = "❌ LOSS"
            else:
                result_symbol = "⚖️  DRAW"

            # Update display every game completion
            print(
                f"📊 Progress: [{bar}] {completed_count}/{TOTAL_GAMES} ({progress:.1f}%)"
            )
            print(
                f"⏱️  Elapsed: {elapsed_time:.1f}s | Remaining: ~{estimated_remaining:.1f}s"
            )
            print(f"🎯 Last Game: {result_symbol}")

            # Show running statistics
            wins = results.count(1)
            losses = results.count(-1)
            draws = results.count(0)

            if completed_count > 0:
                win_rate = (wins / completed_count) * 100
                print(
                    f"📈 Current Stats: {wins}W {losses}L {draws}D | Win Rate: {win_rate:.1f}%"
                )
            else:
                print(f"📈 Current Stats: 0W 0L 0D | Win Rate: 0.0%")

            print("-" * 50)

    # Final results
    total_time = time.time() - start
    wins = results.count(1)
    losses = results.count(-1)
    draws = results.count(0)
    win_rate = (wins / TOTAL_GAMES) * 100 if TOTAL_GAMES > 0 else 0

    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS")
    print("=" * 60)
    print(f"✅ Wins: {wins} | ❌ Losses: {losses} | ⚖️  Draws: {draws}")
    print(f"📊 Win Rate: {win_rate:.1f}%")
    print(f"⏱️  Total Time: {total_time:.1f} seconds")

    # Performance summary
    if wins > losses:
        print("🎉 Excellent performance! Student agent is winning!")
    elif wins < losses:
        print("💪 Keep improving! More practice needed.")
    else:
        print("🤝 Very balanced matchup!")


if __name__ == "__main__":
    main()
