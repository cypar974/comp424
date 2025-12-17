# Monolith: The Unbreakable Ataxx Agent

**"Depth Beats Heuristic."**

🏆 Placed **9th** overall (out of 80 teams)

## Overview
Monolith is an optimized Ataxx agent developed for the COMP 424 Final Project at McGill University. Our design philosophy was simple: maximize search depth within the 2-second time limit. By combining **Iterative Deepening Alpha-Beta search** with **Genetic Algorithm (GA)** tuned weights, Monolith consistently searches 3-5 ply deep while avoiding timeouts.

## 🏆 Performance Analysis

We validated Monolith through 40-game match sets against various baselines and engines. The results confirmed that our prioritization of search speed over complex heuristics was the correct approach.

* **100% Win Rate** vs. Hard-Mode Online Ataxx Engine (OnlineSoloGames).
* **100% Win Rate** vs. Random and MCTS Agents.
* **100% Win Rate** vs. Classmate Agents (tested against 5 different agents).
* **97.5% Win Rate** vs. Greedy Agent (39-1 record).
* **Search Depth:** Consistently reaches Depth 4 in average positions and Depth 5 in corner-heavy positions.

## 🧠 The Brain: "The Architect"

Monolith's evaluation function wasn't hard-coded—it was evolved. We utilized a **Genetic Algorithm** over nearly 350 generations of self-play to tune our weights.

### The "Depth Beats Heuristic" Pivot
Initially, we focused on complex checks like identifying 2x2 "Quad" structures. However, during training, we realized these checks were computationally expensive in Python. We stripped the evaluation function down to four core pillars, allowing the search to go deeper:
1.  **Material:** Raw piece count.
2.  **Corner Control:** Highly valued (weights 30-55) for stability.
3.  **Positional Control:** Central squares were given negative weights (-8 to -30) by the GA, as the agent learned center pieces are vulnerable to multi-directional attacks.
4.  **Mobility:** Kept simple to avoid bottlenecks.

## ⚙️ The Body: Technical Optimizations

Python is naturally slow for this type of recursion. To achieve tournament-level performance, we implemented several aggressive optimizations:

### 1. Iterative Deepening & Alpha-Beta
Instead of a fixed depth, we use **IDDFS**. The agent searches Depth 1, then 2, then 3... If the 1.92s timer (safety buffer) expires, it immediately returns the best move from the last fully completed depth. This ensures we **never** timeout.

### 2. Transposition Tables
We implemented a hash map to cache evaluated board states. If we encounter a position we analyzed 2 seconds ago (via a different move order), we retrieve the score instantly. This effectively turns our search tree into a graph.

### 3. Heuristic Move Ordering
To maximize Alpha-Beta pruning, we order moves dynamically:
* **Killer Move Heuristic:** We prioritize moves that caused a cutoff at the same depth in previous searches.
* **History Heuristic:** We track which moves have historically been successful across the entire game tree.

### 4. Vectorization & Tuple Arithmetic
We bypassed the provided helper library for critical operations. By using **NumPy** for board scanning and replacing object overhead with **tuple arithmetic**, we increased our thinking speed by roughly **300%**.

## 📂 Project Structure

* `student_agent.py`: The Main Submission. Contains the Monolith class, the optimized custom move generator, IDDFS loop, and the final evolved weights.
* `train.py`: The Genetic Algorithm engine used to evolve the weights (Selection, Crossover, Mutation).
* `tournament_runner.py`: A multi-core script used to validate Monolith against classmates and course baselines.

## 👥 Credits

**COMP 424 - Artificial Intelligence**
Authors: Cyprien Armand, Emily Zhang, Austin Wang
