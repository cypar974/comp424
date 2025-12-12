# Monolith: The Unbreakable Ataxx Agent 🗿

"It doesn't just play. It builds."

## Monolith is a highly optimized Ataxx agent developed for the COMP 424 Final Project. It represents the culmination of genetic algorithm training, advanced search optimization, and strategic analysis.

## Performance Highlights

97.5% Win Rate vs. Hard-Coded Heuristics ("Internet Agent")

72.5% Win Rate vs. Depth 2 Alpha-Beta Agents

Optimized Speed: Custom move generation is ~300% faster than standard helpers.

Deep Search: Consistently reaches Depth 4-6 in the 2-second time limit.

## The Brain: "The Architect" Strategy

Monolith's strategy was not hard-coded; it was learned. Over 200 generations of self-play using a genetic algorithm, the agent discovered a counter-intuitive but powerful playstyle:

Structure over Material: Unlike standard agents that greedily capture pieces, Monolith prioritizes 2x2 "Quad" formations.

Insight: A 2x2 block of friendly pieces is incredibly difficult for an opponent to flip completely. It acts as an anchor for the rest of the game.

The "Edge Trap": Early training suggested hugging the edges was safe. Monolith eventually rejected this, realizing that edges limit mobility. It learned to control the "inner ring" of the board instead.

Mobility is King: The agent places a high value on keeping its options open, ensuring it never gets trapped in a corner.

Genetic Evolution Stats

Generation 1: Random flailing.

Generation 6: The "Edge Hugger" phase (High Edge Weight).

Generation 193: The "Architect" is born. Edge weights drop, Quad weights explode.

## The Body: Technical Optimizations

Monolith isn't just smart; it's fast. To achieve tournament-level performance in Python, we implemented several critical optimizations:

1. Internal Fast Move Generation

We bypassed the standard helpers.py library to write a custom, vectorized move generator using NumPy.

Result: Move generation time reduced by ~70%.

Impact: This speed boost allows Monolith to search 1-2 ply deeper than opponents using the default helper functions.

2. Iterative Deepening Alpha-Beta

Instead of a fixed depth search, Monolith uses Iterative Deepening.

It searches Depth 1, then Depth 2, then Depth 3...

If the 2-second timer is about to expire, it instantly returns the best move from the last completed depth.

Benefit: It never times out, and it always uses 100% of the available thinking time.

3. Transposition Tables

We cache the results of board states we have already seen.

If the search encounters a board position it analyzed 2 seconds ago (via a different move order), it retrieves the score instantly instead of re-calculating.

## Project Structure

student_agent.py: The Main Submission. Contains the Monolith class, the optimized move generator, and the learned weights.

train.py: The Genetic Algorithm engine used to evolve the weights.

trainable_agent.py: A flexible agent wrapper used during the training process.

tournament_runner.py: A multi-core tournament script used to validate Monolith against other agents.

## Credits

Developed by Cyprien, Austin and Emily for COMP 424.

