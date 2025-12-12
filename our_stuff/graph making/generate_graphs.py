import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

# Define the names for the weights in order
WEIGHT_NAMES = ["Material", "2x2 Structure", "Mobility", "Corners", "Edges", "Center"]


def load_and_process_data(filenames):
    """
    Loads CSVs, parses the BestWeights string column into lists,
    and separates them into distinct columns.
    """
    processed_runs = []

    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping.")
            continue

        try:
            df = pd.read_csv(filename)

            # Convert the string representation of list to actual list
            df["BestWeights"] = df["BestWeights"].apply(ast.literal_eval)

            # Expand the list of weights into separate columns (Weight_0, Weight_1, etc.)
            weights_df = pd.DataFrame(df["BestWeights"].tolist(), index=df.index)
            weights_df = weights_df.add_prefix("Weight_")

            # Combine back with original data
            full_df = pd.concat([df, weights_df], axis=1)
            processed_runs.append({"name": filename, "data": full_df})
            print(f"Successfully processed {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return processed_runs


def add_run_separators(ax, runs, current_x_offset_list, y_pos_for_text=None):
    """
    Helper to draw vertical dividers and run labels.
    """
    for i, (run, offset) in enumerate(zip(runs, current_x_offset_list)):
        df = run["data"]
        run_duration = df["Gen"].max()

        # Calculate center for label
        center_x = offset + (run_duration / 2)

        # Draw Divider (Skip for the very end)
        if i < len(runs) - 1:
            divider_x = offset + run_duration
            # Less aggressive divider: thin, gray, dashed
            ax.axvline(
                x=divider_x, color="gray", linewidth=1, linestyle="--", alpha=0.5
            )

        # Add Label
        # If y_pos_for_text is not specified, put it near the top automatically
        if y_pos_for_text is not None:
            text_y = y_pos_for_text
        else:
            # Default to 90% of the visible Y range
            ymin, ymax = ax.get_ylim()
            text_y = ymin + (ymax - ymin) * 0.9

        ax.text(
            center_x,
            text_y,
            f"Run {i}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
        )


def plot_score_evolution(runs):
    """
    Plots the evolution of BestScore for all runs on a continuous timeline.
    """
    if not runs:
        return

    plt.figure(figsize=(14, 8))

    current_x_offset = 0
    offsets = []

    # Define a color for the score line
    score_color = "#1f77b4"  # Standard matplotlib blue

    for i, run in enumerate(runs):
        df = run["data"]
        offsets.append(current_x_offset)

        # Calculate X values for this run
        x_values = df["Gen"] + current_x_offset

        plt.plot(x_values, df["BestScore"], color=score_color, linewidth=2)

        current_x_offset += df["Gen"].max()

    # Formatting
    plt.title("Evolution of Best Score (Continuous)", fontsize=16)
    plt.xlabel("Cumulative Generation", fontsize=12)
    plt.ylabel("Best Score", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Add dividers and labels (Auto-placement for scores since range varies wildly)
    ax = plt.gca()
    add_run_separators(ax, runs, offsets)

    output_filename = "evolution_scores.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Saved score graph to {output_filename}")
    plt.close()


def plot_weight_evolution(runs):
    """
    Plots all weights on a single graph, concatenating runs horizontally.
    """
    if not runs:
        return

    plt.figure(figsize=(14, 8))

    # 1. Identify all unique weight columns
    all_weight_cols = set()
    for run in runs:
        cols = [c for c in run["data"].columns if c.startswith("Weight_")]
        all_weight_cols.update(cols)

    sorted_weights = sorted(list(all_weight_cols), key=lambda x: int(x.split("_")[1]))

    # 2. Distinct Colors (Manual list for high contrast)
    # Mapping specific distinct colors to the 6 known weights
    distinct_colors = [
        "#d62728",  # Material (Red)
        "#2ca02c",  # 2x2 Structure (Green)
        "#1f77b4",  # Mobility (Blue)
        "#9467bd",  # Corners (Purple)
        "#ff7f0e",  # Edges (Orange)
        "#8c564b",  # Center (Brown)
    ]

    # 3. Plot variables
    current_x_offset = 0
    offsets = []

    for i, run in enumerate(runs):
        df = run["data"]
        offsets.append(current_x_offset)

        x_values = df["Gen"] + current_x_offset

        for w_idx, weight_col in enumerate(sorted_weights):
            if weight_col in df.columns:
                # Get mapped name if available, else fallback to column name
                if w_idx < len(WEIGHT_NAMES):
                    friendly_name = WEIGHT_NAMES[w_idx]
                else:
                    friendly_name = weight_col

                # Cycle colors if we have more weights than colors (unlikely here)
                color = distinct_colors[w_idx % len(distinct_colors)]

                # Only label the first run to avoid duplicate legend entries
                label = friendly_name if i == 0 else None

                plt.plot(
                    x_values, df[weight_col], color=color, linewidth=2, label=label
                )

        current_x_offset += df["Gen"].max()

    # 4. Final Formatting
    plt.title("Evolution of Weights Across Runs", fontsize=16)
    plt.xlabel("Cumulative Generation", fontsize=12)
    plt.ylabel("Weight Value", fontsize=12)

    # Add dividers and labels
    # User requested labels between 50 and 80 on Y axis
    ax = plt.gca()
    add_run_separators(ax, runs, offsets, y_pos_for_text=65)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.12, 1))
    plt.tight_layout()

    output_filename = "evolution_weights.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Saved weights graph to {output_filename}")
    plt.close()


def main():
    files = [
        "training_history_run0.csv",
        "training_history_run1.csv",
        "training_history_run2.csv",
        "training_history_run3.csv",
        "training_history_run4.csv",
        "training_history_run5.csv",
        "training_history_run6.csv",
        "training_history_run7.csv",
        "training_history_run8.csv",
        "training_history_run9.csv",
    ]

    print("Loading data...")
    runs_data = load_and_process_data(files)

    if runs_data:
        print("Generating graphs...")
        plot_score_evolution(runs_data)
        plot_weight_evolution(runs_data)
        print("Done! Check 'evolution_scores.png' and 'evolution_weights.png'.")
    else:
        print("No valid data found to plot.")


if __name__ == "__main__":
    main()
