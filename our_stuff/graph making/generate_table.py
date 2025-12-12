import pandas as pd
import ast
import os

# Define the names for the weights in order
WEIGHT_NAMES = ["Material", "2x2 Structure", "Mobility", "Corners", "Edges", "Center"]


def load_and_process_data(filenames):
    """
    Loads CSVs, parses the BestWeights string column into lists,
    and separates them into distinct columns.
    """
    processed_dfs = []

    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping.")
            continue

        try:
            df = pd.read_csv(filename)

            # Convert the string representation of list to actual list
            df["BestWeights"] = df["BestWeights"].apply(ast.literal_eval)

            # Expand the list of weights into separate columns with proper names
            weights_df = pd.DataFrame(df["BestWeights"].tolist(), index=df.index)

            # Use friendly names if available, otherwise fall back to Weight_0, Weight_1, etc.
            weight_column_names = []
            for i in range(len(weights_df.columns)):
                if i < len(WEIGHT_NAMES):
                    weight_column_names.append(WEIGHT_NAMES[i])
                else:
                    weight_column_names.append(f"Weight_{i}")

            weights_df.columns = weight_column_names

            # Add a Run column to identify which file this data came from
            run_name = filename.replace(".csv", "").replace("training_history_", "")
            df["Run"] = run_name

            # Combine back with original data
            full_df = pd.concat([df, weights_df], axis=1)
            processed_dfs.append(full_df)
            print(f"Successfully processed {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return processed_dfs


def combine_and_save_data(processed_dfs):
    """
    Combines all processed DataFrames into one and saves as evolution.csv
    """
    if not processed_dfs:
        print("No data to combine.")
        return

    # Combine all DataFrames
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    # Reorder columns to have Run and Gen first, then weights, then other columns
    base_columns = ["Run", "Gen", "BestScore"]
    weight_columns = [
        col
        for col in combined_df.columns
        if col in WEIGHT_NAMES or col.startswith("Weight_")
    ]
    other_columns = [
        col
        for col in combined_df.columns
        if col not in base_columns + weight_columns and col != "BestWeights"
    ]

    # Final column order
    final_columns = base_columns + weight_columns + other_columns

    # Reindex with the desired column order
    combined_df = combined_df[final_columns]

    # Save to CSV
    output_filename = "evolution.csv"
    combined_df.to_csv(output_filename, index=False)
    print(f"Successfully saved combined data to {output_filename}")

    # Print some summary statistics
    print(f"\nSummary:")
    print(f"Total runs: {combined_df['Run'].nunique()}")
    print(f"Total generations: {len(combined_df)}")
    print(f"Runs included: {sorted(combined_df['Run'].unique())}")

    return combined_df


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

    print("Loading and processing data...")
    processed_dfs = load_and_process_data(files)

    if processed_dfs:
        print("\nCombining data into single CSV...")
        combined_data = combine_and_save_data(processed_dfs)
        print("\nDone! Check 'evolution.csv' for the combined data.")
    else:
        print("No valid data found to combine.")


if __name__ == "__main__":
    main()
