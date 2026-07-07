import pandas as pd

def filter_csv_by_family(input_file, output_file, family_to_remove="proposed"):
    try:
        # Read the dataset into a pandas DataFrame
        df = pd.read_csv(input_file)
        
        # Check if the 'family' column actually exists to prevent errors
        if 'family' not in df.columns:
            print("Error: The column 'family' does not exist in the provided CSV.")
            return

        # Keep only the rows where the 'family' column does NOT equal 'proposed'
        filtered_df = df[df['family'] != family_to_remove]
        
        # Save the filtered DataFrame to a new CSV file without the index column
        filtered_df.to_csv(output_file, index=False)
        
        # Output the results
        print(f"Processing complete!")
        print(f"Original row count: {len(df)}")
        print(f"Filtered row count: {len(filtered_df)}")
        print(f"Removed {len(df) - len(filtered_df)} records.")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Make sure it's in the same directory as this script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function on your specific file
input_filename = 'summary.csv'
output_filename = 'summary.csv'

filter_csv_by_family(input_filename, output_filename)