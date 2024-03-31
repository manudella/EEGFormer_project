import pandas as pd
import os

folder_path = "./data"

for foldername in os.listdir(folder_path):
        if foldername.startswith("sub"):
            # Process both session 0 and session 1 if present
            for session in ["ses-0", "ses-1"]:
                subject_dir = os.path.join(folder_path, foldername, session, "beh")
                if os.path.isdir(subject_dir):
                    for filename in os.listdir(subject_dir):
                        if filename.endswith(".tsv"):
                            filepath = os.path.join(subject_dir, filename)
                            
                            # Load the TSV file
                            df = pd.read_csv(filepath, sep='\t')
                            print(df)
                            # Filter the data to include only the columns you need
                            print(df.columns)
                            words_df = df[['onset','item_name','duration']]

                            # Optionally, you might want to rename the columns to match what's expected in the preprocessing code
                            words_df.rename(columns={'onset': 'start_time', 'item_name': 'label'}, inplace=True)

                            # Save the filtered data to a new TSV file for later use in preprocessing
                            filtered_tsv_path = os.path.join(subject_dir, filename.split('.')[0] + '_filtered.tsv')
                            words_df.to_csv(filtered_tsv_path, sep='\t', index=False)
                            print(words_df)

                            print(f"Filtered words data saved to: {filtered_tsv_path}")
