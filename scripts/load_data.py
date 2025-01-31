import pandas as pd

def read_csv_file(file_path):
    # Load a CSV file
    try:
        data = pd.read_csv(file_path,)
         #remove unnamed columns
        data=data.loc[:, ~data.columns.str.contains('^unnamed', case=False)]
        print(f"Dataset loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading file: {e}")
        raise