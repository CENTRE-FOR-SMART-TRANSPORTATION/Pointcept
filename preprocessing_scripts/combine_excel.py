import pandas as pd
import glob
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Enter the desired matrix dimension
n = 12

# Function to read the entire first row and first column from an Excel file
def read_first_row_and_column(file_path):
    try:
        df = pd.read_excel(file_path, header=None)
        first_row = df.iloc[0, :].tolist()
        first_column = df.iloc[:, 0].tolist()
        return first_row, first_column
    except Exception as e:
        print(f"Error reading first row and column from {file_path}: {e}")
        return None, None

# Function to read a specific range from an Excel file
def read_matrix_from_excel(file_path):
    try:
        # Read the specific range B2:I13
        df = pd.read_excel(file_path, header=None, usecols="B:M", skiprows=1, nrows=n)
        if df.shape != (n, n):
            print(f"Warning: Matrix from file {file_path} has incorrect shape {df.shape}. Skipping this file.")
            return None
        return df.values
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Define the expected shape of the matrix
expected_shape = (n, n)

# Get all Excel files in the current directory, whose names end with 'num.xlsx'
excel_files = [file for file in glob.glob('*.xlsx') if file.endswith('num.xlsx')]

# Select one file to read the first row and first column from
source_file = excel_files[0] if excel_files else None

if source_file:
    first_row, first_column = read_first_row_and_column(source_file)
    if first_row and first_column:
        # Create a new Excel workbook and select the active worksheet
        wb = Workbook()
        ws_sum = wb.active
        ws_sum.title = "Summed Matrix"
        ws_percentage = wb.create_sheet(title="Relative Percentage")

        # Write the entire first row and first column to the 'Summed Matrix' sheet
        for idx, value in enumerate(first_row, start=1):
            ws_sum.cell(row=1, column=idx, value=value)
        for idx, value in enumerate(first_column, start=1):
            ws_sum.cell(row=idx, column=1, value=value)

        # Write the entire first row and first column to the 'Relative Percentage' sheet
        for idx, value in enumerate(first_row, start=1):
            ws_percentage.cell(row=1, column=idx, value=value)
        for idx, value in enumerate(first_column, start=1):
            ws_percentage.cell(row=idx, column=1, value=value)

        # Initialize a matrix to store the sum
        sum_matrix = None

        # Iterate over all Excel files and accumulate the matrix sum
        for file in excel_files:
            matrix = read_matrix_from_excel(file)
            if matrix is not None:
                if sum_matrix is None:
                    sum_matrix = matrix
                else:
                    sum_matrix += matrix

        if sum_matrix is not None:
            # Convert the sum_matrix to a DataFrame
            sum_df = pd.DataFrame(sum_matrix)

            # Calculate the row sums
            row_sums = sum_df.sum(axis=1)

            # Calculate the relative percentage for each cell
            percentage_df = sum_df.div(row_sums, axis=0) * 100

            # Round the percentage values to 2 decimal places
            percentage_df = percentage_df.round(2)

            # Write the summed matrix to the specific range (B2:K11) in 'Summed Matrix' sheet
            for r_idx, row in enumerate(dataframe_to_rows(sum_df, index=False, header=False), 2):
                for c_idx, value in enumerate(row, 2):
                    ws_sum.cell(row=r_idx, column=c_idx, value=value)

            # Write the percentage matrix to the specific range (B2:K11) in 'Relative Percentage' sheet
            for r_idx, row in enumerate(dataframe_to_rows(percentage_df, index=False, header=False), 2):
                for c_idx, value in enumerate(row, 2):
                    ws_percentage.cell(row=r_idx, column=c_idx, value=value)

            # Save the workbook to a new file
            output_file = 'final.xlsx'
            wb.save(output_file)

            print(f"Summed matrix and relative percentages saved to {output_file}")

