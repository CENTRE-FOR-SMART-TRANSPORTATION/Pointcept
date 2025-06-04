import openpyxl

def copy_data(source_sheet, target_sheet, source_range, target_cell):
    total_row_sums = {}  # Store the sum of each row
    
    for row_num, row in enumerate(source_sheet[source_range], start=1):
        row_sum = sum(cell.value or 0 for cell in row)
        total_row_sums[row_num] = row_sum  # Store the row sum for later use
        
        for cell in row:
            # Calculate the percentage
            percentage = cell.value / row_sum * 100 if row_sum != 0 else 0
            target_sheet[target_cell].value = percentage
            target_cell = chr(ord(target_cell[0]) + 1) + target_cell[1]

def main():
    # List of Excel files to read
    excel_files = ["file1.xlsx", "file2.xlsx", "file3.xlsx"]

    # Open target Excel file where data will be added
    target_workbook = openpyxl.load_workbook("target.xlsx")
    target_sheet = target_workbook.active

    # Define the target starting cell
    target_cell = 'E2'

    for file in excel_files:
        # Open each source Excel file
        source_workbook = openpyxl.load_workbook(file, data_only=True)
        source_sheet = source_workbook.active

        # Define the source range (10x11 grid)
        source_range = 'B2:L11'

        # Copy data from source to target
        copy_data(source_sheet, target_sheet, source_range, target_cell)

        # Adjust target cell for the next file
        target_cell = chr(ord(target_cell[0]) + 11) + target_cell[1]

    # Save the changes to the target Excel file
    target_workbook.save("target.xlsx")

if __name__ == "__main__":
    main()
