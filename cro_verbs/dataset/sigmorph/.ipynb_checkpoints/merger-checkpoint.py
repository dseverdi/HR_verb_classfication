import csv
import argparse

def merge_csv_files(file1, file2, output_file):
    # Read the first CSV file
    with open(file1, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        rows_file1 = list(csv_reader)

    # Read the second CSV file
    with open(file2, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        rows_file2 = list(csv_reader)

    # Create a dictionary to store the rows from the second file
    rows_dict = {}
    for row in rows_file2:
        key = tuple(row[:2])  # Use first two columns as the key
        value = row[-1]  # Use last column as the value
        rows_dict[key] = value

    # Merge the rows from both files
    merged_rows = []
    for row1 in rows_file1:
        key = tuple(row1[:2])  # Use first two columns as the key
        if key in rows_dict:
            merged_row = row1 + [rows_dict[key]]  # Add the corresponding value from the second file
            merged_rows.append(merged_row)

    # Write the merged data to a new CSV file
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerows(merged_rows)

    print(f"The merged file '{output_file}' has been created successfully.")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Merge two CSV files.')
parser.add_argument('file1', help='Path to the first CSV file')
parser.add_argument('file2', help='Path to the second CSV file')
parser.add_argument('output_file', help='Path to the output merged CSV file')
args = parser.parse_args()

# Call the function to merge the files
merge_csv_files(args.file1, args.file2, args.output_file)
