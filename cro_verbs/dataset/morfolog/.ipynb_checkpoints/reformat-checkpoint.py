import argparse

def reformat_line(line):
    words = line.strip().split(';')
    verb = words[0]
    conjugation = words[1]    
    return f'{verb}\tV;PRS;NOM(3;SG)\t{conjugation}'

def reformat_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    reformatted_lines = [reformat_line(line) for line in lines]

    with open(output_file, 'w') as file:
        file.write('\n'.join(reformatted_lines))  # Add new line character between lines

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Reformat lines in a file.')
parser.add_argument('input_file', help='path to the input file')
parser.add_argument('output_file', help='path to the output file')
args = parser.parse_args()

# Reformat the file
reformat_file(args.input_file, args.output_file)
