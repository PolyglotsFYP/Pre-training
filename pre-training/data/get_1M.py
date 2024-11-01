def read_and_write_first_1M_lines(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:
        
        # Iterate through lines and write only the first 1 million lines
        for i, line in enumerate(input_file):
            if i >= 10**5:  # Stop after 1 million lines
                break
            output_file.write(line)

input_file_path = './filtered_output.txt'
output_file_path = './output_1L_lines.txt'
read_and_write_first_1M_lines(input_file_path, output_file_path)