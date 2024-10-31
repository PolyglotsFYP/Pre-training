def read_and_write_first_1M_chars(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        # Read only the first 1 million characters
        content = input_file.read(10**6)

    # Write the content to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(content)

input_file_path = './filtered_output.txt'
output_file_path = './output_1M.txt'
read_and_write_first_1M_chars(input_file_path, output_file_path)