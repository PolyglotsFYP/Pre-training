import math

def count_lines_in_file(file_path):
    # Open the file with UTF-8 encoding to avoid Unicode errors
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read all lines and count them
        lines = file.readlines()
        return len(lines), lines

def save_dataset_in_chunks(file_path, output_file_prefix, chunk_size=500000):
    # Get the total number of lines and the actual lines
    num_lines, lines = count_lines_in_file(file_path)
    
    # Calculate the number of chunks required (each chunk having up to 500,000 lines)
    num_chunks = math.ceil(num_lines / chunk_size)
    
    for i in range(num_chunks):
        # Determine the start and end line indices for this chunk
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_lines)
        
        # Create an output file name for this chunk (e.g., "output_file_prefix_1.txt", "output_file_prefix_2.txt", etc.)
        output_file = f"{output_file_prefix}_{i + 1}.txt"
        
        # Save the chunk to a new file
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for line in lines[start_idx:end_idx]:
                out_file.write(line)
        
        print(f"Saved lines {start_idx + 1} to {end_idx} to {output_file}.")
    
    print(f"Total lines: {num_lines}. Saved in {num_chunks} files.")

# Example usage
file_path = './train-CulturaX.si-1.txt'
output_file_prefix = './output_file'
save_dataset_in_chunks(file_path, output_file_prefix)