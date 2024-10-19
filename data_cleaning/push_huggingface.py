import os
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import login

# Function to create the dataset from a text file
def create_dataset_from_txt(file_path, lang, src):
    # Read the file and store each line (sentence) as an entry
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Strip whitespace from each line and remove empty lines
    sentences = [line.strip() for line in lines if line.strip()]

    # Create a Hugging Face dataset
    dataset = Dataset.from_dict({"text": sentences, "lang": [lang] * len(sentences), "src": [src] * len(sentences)})
    
    return dataset

# Function to append data to an existing dataset and push to Hugging Face
def append_and_push_to_huggingface(new_dataset, dataset_name, user_name):
    # Load the existing dataset from Hugging Face
    try:
        existing_dataset = load_dataset(f"{user_name}/{dataset_name}")
        existing_dataset = existing_dataset['train']  # Assuming the dataset has a 'train' split

        # Concatenate the existing dataset with the new data
        updated_dataset = concatenate_datasets([existing_dataset, new_dataset])
        print(f"Existing dataset found and data has been appended. Total size: {len(updated_dataset)} samples.")
    except Exception as e:
        print(f"No existing dataset found, or an error occurred: {e}")
        print("Creating a new dataset.")
        updated_dataset = new_dataset  # If no dataset exists, use the new dataset
    
    # Push the updated dataset to Hugging Face
    updated_dataset.push_to_hub(f"{user_name}/{dataset_name}")

dataset_name = 'CulturaX-Clean-Corpus'  # Name of the dataset on Hugging Face
user_name = 'polyglots'  # Hugging Face username

# Call the login function to authenticate
login()

for i in range(2, 19):
    # Create the new dataset
    new_dataset = create_dataset_from_txt(f'./output_file_{i}.txt', lang='si', src='CulturaX')

    # Append to the existing dataset and push to Hugging Face
    append_and_push_to_huggingface(new_dataset, dataset_name, user_name)

print(f"Dataset {dataset_name} has been updated and uploaded to Hugging Face Hub.")