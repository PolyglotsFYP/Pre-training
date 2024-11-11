# Import dependencies
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
import pandas as pd
import os
from datasets import load_dataset
import pyrebase
import sys

# Load the base model
max_seq_length = 2048
dtype = None  # Set to Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False
model_name = "polyglots/LLaMA-Continual-Checkpoint-73456"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    resize_model_vocab=139336
)

# Load the datasets
datasets = {
    "Sinhala-NER": load_dataset("polyglots/Sinhala-NER", split="test"),
    "Sinhala-Classification": load_dataset("polyglots/Sinhala-Classification", split="test")
}

# Function to create message format
def create_message(data_point):
    instruction = data_point['instructions']
    input_text = data_point['input']
    return f"Instruction: {instruction}\n\nInput: {input_text}\n\nResponse: "

# Initialize model for inference
FastLanguageModel.for_inference(model)
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# Function to generate model response
def get_model_response(message):
    inputs = tokenizer(message, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        streamer=text_streamer
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True).split("Response: ")[-1].strip()
    return response

# Suppress output
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Create output folder
os.makedirs("responses", exist_ok=True)

# Process datasets and save to CSV
for dataset_name, dataset in datasets.items():
    model_responses = []
    csv_data = []
    print(f"Processing {dataset_name} dataset..., length: {len(dataset)}")

    for i in range(len(dataset)):
        if i % 100 == 0:
            print(f"Processing {i}...")
        data_point = dataset[i]
        message = create_message(data_point)
        response = get_model_response(message)
        model_responses.append(response)

        csv_data.append({
            "Instruction": data_point['instructions'],
            "Input": data_point['input'],
            "Desired Output": data_point['output'],
            "Model Response": response
        })

    df = pd.DataFrame(csv_data)
    output_file = f"responses/{dataset_name}_responses_basemodel.csv"
    df.to_csv(output_file, index=False)
    print(f"Responses saved to {output_file}")

# Restore stdout and stderr
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
