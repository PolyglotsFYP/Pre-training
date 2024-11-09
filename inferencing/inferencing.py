from unsloth import FastLanguageModel
import torch

# Load the model and tokenizer for inference
model_name = "polyglots/LLaMA-Continual-Checkpoint-73456"  # Replace with the actual model name or path
max_seq_length = 512 # Example sequence length, adjust as needed
dtype = torch.float16  # Use float16 for efficient computation
load_in_4bit = True  # Load model in 4-bit precision for memory efficiency

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    resize_model_vocab = 139336
)

# Enable faster inference
FastLanguageModel.for_inference(model)

# Define a simple inference function
def generate_text(prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "අපි කාටත් ජීවිතයේ ඇති"
generated_text = generate_text(prompt)
print("Generated Text:", generated_text)
