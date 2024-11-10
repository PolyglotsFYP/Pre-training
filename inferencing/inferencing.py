from unsloth import FastLanguageModel
from transformers import AutoTokenizer  # Ensure you have this imported if using it
import torch

# Model and tokenizer paths
model_name = "polyglots/LLaMA-Continual-Checkpoint-73456"
tokenizer_name = "polyglots/Extended-Sinhala-LLaMA"
dtype = torch.float16

# Load model and tokenizer
model = FastLanguageModel.from_pretrained(
    model_name=model_name,
    dtype=dtype
).half()  # Convert model to half precision for faster inference if supported

# Load the tokenizer using AutoTokenizer for compatibility
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Move model to appropriate device (CPU or GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define an inference function
def generate_text(prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate text
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "අපි කාටත් ජීවිතයේ ඇති"
generated_text = generate_text(prompt)
print("Generated Text:", generated_text)
