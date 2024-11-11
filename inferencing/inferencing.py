from unsloth import FastLanguageModel
import torch

# Load the model and tokenizer for inference
model_name = "polyglots/LLaMA-Continual-Checkpoint-73456"  # Replace with the actual model name or path
dtype = torch.float16  # Use float16 for efficient computation

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    dtype=dtype,
    resize_model_vocab = 139336,
)

# Enable faster inference
FastLanguageModel.for_inference(model)

# Define a simple inference function
def generate_text(prompt, max_new_tokens=512):
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