from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Step 1: Load the model and tokenizer from Hugging Face
MODEL_NAME = "polyglots/LLaMA-Continual-Checkpoint-73456"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, resize_model_vocab =139336)

# Load the model for text generation
try:
    # Load the model with ignore_mismatched_sizes argument
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, resize_model_vocab =139336)
except RuntimeError as e:
    print("Error loading model:", e)
    print("Model and checkpoint might have incompatible configurations.")

# Ensure model is in evaluation mode
model.eval()

# Optional: Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Define a function for inference
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate text (adjust max_length, temperature, etc., as needed)
    outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Step 3: Run inference
# Example input (replace with your own text prompt)
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print("Generated Text:", generated_text)
