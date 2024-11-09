from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
import torch

# Step 1: Load the model and tokenizer from Hugging Face
MODEL_NAME = "polyglots/LLaMA-Continual-Checkpoint-73456"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the model (update model class based on your task, e.g., AutoModelForCausalLM or AutoModelForSequenceClassification)
# For text generation
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# For sequence classification, use AutoModelForSequenceClassification
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

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

# For classification tasks, use the pipeline
# classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Step 3: Run inference
# Example input (replace with your own text prompt)
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print("Generated Text:", generated_text)