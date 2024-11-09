from unsloth import FastLanguageModel
import torch

# Load the model and tokenizer for translation inference
model_name = "polyglots/LLaMA-Continual-Checkpoint-73456"  # Replace with your model's name or path
dtype = torch.float16  # Use float16 for efficiency
load_in_4bit = True  # Load model in 4-bit precision for memory efficiency

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    dtype=dtype,
    resize_model_vocab = 139336  # Resize vocabulary to match your model's settings
)

# Enable faster inference
FastLanguageModel.for_inference(model)

# Define a simple translation function
def translate_text(prompt, max_new_tokens=64):
    """Translates an English prompt to Sinhala."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
english_prompt = "Life is full of beautiful moments."  # English text to translate
translated_text = translate_text(english_prompt)
print("Translated Text:", translated_text)
