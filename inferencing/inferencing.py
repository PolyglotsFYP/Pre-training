import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# Configuration for paths
base_model_path = "polyglots/LLaMA-Continual-Checkpoint-73456"  # Replace with your model path
tokenizer_path = "polyglots/Extended-Sinhala-LLaMA"

# Load model and tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Check and adjust embedding size if necessary
model_vocab_size = model.get_input_embeddings().weight.size(0)
tokenizer_vocab_size = len(tokenizer)

print(f"Model Embedding Size: {model_vocab_size}")
print(f"Tokenizer Vocabulary Size: {tokenizer_vocab_size}")

if model_vocab_size != tokenizer_vocab_size:
    print("Adjusting model embeddings to match tokenizer vocabulary size...")
    model.resize_token_embeddings(tokenizer_vocab_size)

# Define the prompt and example input in Sinhala
prompt = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)

instruction_text = "අපි කාටත් ජීවිතයේ ඇති"

# Generate the input text by formatting with the instruction
input_text = prompt.format(instruction=instruction_text)

# Tokenize and move to the appropriate device
inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

# Generate output
with torch.no_grad():
    output = model.generate(input_ids=inputs["input_ids"], max_new_tokens=50)
    
# Decode the generated tokens to get the response in Sinhala
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
response_text = generated_text.split("### Response:")[-1].strip()

# Print the input and generated response
print(f"Input: {instruction_text}")
print(f"Generated Response: {response_text}")
