from huggingface_hub import HfApi, Repository
import os

# Set your variables
checkpoint_dir = "./checkpoint-30000"
repo_name = "polyglots/LLaMA-Continual-Checkpoint-30000"
hf_username = "AravindaHWK"
hf_token = "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"

# Initialize the API and repository
api = HfApi()
repo_url = api.create_repo(name=repo_name, token=hf_token, exist_ok=True)

# Clone repository (or initialize it if already exists) to checkpoint directory
repo = Repository(local_dir=checkpoint_dir, clone_from=repo_url, use_auth_token=hf_token)

# Add files to the repository
repo.git_add(auto_lfs_track=True)  # Tracks large files with Git LFS automatically if needed
repo.git_commit("Add checkpoint-30000")

# Push the changes
repo.git_push()

print("Checkpoint uploaded successfully to Hugging Face Hub!")
