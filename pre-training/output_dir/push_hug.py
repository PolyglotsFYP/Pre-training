from huggingface_hub import Repository, HfApi

# Set up parameters
repo_url = "https://huggingface.co/polyglots/LLaMA-Continual-Checkpoint-30000"  # Replace with your Hugging Face Hub repo URL
local_dir = "./checkpoint-30000"  # Path to the local folder with your files
commit_message = "Upload large files to repository"

# Authenticate using your Hugging Face token
hf_token = "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"  # Replace with your Hugging Face token

# Enable large file support for files >5GB
HfApi().lfs_enable_largefiles(repo_url, hf_token=hf_token)

# Initialize repository
repo = Repository(local_dir=local_dir, clone_from=repo_url, use_auth_token=hf_token)

# Pull the latest changes from the remote repository
repo.git_pull()

# Add, commit, and push all files in the folder to the Hub
repo.push_to_hub(commit_message=commit_message)