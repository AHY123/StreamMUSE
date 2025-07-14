from huggingface_hub import snapshot_download

repo_id = "S-tanley/Abnormal"
local_dir = "abnormal_model"

print(f"Downloading model from {repo_id} to {local_dir}...")
snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model")
print("Download complete.") 