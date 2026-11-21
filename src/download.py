"""Downloads the trained model weights from the Hugging Face Hub into ./models
(weights are not stored in the Git repository, as required).

Run:  python -m src.download                 (uses HF_MODEL_REPO from src/config.py)
      python -m src.download --repo user/name
Publish your own weights once (after `hf auth login`):
      hf upload <user>/realtime-chat-moderation-models models . --repo-type model
"""
import argparse

from huggingface_hub import snapshot_download

from .config import HF_MODEL_REPO, MODELS_DIR


def download(repo=HF_MODEL_REPO):
    if not repo or repo.startswith("<"):
        raise SystemExit("Set HF_MODEL_REPO in src/config.py (or pass --repo), or train locally: python -m src.train_all")
    snapshot_download(repo_id=repo, repo_type="model", local_dir=MODELS_DIR)
    print(f"Weights downloaded to {MODELS_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=HF_MODEL_REPO)
    download(ap.parse_args().repo)
