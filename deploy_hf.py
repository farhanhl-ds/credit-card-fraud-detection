"""Deploy the Docker service to a Hugging Face Space (Docker SDK).

Prerequisites:
    - a Hugging Face account and a token with write access: export HF_TOKEN=hf_...
    - a trained model: python train.py

Usage:
    python deploy_hf.py --space <username>/credit-card-fraud
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

FILES = ["Dockerfile", "requirements-serve.txt", "fraud_pipeline.py", "predict.py", "models/model.joblib"]

SPACE_README = """---
title: Credit Card Fraud API
sdk: docker
app_port: 9696
pinned: false
---

Flask service that scores a card transaction with a fraud model.

- `GET /health`
- `POST /predict` with a JSON object of transaction features
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--space", required=True, help="<username>/<space-name>")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face token with write access")

    root = Path(__file__).parent
    missing = [f for f in FILES if not (root / f).exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing}. Run python train.py first.")

    stage = Path(tempfile.mkdtemp())
    for rel in FILES:
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)
    (stage / "README.md").write_text(SPACE_README)

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=args.space, repo_type="space")

    owner, name = args.space.split("/")
    print(f"Uploaded. Space page: https://huggingface.co/spaces/{args.space}")
    print(f"Once the build finishes the API is usually at: https://{owner}-{name}.hf.space".lower().replace("_", "-"))


if __name__ == "__main__":
    main()
