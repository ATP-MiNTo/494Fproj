from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    repo_id = os.environ.get("HF_SPACE_REPO")
    token = os.environ.get("HF_TOKEN")
    if not repo_id or not token:
        raise SystemExit("HF_SPACE_REPO and HF_TOKEN must be set")

    root = Path(__file__).resolve().parents[1]
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(root),
        commit_message="Deploy FastAPI image classifier",
        ignore_patterns=[
            ".git/*",
            ".pytest_cache/*",
            "__pycache__/*",
            "tests/*",
            "docs/benchmark_results.json",
        ],
    )


if __name__ == "__main__":
    main()
