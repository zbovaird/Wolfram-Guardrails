"""Shared runtime helpers for colab_parser_qlora_finetune.ipynb."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import time
from pathlib import Path

REPO_DIR_DEFAULT = Path("/content/wolfram-guardrails")
MERGED_DIR_DEFAULT = REPO_DIR_DEFAULT / "results/finetune/colab_qwen25_3b/merged_hf"
COMPARISON_SCRIPTS = (
    "examples/compare_colab_qlora_parser.py",
    "examples/compare_english_vs_finetuned_wolfram.py",
)


def repair_colab_deps() -> None:
    """Fix protobuf runtime_version ImportError after torchao or other pip installs."""
    try:
        from google.protobuf import runtime_version  # noqa: F401

        print("protobuf ok")
        return
    except ImportError:
        print("Repairing protobuf...")

    subprocess.run(
        ["pip", "install", "-q", "--force-reinstall", "protobuf>=5.29.1,<6.0.0"],
        check=True,
    )
    import google.protobuf

    importlib.reload(google.protobuf)
    from google.protobuf import runtime_version  # noqa: F401

    print("protobuf repaired")


def repair_merged_hf_tokenizer(merged_dir: Path) -> None:
    """Merged checkpoints may ship extra_special_tokens as a list; fast tokenizer needs a dict."""
    tok_cfg = Path(merged_dir) / "tokenizer_config.json"
    if not tok_cfg.is_file():
        return
    cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
    if isinstance(cfg.get("extra_special_tokens"), list):
        del cfg["extra_special_tokens"]
        tok_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print("Sanitized merged_hf tokenizer_config.json")


def verify_repo_layout(*, repo_dir: Path, dataset_paths: list[Path]) -> None:
    missing = [str(p.relative_to(repo_dir)) for p in dataset_paths if not p.exists()]
    missing += [rel for rel in COMPARISON_SCRIPTS if not (repo_dir / rel).exists()]
    if missing:
        raise FileNotFoundError("Missing after sync: " + ", ".join(missing))
    for path in dataset_paths:
        lines = sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
        print(f"{path.relative_to(repo_dir)}: {lines} rows")
    for rel in COMPARISON_SCRIPTS:
        print("script ok:", rel)


def ensure_repo_cwd(
    *,
    repo_dir: Path,
    repo_branch: str = "main",
    pull_latest: bool = False,
    dataset_paths: list[Path] | None = None,
) -> Path:
    repo_dir = Path(repo_dir)
    if pull_latest and (repo_dir / ".git").exists():
        print("Pulling latest scripts from", repo_branch)
        subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", repo_branch],
            check=True,
        )
    os.chdir(repo_dir)
    if dataset_paths:
        verify_repo_layout(repo_dir=repo_dir, dataset_paths=dataset_paths)
    print("Working directory:", os.getcwd())
    return repo_dir


def clone_or_pull_repo(
    *,
    repo_dir: Path,
    repo_url: str,
    repo_branch: str = "main",
    dataset_paths: list[Path] | None = None,
) -> Path:
    import shutil

    repo_dir = Path(repo_dir)
    if repo_dir.exists() and (repo_dir / ".git").exists():
        print("Repo already cloned — pulling latest (keeps results/finetune/ checkpoints)")
        subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", repo_branch],
            check=True,
        )
    else:
        if repo_dir.exists():
            print("Removing non-git folder at", repo_dir)
            shutil.rmtree(repo_dir)
        clone_cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            repo_branch,
            repo_url,
            str(repo_dir),
        ]
        print("Running:", " ".join(clone_cmd))
        subprocess.run(clone_cmd, check=True)
    return ensure_repo_cwd(
        repo_dir=repo_dir,
        repo_branch=repo_branch,
        pull_latest=False,
        dataset_paths=dataset_paths,
    )


def ensure_ollama(*, base_url: str = "http://127.0.0.1:11434", model: str | None = None) -> None:
    import httpx

    def ollama_is_up() -> bool:
        try:
            response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    if not ollama_is_up():
        print("Starting ollama serve...")
        log = open("/content/ollama.log", "w")
        subprocess.Popen(["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT)
        for _ in range(30):
            time.sleep(1)
            if ollama_is_up():
                break
        else:
            raise RuntimeError("Ollama did not start. Check /content/ollama.log")

    if model:
        print(f"Pulling Ollama model {model}...")
        subprocess.run(["ollama", "pull", model], check=True)
        subprocess.run(["ollama", "list"], check=True)
    else:
        print("Ollama is up.")


def prepare_english_vs_wolfram(
    *,
    repo_dir: Path | None = None,
    repo_branch: str | None = None,
    merged_dir: Path | None = None,
    english_model: str = "llama3",
    pull_latest: bool = True,
) -> Path:
    """Run before section 14 comparison cells (safe after runtime restart)."""
    repo_dir = Path(repo_dir or os.environ.get("WOLFRAM_GUARDRAILS_DIR", REPO_DIR_DEFAULT))
    repo_branch = repo_branch or os.environ.get("WOLFRAM_GUARDRAILS_BRANCH", "main")
    merged_dir = Path(merged_dir or (repo_dir / "results/finetune/colab_qwen25_3b/merged_hf"))

    if not repo_dir.exists():
        raise FileNotFoundError(
            f"Repo not found at {repo_dir}. Run section 4 (clone) first.\n"
            "Colab path is /content/wolfram-guardrails (lowercase)."
        )

    ensure_repo_cwd(repo_dir=repo_dir, repo_branch=repo_branch, pull_latest=pull_latest)
    repair_colab_deps()

    if not merged_dir.exists():
        raise FileNotFoundError(
            f"Merged model not found at {merged_dir}. Run sections 7-9 (train + merge) first."
        )
    repair_merged_hf_tokenizer(merged_dir)
    ensure_ollama(model=english_model)
    print("Section 14 ready: English vs fine-tuned Wolfram comparison.")
    return repo_dir
