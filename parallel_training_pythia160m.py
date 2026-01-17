#!/usr/bin/env python3
"""
Parallel training script for Pythia-160M SAE experiments.

This script trains 7 SAE architectures across 3 widths (4k, 16k, 64k) and 6 sparsity levels,
using 4x A100 GPUs in parallel, with automatic HuggingFace upload per width.

Architectures:
- Target L0-based (4): top_k, batch_top_k, jump_relu, matryoshka_batch_top_k
- Penalty-based (3): standard, p_anneal, gated

Total SAEs: 126 (72 target L0-based + 54 penalty-based)

Usage:
    python parallel_training_pythia160m.py
"""

import subprocess
import time
import os
import sys
from pathlib import Path

import huggingface_hub

# Load settings from .env
from settings import settings

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "EleutherAI/pythia-160m-deduped"
LAYER = 8
WIDTHS = [4096, 16384, 65536]  # 4k, 16k, 64k
HF_REPO_TEMPLATE = "raymondzmc/saebench_pythia-160m-deduped_width-{width}"
WANDB_PROJECT = "pythia-160m"

# GPU assignments based on training speed
# Relative speed: standard/p_anneal > top_k > batch_top_k/matryoshka > jump_relu > gated
GPU_ASSIGNMENTS = {
    "cuda:0": ["jump_relu"],                              # Slowest (6 SAEs per width)
    "cuda:1": ["gated", "standard"],                      # Slow + Fast (12 SAEs per width)
    "cuda:2": ["top_k", "p_anneal"],                      # Medium + Fast (12 SAEs per width)
    "cuda:3": ["batch_top_k", "matryoshka_batch_top_k"],  # Medium (12 SAEs per width)
}

SAVE_DIR_BASE = "trained_saes_pythia160m"

# ============================================================================
# Setup Functions
# ============================================================================

def setup_credentials():
    """Setup HuggingFace and Wandb credentials from settings."""
    print("=" * 60)
    print("Setting up credentials...")
    print("=" * 60)
    
    # Verify settings are loaded
    if not settings.has_hf_config():
        raise ValueError("HF_ACCESS_TOKEN not found in .env file")
    if not settings.has_wandb_config():
        raise ValueError("WANDB_API_KEY and WANDB_ENTITY not found in .env file")
    
    # Login to HuggingFace
    huggingface_hub.login(token=settings.hf_access_token)
    print(f"Logged in to HuggingFace")
    
    # Set wandb environment variable
    os.environ["WANDB_API_KEY"] = settings.wandb_api_key
    print(f"Wandb entity: {settings.wandb_entity}")
    print(f"Wandb project: {WANDB_PROJECT}")
    
    return settings.wandb_entity


def create_hf_repos_if_needed():
    """Create HuggingFace repositories if they don't exist."""
    print("\n" + "=" * 60)
    print("Checking HuggingFace repositories...")
    print("=" * 60)
    
    for width in WIDTHS:
        repo_id = HF_REPO_TEMPLATE.format(width=width)
        try:
            if huggingface_hub.repo_exists(repo_id=repo_id, repo_type="model"):
                print(f"  [EXISTS] {repo_id}")
            else:
                print(f"  [CREATING] {repo_id}")
                huggingface_hub.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    exist_ok=True,
                    private=False,
                )
                print(f"  [CREATED] {repo_id}")
        except Exception as e:
            print(f"  [ERROR] {repo_id}: {e}")
            raise


def get_all_architectures():
    """Get flat list of all architectures being trained."""
    all_archs = []
    for archs in GPU_ASSIGNMENTS.values():
        all_archs.extend(archs)
    return all_archs


# ============================================================================
# Training Functions
# ============================================================================

def launch_training_job(
    device: str,
    architectures: list[str],
    width: int,
    save_dir: str,
    wandb_entity: str,
    log_dir: Path,
) -> subprocess.Popen:
    """Launch a single training job on a specific GPU."""
    
    arch_str = " ".join(architectures)
    log_file = log_dir / f"{device.replace(':', '_')}_{arch_str.replace(' ', '_')}_w{width}.log"
    
    cmd = [
        "python", "demo.py",
        "--save_dir", save_dir,
        "--model_name", MODEL_NAME,
        "--layers", str(LAYER),
        "--architectures", *architectures,
        "--device", device,
        "--dictionary_width", str(width),
        "--wandb_entity", wandb_entity,
        "--wandb_project", WANDB_PROJECT,
        "--use_wandb",
    ]
    
    print(f"  Launching on {device}: {arch_str}")
    print(f"    Command: {' '.join(cmd)}")
    print(f"    Log: {log_file}")
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    
    return process


def run_evaluation(save_dir: str):
    """Run evaluation on all trained SAEs in the save directory."""
    print(f"\n  Running evaluation on {save_dir}...")
    
    # Import here to avoid circular imports and ensure CUDA is ready
    from dictionary_learning.dictionary_learning.utils import get_nested_folders
    import demo_config
    from demo import eval_saes
    
    ae_paths = get_nested_folders(save_dir)
    if not ae_paths:
        print(f"  WARNING: No SAEs found in {save_dir}")
        return
    
    print(f"  Found {len(ae_paths)} SAEs to evaluate")
    
    # Run evaluation on first available GPU
    eval_saes(
        MODEL_NAME,
        ae_paths,
        demo_config.eval_num_inputs,
        "cuda:0",
        overwrite_prev_results=True,
    )
    print(f"  Evaluation complete")


def upload_to_huggingface(save_dir: str, width: int):
    """Upload trained SAEs to HuggingFace."""
    repo_id = HF_REPO_TEMPLATE.format(width=width)
    print(f"\n  Uploading to HuggingFace: {repo_id}")
    
    api = huggingface_hub.HfApi()
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=f"resid_post_layer_{LAYER}",
    )
    print(f"  Upload complete: {repo_id}")


def train_width(width: int, wandb_entity: str, log_dir: Path):
    """Train all architectures for a single width."""
    print("\n" + "=" * 60)
    print(f"Training width: {width}")
    print("=" * 60)
    
    # Create save directory for this width
    save_dir = f"{SAVE_DIR_BASE}/width_{width}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Launch all GPU jobs in parallel
    processes = []
    for device, architectures in GPU_ASSIGNMENTS.items():
        process = launch_training_job(
            device=device,
            architectures=architectures,
            width=width,
            save_dir=save_dir,
            wandb_entity=wandb_entity,
            log_dir=log_dir,
        )
        processes.append((device, process))
        time.sleep(5)  # Stagger launches to avoid race conditions
    
    # Wait for all jobs to complete
    print(f"\n  Waiting for {len(processes)} jobs to complete...")
    for device, process in processes:
        return_code = process.wait()
        if return_code != 0:
            print(f"  WARNING: Job on {device} exited with code {return_code}")
        else:
            print(f"  Job on {device} completed successfully")
    
    # Find the actual save directory (demo.py adds model name and architectures to path)
    # The structure will be: save_dir/resid_post_layer_8/trainer_*/
    actual_save_dir = save_dir
    
    # Run evaluation
    run_evaluation(actual_save_dir)
    
    # Upload to HuggingFace
    upload_to_huggingface(actual_save_dir, width)
    
    return actual_save_dir


# ============================================================================
# Main
# ============================================================================

def print_config():
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("Pythia-160M SAE Training Configuration")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"Widths: {WIDTHS}")
    print(f"Wandb Project: {WANDB_PROJECT}")
    print(f"HF Repo Template: {HF_REPO_TEMPLATE}")
    print(f"\nGPU Assignments:")
    total_saes = 0
    for device, archs in GPU_ASSIGNMENTS.items():
        # Count SAEs: target L0 based have 6 L0s, penalty based have 6 penalties
        target_l0_archs = ["top_k", "batch_top_k", "jump_relu", "matryoshka_batch_top_k"]
        n_saes = sum(6 for a in archs)  # 6 sparsity levels per arch
        total_saes += n_saes
        print(f"  {device}: {', '.join(archs)} ({n_saes} SAEs per width)")
    print(f"\nTotal SAEs per width: {total_saes}")
    print(f"Total SAEs overall: {total_saes * len(WIDTHS)}")


def main():
    """Main entry point."""
    start_time = time.time()
    
    # Print configuration
    print_config()
    
    # Setup credentials
    wandb_entity = setup_credentials()
    
    # Create HF repos
    create_hf_repos_if_needed()
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Train each width sequentially (upload after each)
    for width in WIDTHS:
        width_start = time.time()
        train_width(width, wandb_entity, log_dir)
        width_time = time.time() - width_start
        print(f"\n  Width {width} completed in {width_time/3600:.2f} hours")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"All training complete!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print("=" * 60)


if __name__ == "__main__":
    main()
