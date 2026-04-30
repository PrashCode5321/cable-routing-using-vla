"""
Patch a finetuned OpenVLA checkpoint with dataset normalization statistics.

This script reads the saved dataset statistics from the training run and registers them
in the model config so that the unnorm_key works during inference.
"""

import json
import sys
from pathlib import Path


def patch_checkpoint_with_stats(checkpoint_dir: Path, dataset_name: str, stats_file: str = "dataset_statistics.json"):
    """
    Patch a checkpoint with dataset statistics.
    
    Args:
        checkpoint_dir: Path to the finetuned checkpoint
        dataset_name: Name of the dataset (e.g., 'my_robot_dataset')
        stats_file: Name of the statistics file to load
    """
    checkpoint_dir = Path(checkpoint_dir)
    stats_path = checkpoint_dir / stats_file
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Dataset statistics file not found at {stats_path}")
    
    print(f"Loading dataset statistics from {stats_path}")
    with open(stats_path, 'r') as f:
        loaded_stats = json.load(f)
    
    # Extract the inner stats (file has structure: {dataset_name: {action: {...}, ...}})
    if dataset_name in loaded_stats:
        dataset_stats = loaded_stats[dataset_name]
        print(f"Extracted stats for '{dataset_name}' from file")
    else:
        dataset_stats = loaded_stats
        print(f"Using loaded stats directly")
    
    # Load config
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Add dataset statistics to config
    if "norm_stats" not in config_dict:
        config_dict["norm_stats"] = {}
    
    # Add the custom dataset statistics under the dataset name
    config_dict["norm_stats"][dataset_name] = dataset_stats
    
    print(f"Added '{dataset_name}' to norm_stats")
    print(f"Available norm_stats keys: {list(config_dict['norm_stats'].keys())}")
    
    # Save updated config
    print(f"Saving updated config to {config_path}")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("✓ Checkpoint patched successfully!")
    print(f"You can now use: vla.predict_action(..., unnorm_key='{dataset_name}', do_sample=False)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python patch_checkpoint_with_stats.py <checkpoint_dir> <dataset_name> [stats_file]")
        print("\nExample:")
        print("  python patch_checkpoint_with_stats.py ./runs/openvla-7b+my_robot_dataset+b1+... my_robot_dataset")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    dataset_name = sys.argv[2]
    stats_file = sys.argv[3] if len(sys.argv) > 3 else "dataset_statistics.json"
    
    patch_checkpoint_with_stats(checkpoint_dir, dataset_name, stats_file)
