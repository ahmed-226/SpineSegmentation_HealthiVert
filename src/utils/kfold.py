"""
K-Fold Cross-Validation utilities for VerSe2019 Pipeline
"""
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import random


def create_kfold_splits(
    id_list: List[str],
    n_folds: int = 5,
    seed: int = 42,
    shuffle: bool = True
) -> List[Tuple[List[str], List[str]]]:
    """
    Create k-fold cross-validation splits.
    
    Args:
        id_list: List of sample IDs
        n_folds: Number of folds (default: 5)
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle before splitting
        
    Returns:
        List of (train_ids, val_ids) tuples for each fold
    """
    ids = list(id_list)
    
    if shuffle:
        random.seed(seed)
        random.shuffle(ids)
    
    # Calculate fold sizes
    n_samples = len(ids)
    fold_sizes = [n_samples // n_folds] * n_folds
    for i in range(n_samples % n_folds):
        fold_sizes[i] += 1
    
    # Create folds
    folds = []
    current = 0
    for fold_size in fold_sizes:
        val_ids = ids[current:current + fold_size]
        train_ids = ids[:current] + ids[current + fold_size:]
        folds.append((train_ids, val_ids))
        current += fold_size
    
    return folds


def save_fold_splits(
    folds: List[Tuple[List[str], List[str]]],
    output_dir: str
):
    """
    Save fold splits to text files.
    
    Args:
        folds: List of (train_ids, val_ids) tuples
        output_dir: Directory to save split files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        # Save train IDs
        train_file = output_path / f'fold_{fold_idx}_train.txt'
        with open(train_file, 'w') as f:
            f.write('\n'.join(train_ids))
        
        # Save val IDs
        val_file = output_path / f'fold_{fold_idx}_val.txt'
        with open(val_file, 'w') as f:
            f.write('\n'.join(val_ids))
    
    print(f"Saved {len(folds)} fold splits to {output_dir}")


def load_fold_split(fold_dir: str, fold_idx: int) -> Tuple[List[str], List[str]]:
    """
    Load a specific fold split from files.
    
    Args:
        fold_dir: Directory containing fold split files
        fold_idx: Fold index to load
        
    Returns:
        Tuple of (train_ids, val_ids)
    """
    fold_path = Path(fold_dir)
    
    train_file = fold_path / f'fold_{fold_idx}_train.txt'
    val_file = fold_path / f'fold_{fold_idx}_val.txt'
    
    with open(train_file, 'r') as f:
        train_ids = [line.strip() for line in f if line.strip()]
    
    with open(val_file, 'r') as f:
        val_ids = [line.strip() for line in f if line.strip()]
    
    return train_ids, val_ids


def get_fold_summary(folds: List[Tuple[List[str], List[str]]]) -> Dict:
    """
    Get summary statistics for fold splits.
    
    Args:
        folds: List of (train_ids, val_ids) tuples
        
    Returns:
        Dictionary with summary statistics
    """
    n_folds = len(folds)
    train_sizes = [len(train) for train, _ in folds]
    val_sizes = [len(val) for _, val in folds]
    
    return {
        'n_folds': n_folds,
        'train_sizes': train_sizes,
        'val_sizes': val_sizes,
        'avg_train_size': np.mean(train_sizes),
        'avg_val_size': np.mean(val_sizes),
        'total_samples': train_sizes[0] + val_sizes[0]  # Same for all folds
    }
