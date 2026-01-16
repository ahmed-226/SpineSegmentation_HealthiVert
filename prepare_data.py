#!/usr/bin/env python
"""
Data preparation script for VerSe2019 Vertebrae Segmentation Pipeline

Scans the VerSe19 data directory structure and generates:
1. train.txt - List of training subject IDs
2. val.txt - List of validation subject IDs

VerSe19 Expected Structure:
    verse19/
    ├── dataset-verse19training/
    │   ├── rawdata/
    │   │   └── sub-verseXXX/
    │   │       └── sub-verseXXX_ct.nii.gz
    │   └── derivatives/
    │       └── sub-verseXXX/
    │           ├── sub-verseXXX_seg-vert_msk.nii.gz
    │           └── sub-verseXXX_seg-subreg_ctd.json
    ├── dataset-verse19validation/
    │   └── (same structure)
    └── dataset-verse19test/
        └── (same structure)

Usage:
    python prepare_data.py --data_dir /path/to/verse19
"""

import argparse
import os
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple


def find_verse_subjects(data_dir: Path) -> Dict[str, Dict]:
    """
    Find all subjects in VerSe19 directory structure.
    
    Returns:
        Dict mapping subject_id to paths dict:
        {
            'sub-verse004': {
                'image': Path to CT image,
                'mask': Path to segmentation mask,
                'landmarks': Path to landmarks JSON,
                'split': 'training'/'validation'/'test'
            }
        }
    """
    subjects = {}
    
    # Search for dataset-* directories
    dataset_dirs = list(data_dir.glob('dataset-verse19*'))
    
    if not dataset_dirs:
        # Maybe we're already inside a dataset directory
        if (data_dir / 'rawdata').exists():
            dataset_dirs = [data_dir]
    
    for dataset_dir in dataset_dirs:
        # Determine split from directory name
        dir_name = dataset_dir.name.lower()
        if 'training' in dir_name:
            split = 'training'
        elif 'validation' in dir_name:
            split = 'validation'
        elif 'test' in dir_name:
            split = 'test'
        else:
            split = 'unknown'
        
        rawdata_dir = dataset_dir / 'rawdata'
        derivatives_dir = dataset_dir / 'derivatives'
        
        if not rawdata_dir.exists():
            continue
        
        # Find all subject directories in rawdata
        for subject_dir in rawdata_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            
            subject_id = subject_dir.name  # e.g., 'sub-verse004'
            
            # Find CT image - filter to only include actual files (not directories)
            ct_files = [f for f in subject_dir.glob('*_ct.nii.gz') if f.is_file()]
            ct_files += [f for f in subject_dir.glob('*_ct.nii') if f.is_file()]
            if not ct_files:
                ct_files = [f for f in subject_dir.glob('*.nii.gz') if f.is_file()]
                ct_files += [f for f in subject_dir.glob('*.nii') if f.is_file()]
            
            if not ct_files:
                print(f"Warning: No CT image found for {subject_id}")
                continue
            
            # Prefer .nii.gz if duplicates exist, otherwise take first
            # Sort to prioritize .nii.gz (shorter path length after extension swap) and ensure deterministic selection
            ct_files.sort(key=lambda p: (0 if str(p).endswith('.nii.gz') else 1, str(p)))
            image_path = ct_files[0]
            
            # Find corresponding derivatives
            deriv_subject_dir = derivatives_dir / subject_id
            mask_path = None
            landmarks_path = None
            
            if deriv_subject_dir.exists():
                # Find segmentation mask - filter to only include actual files
                mask_files = [f for f in deriv_subject_dir.glob('*_seg-vert_msk.nii.gz') if f.is_file()]
                mask_files += [f for f in deriv_subject_dir.glob('*_seg-vert_msk.nii') if f.is_file()]
                if mask_files:
                    mask_files.sort(key=lambda p: (0 if str(p).endswith('.nii.gz') else 1, str(p)))
                    mask_path = mask_files[0]
                
                # Find landmarks JSON - check both naming conventions
                landmarks_files = list(deriv_subject_dir.glob('*_seg-subreg_ctd.json'))
                if not landmarks_files:
                    landmarks_files = list(deriv_subject_dir.glob('*_seg-vb_ctd.json'))
                if landmarks_files:
                    landmarks_path = landmarks_files[0]
            
            subjects[subject_id] = {
                'image': image_path,
                'mask': mask_path,
                'landmarks': landmarks_path,
                'split': split
            }
    
    return subjects


def main():
    parser = argparse.ArgumentParser(description='Prepare data split for VerSe pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to VerSe19 dataset directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (only used if no predefined splits)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_predefined_splits', action='store_true', default=True,
                        help='Use predefined training/validation splits from VerSe')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist!")
        return
    
    print(f"Scanning for subjects in {data_dir}...")
    subjects = find_verse_subjects(data_dir)
    
    if not subjects:
        print("No subjects found! Please check your data directory structure.")
        print("\nExpected VerSe19 structure:")
        print("  verse19/")
        print("  ├── dataset-verse19training/")
        print("  │   ├── rawdata/")
        print("  │   │   └── sub-verseXXX/")
        print("  │   │       └── sub-verseXXX_ct.nii.gz")
        print("  │   └── derivatives/")
        print("  │       └── sub-verseXXX/")
        print("  │           └── sub-verseXXX_seg-subreg_ctd.json")
        print("  ├── dataset-verse19validation/")
        print("  └── dataset-verse19test/")
        return
    
    print(f"Found {len(subjects)} subjects.")
    
    # Count by split
    split_counts = {}
    for subj_id, info in subjects.items():
        split = info['split']
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print(f"Split distribution: {split_counts}")
    
    # Check for landmarks
    with_landmarks = sum(1 for s in subjects.values() if s['landmarks'] is not None)
    with_masks = sum(1 for s in subjects.values() if s['mask'] is not None)
    print(f"Subjects with landmarks: {with_landmarks}")
    print(f"Subjects with segmentation masks: {with_masks}")
    
    # Use predefined splits if available
    if args.use_predefined_splits and 'training' in split_counts:
        train_ids = [s for s, info in subjects.items() if info['split'] == 'training']
        val_ids = [s for s, info in subjects.items() if info['split'] == 'validation']
        test_ids = [s for s, info in subjects.items() if info['split'] == 'test']
        
        # If no validation split, take some from training
        if not val_ids and train_ids:
            random.shuffle(train_ids)
            split_idx = int(len(train_ids) * (1 - args.val_split))
            val_ids = train_ids[split_idx:]
            train_ids = train_ids[:split_idx]
    else:
        # Random split
        all_ids = list(subjects.keys())
        random.shuffle(all_ids)
        split_idx = int(len(all_ids) * (1 - args.val_split))
        train_ids = all_ids[:split_idx]
        val_ids = all_ids[split_idx:]
        test_ids = []
    
    print(f"\nFinal splits:")
    print(f"  Training: {len(train_ids)}")
    print(f"  Validation: {len(val_ids)}")
    if test_ids:
        print(f"  Test: {len(test_ids)}")
    
    # Write split files
    train_file = data_dir / 'train.txt'
    val_file = data_dir / 'val.txt'
    test_file = data_dir / 'test.txt'
    
    with open(train_file, 'w') as f:
        f.write('\n'.join(sorted(train_ids)))
    
    with open(val_file, 'w') as f:
        f.write('\n'.join(sorted(val_ids)))
    
    if test_ids:
        with open(test_file, 'w') as f:
            f.write('\n'.join(sorted(test_ids)))
    
    # Write subjects info JSON for reference
    info_file = data_dir / 'subjects_info.json'
    with open(info_file, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        subjects_json = {}
        for subj_id, info in subjects.items():
            subjects_json[subj_id] = {
                'image': str(info['image']) if info['image'] else None,
                'mask': str(info['mask']) if info['mask'] else None,
                'landmarks': str(info['landmarks']) if info['landmarks'] else None,
                'split': info['split']
            }
        json.dump(subjects_json, f, indent=2)
    
    print(f"\nSuccessfully generated:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    if test_ids:
        print(f"  - {test_file}")
    print(f"  - {info_file}")
    print("\nYou can now run training!")


if __name__ == '__main__':
    main()
