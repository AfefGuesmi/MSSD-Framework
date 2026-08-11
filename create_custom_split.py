# -*- coding: utf-8 -*-
"""
Generate a custom 80/10/10 train/val/test split for MARIDA, as an
alternative to the official ~50/24/26 split.

IMPORTANT -- read before using:
MARIDA's official split explicitly keeps every patch from the same S2
scene/acquisition date together in one split ("the data of each
scene/unique date were retained in the same set" -- Kikaki et al., 2022),
to avoid data leakage: patches from the same scene are visually and
spectrally very similar, so letting some end up in train and others in
test would let the model "cheat" by recognising the scene rather than
learning to detect debris. This script preserves that property -- it
splits at the SCENE level, then assigns every patch belonging to a
chosen scene to that scene's split -- it does not split at the raw patch
level (which would leak data and inflate apparent performance).

Because scenes contain different numbers of patches, the resulting
patch-level ratios will be close to 80/10/10 but not exact -- this is
expected and correct; forcing an exact patch-level ratio would require
either splitting scenes across sets (leakage) or a much larger dataset.

USAGE:
    python create_custom_split.py --marida_path /path/to/MARIDA --seed 0

This reads the existing official split files (train_X.txt, val_X.txt,
test_X.txt in <marida_path>/splits/) purely to get the full list of
patches -- it does NOT reuse their assignment -- and writes new files to
<marida_path>/splits_80_10_10/{train,val,test}_X.txt in the same format
GenDEBRIS already expects, so you can point --splits_dir at the new
folder without touching the original official split.
"""

import argparse
import os
import random
from collections import defaultdict


def scene_key(roi):
    """Everything except the last underscore-separated token (patch index)."""
    return roi.rsplit('_', 1)[0]


def load_all_rois(splits_dir):
    """Read every ROI from the three official split files (order doesn't matter,
    we're only using them as a source of the full patch list)."""
    rois = []
    for split_name in ('train_X.txt', 'val_X.txt', 'test_X.txt'):
        path = os.path.join(splits_dir, split_name)
        with open(path, 'r') as f:
            rois.extend(line.strip() for line in f if line.strip())
    return rois


def make_scene_level_split(rois, train_frac=0.8, val_frac=0.1, seed=0):
    """
    Group ROIs by scene, shuffle scenes, then greedily assign whole scenes
    to train/val/test until each bucket's PATCH count is close to its
    target fraction of the total. Returns three lists of ROIs.
    """
    scenes = defaultdict(list)
    for roi in rois:
        scenes[scene_key(roi)].append(roi)

    scene_keys = list(scenes.keys())
    rng = random.Random(seed)
    rng.shuffle(scene_keys)

    total_patches = len(rois)
    target_train = train_frac * total_patches
    target_val = val_frac * total_patches
    # test gets whatever remains

    train_rois, val_rois, test_rois = [], [], []
    n_train, n_val = 0, 0

    for key in scene_keys:
        patches = scenes[key]
        # Assign this whole scene to whichever bucket is furthest below its
        # target, so all three converge toward the requested ratios together
        # rather than greedily filling train first and starving val/test.
        train_deficit = target_train - n_train
        val_deficit = target_val - n_val
        if train_deficit >= val_deficit and train_deficit > 0:
            train_rois.extend(patches)
            n_train += len(patches)
        elif val_deficit > 0:
            val_rois.extend(patches)
            n_val += len(patches)
        else:
            test_rois.extend(patches)

    return train_rois, val_rois, test_rois


def write_split_file(path, rois):
    with open(path, 'w') as f:
        for roi in rois:
            f.write(roi + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--marida_path', required=True,
                         help='Root MARIDA directory (containing splits/ and patches/)')
    parser.add_argument('--train_frac', default=0.8, type=float)
    parser.add_argument('--val_frac', default=0.1, type=float)
    parser.add_argument('--seed', default=0, type=int,
                         help='Shuffle seed -- change this to get a different random split '
                              'if you want to sanity-check result stability across splits.')
    parser.add_argument('--output_subdir', default='splits_80_10_10',
                         help='Written under <marida_path>/, alongside the original splits/ '
                              'folder -- never overwrites the official split.')
    args = parser.parse_args()

    splits_dir = os.path.join(args.marida_path, 'splits')
    rois = load_all_rois(splits_dir)
    print(f'Loaded {len(rois)} total patches from the official split files.')

    n_scenes = len(set(scene_key(r) for r in rois))
    print(f'Grouped into {n_scenes} distinct scenes (date+tile).')

    train_rois, val_rois, test_rois = make_scene_level_split(
        rois, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )

    total = len(rois)
    print(f'\nResulting split (patch-level):')
    print(f'  train: {len(train_rois):4d} patches ({100*len(train_rois)/total:.1f}%)')
    print(f'  val:   {len(val_rois):4d} patches ({100*len(val_rois)/total:.1f}%)')
    print(f'  test:  {len(test_rois):4d} patches ({100*len(test_rois)/total:.1f}%)')

    # Sanity check: no scene appears in more than one split (no leakage).
    train_scenes = set(scene_key(r) for r in train_rois)
    val_scenes = set(scene_key(r) for r in val_rois)
    test_scenes = set(scene_key(r) for r in test_rois)
    overlap = (train_scenes & val_scenes) | (train_scenes & test_scenes) | (val_scenes & test_scenes)
    if overlap:
        raise RuntimeError(f'Data leakage detected: {len(overlap)} scene(s) appear in multiple splits: {overlap}')
    print('\nNo scene-level leakage between splits: confirmed.')

    out_dir = os.path.join(args.marida_path, args.output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    write_split_file(os.path.join(out_dir, 'train_X.txt'), train_rois)
    write_split_file(os.path.join(out_dir, 'val_X.txt'), val_rois)
    write_split_file(os.path.join(out_dir, 'test_X.txt'), test_rois)
    print(f'\nWrote new split files to: {out_dir}')
    print('Point GenDEBRIS/training scripts at this folder via --splits_dir to use it.')


if __name__ == '__main__':
    main()
