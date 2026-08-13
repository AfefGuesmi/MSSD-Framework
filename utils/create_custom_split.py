# -*- coding: utf-8 -*-
"""
Generate a custom 80/10/10 train/val/test split for MARIDA, as an
alternative to the official ~50/24/26 split -- STRATIFIED by class, so
that every class is guaranteed to appear in val AND test, not just train.

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

WHY THIS VERSION EXISTS:
A purely random scene-level split (the previous version of this script)
can, by chance, leave a rare class (e.g. Natural Organic Material, only
49 pixels in the entire 1,381-patch dataset) completely absent from a
small val or test split. This isn't just a cosmetic problem -- it can
crash evaluation code that assumes a fixed number of classes, and it
silently distorts macro-averaged metrics (a missing class gets dropped
from the average instead of counted as 0, quietly inflating scores).
This script fixes that at the source: before doing the usual ratio-based
scene assignment, it FIRST guarantees at least one scene containing each
class ends up in val, and at least one (different) scene containing each
class ends up in test -- by actually reading each patch's mask to know
which classes it contains, not just guessing from the ROI name.

Because scenes contain different numbers of patches, and now some
scenes are pinned in place to guarantee class coverage, the resulting
patch-level ratios will be close to 80/10/10 but not exact -- this is
expected and correct.

REQUIRES: GDAL (same dependency as dataloader.py), and to be run where
the actual MARIDA patches/ directory (with real mask .tif files) exists
-- this cannot run in a sandbox without the real dataset. The scene-key
parsing (ROI naming convention) has been verified against a real MARIDA
split-file listing; the mask-reading path has NOT been run against real
data by the assistant and should be checked against your actual
directory layout on first run.

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
import logging
import os
import random
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')


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


def load_scene_classes(marida_path, scenes, agg_to_water=True, num_classes=11):
    """
    For each scene, read every one of its patches' mask rasters and
    return the set of classes (0-indexed, matching GenDEBRIS's label
    space) present anywhere in that scene.

    This requires GDAL and the real MARIDA patches/ directory -- it
    mirrors the mask-loading logic in dataloader.py's GenDEBRIS.__init__
    exactly (same class-aggregation, same 1-indexed-to-0-indexed shift)
    so the class IDs here match what the model actually trains on.
    """
    from osgeo import gdal

    scene_classes = {}
    for scene, patches in scenes.items():
        classes_in_scene = set()
        for roi in patches:
            folder = '_'.join(['S2'] + roi.split('_')[:-1])
            name = '_'.join(['S2'] + roi.split('_'))
            mask_path = os.path.join(marida_path, 'patches', folder, f'{name}_cl.tif')

            ds = gdal.Open(mask_path)
            if ds is None:
                logging.warning('Could not open mask for %s at %s -- skipping this patch '
                                 'for class-coverage purposes.', roi, mask_path)
                continue
            mask = ds.ReadAsArray().astype(np.int64)
            ds = None

            if agg_to_water:
                mask[mask == 15] = 7
                mask[mask == 14] = 7
                mask[mask == 13] = 7
                mask[mask == 12] = 7
            mask = mask - 1  # shift 1..15 -> 0..14, matching GenDEBRIS

            present = set(np.unique(mask).tolist())
            present = {c for c in present if 0 <= c < num_classes}  # drop any out-of-range/void codes
            classes_in_scene.update(present)
        scene_classes[scene] = classes_in_scene
    return scene_classes


def make_stratified_scene_split(rois, scene_classes, num_classes=11,
                                 train_frac=0.8, val_frac=0.1, seed=0):
    """
    Scene-level split that GUARANTEES every class in range(num_classes)
    is represented by at least one scene in val, and at least one
    (different) scene in test, before filling the remaining scenes by
    the usual size-based greedy ratio matching (as in the plain,
    non-stratified version of this split).

    Returns:
        (train_rois, val_rois, test_rois, uncovered) -- uncovered is a
        dict {'val': set_of_missing_classes, 'test': set_of_missing_classes}
        listing any class that could not be guaranteed (only happens if
        that class doesn't exist in ANY scene, or all its scenes were
        already claimed by an earlier phase -- should be empty/rare for
        MARIDA, but check this in the printed output before trusting the
        split).
    """
    scenes = defaultdict(list)
    for roi in rois:
        scenes[scene_key(roi)].append(roi)

    scene_keys_all = list(scenes.keys())
    rng = random.Random(seed)
    rng.shuffle(scene_keys_all)

    assigned = {}  # scene_key -> 'train' / 'val' / 'test'
    uncovered = {'val': set(), 'test': set()}

    remaining = list(scene_keys_all)

    # ---- Phase 1: guarantee class coverage in test, then val ----
    # test first (usually the smaller/harder-to-cover set); prefer the
    # SMALLEST qualifying scene for each class, to waste as little of the
    # train budget as possible on forced assignments.
    for target_name in ('test', 'val'):
        covered = set()
        for c in range(num_classes):
            if c in covered:
                continue
            candidates = [s for s in remaining if c in scene_classes.get(s, set())]
            if not candidates:
                uncovered[target_name].add(c)
                continue
            candidates.sort(key=lambda s: len(scenes[s]))
            chosen = candidates[0]
            assigned[chosen] = target_name
            remaining.remove(chosen)
            covered.update(scene_classes[chosen])

    # ---- Phase 2: fill the rest by ratio, same greedy logic as before ----
    total_patches = len(rois)
    target_train = train_frac * total_patches
    target_val = val_frac * total_patches

    n_train = sum(len(scenes[s]) for s, v in assigned.items() if v == 'train')
    n_val = sum(len(scenes[s]) for s, v in assigned.items() if v == 'val')

    for key in remaining:
        patches = scenes[key]
        train_deficit = target_train - n_train
        val_deficit = target_val - n_val
        if train_deficit >= val_deficit and train_deficit > 0:
            assigned[key] = 'train'
            n_train += len(patches)
        elif val_deficit > 0:
            assigned[key] = 'val'
            n_val += len(patches)
        else:
            assigned[key] = 'test'

    train_rois, val_rois, test_rois = [], [], []
    for key, split in assigned.items():
        target = train_rois if split == 'train' else (val_rois if split == 'val' else test_rois)
        target.extend(scenes[key])

    return train_rois, val_rois, test_rois, uncovered


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
    parser.add_argument('--num_classes', default=11, type=int,
                         help='Number of classes to guarantee coverage for (matches '
                              '--output_channels in the training script).')
    parser.add_argument('--agg_to_water', default=True, type=bool,
                         help='Must match --agg_to_water used elsewhere, so class IDs line up.')
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

    scenes = defaultdict(list)
    for roi in rois:
        scenes[scene_key(roi)].append(roi)
    print(f'Grouped into {len(scenes)} distinct scenes (date+tile).')

    print('\nReading mask rasters to determine class coverage per scene '
          '(this requires GDAL and the real patches/ directory; may take a while)...')
    scene_classes = load_scene_classes(
        args.marida_path, scenes, agg_to_water=args.agg_to_water, num_classes=args.num_classes
    )

    train_rois, val_rois, test_rois, uncovered = make_stratified_scene_split(
        rois, scene_classes, num_classes=args.num_classes,
        train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )

    total = len(rois)
    print(f'\nResulting split (patch-level):')
    print(f'  train: {len(train_rois):4d} patches ({100*len(train_rois)/total:.1f}%)')
    print(f'  val:   {len(val_rois):4d} patches ({100*len(val_rois)/total:.1f}%)')
    print(f'  test:  {len(test_rois):4d} patches ({100*len(test_rois)/total:.1f}%)')

    # Sanity check 1: no scene appears in more than one split (no leakage).
    train_scenes = set(scene_key(r) for r in train_rois)
    val_scenes = set(scene_key(r) for r in val_rois)
    test_scenes = set(scene_key(r) for r in test_rois)
    overlap = (train_scenes & val_scenes) | (train_scenes & test_scenes) | (val_scenes & test_scenes)
    if overlap:
        raise RuntimeError(f'Data leakage detected: {len(overlap)} scene(s) appear in multiple splits: {overlap}')
    print('No scene-level leakage between splits: confirmed.')

    # Sanity check 2: report any class that could NOT be covered.
    if uncovered['val'] or uncovered['test']:
        print('\nWARNING -- could not guarantee coverage for:')
        if uncovered['val']:
            print(f"  val:  class IDs {sorted(uncovered['val'])} appear in NO available scene "
                  f"(or all their scenes were already used by test).")
        if uncovered['test']:
            print(f"  test: class IDs {sorted(uncovered['test'])} appear in NO available scene.")
        print('  This means that class genuinely has too little data to guarantee split '
              'coverage -- check whether it exists at all in your dataset.')
    else:
        print('Every class (0-%d) is present in both val and test: confirmed.' % (args.num_classes - 1))

    out_dir = os.path.join(args.marida_path, args.output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    write_split_file(os.path.join(out_dir, 'train_X.txt'), train_rois)
    write_split_file(os.path.join(out_dir, 'val_X.txt'), val_rois)
    write_split_file(os.path.join(out_dir, 'test_X.txt'), test_rois)
    print(f'\nWrote new split files to: {out_dir}')
    print('Point GenDEBRIS/training scripts at this folder via --splits_dir to use it.')


if __name__ == '__main__':
    main()