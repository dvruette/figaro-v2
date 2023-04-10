import os
import glob
import re

import tqdm
import numpy as np
import pretty_midi
from typing import List, Tuple

from figaro.chord_recognition.algorithms.bar_wise_template import barwise_chords
from figaro.chord_recognition.algorithms.extract import extract_chords
# from figaro.chord_recognition.algorithms.chordino import chordino_chords
from figaro.chord_recognition.v1 import extract_chords_legacy
from figaro.chord_recognition.utils import normalize_chord, root_to_chroma


def chord_accuracy(y_pred: str, y_true: str) -> float:
    y_true, y_pred = normalize_chord(y_true), normalize_chord(y_pred)
    if y_true == "N" or y_pred == "N":
        return 1 if y_true == y_pred else 0
    
    pred_root, pred_quality = y_pred.split(":")
    true_root, true_quality = y_true.split(":")
    pred_chroma, true_chroma = root_to_chroma(pred_root), root_to_chroma(true_root)
    if true_chroma == pred_chroma and true_quality == pred_quality:
        return 1
    elif true_chroma == pred_chroma:
        return 0.5
    else:
        return 0


def load_chord_annotations(annotation_file: str) -> List[Tuple[float, float, str]]:
    chords = []
    with open(annotation_file, "r") as f:
        for line in f.readlines():
            start, end, chord_label = line.strip().split()
            chords.append((float(start), float(end), normalize_chord(chord_label)))

    # consolidate neighboring chords with the same label
    consolidated_chords = []
    chord_start = 0
    for i, (start, end, chord) in enumerate(chords):
        if i == len(chords) - 1 or chord != chords[i + 1]:
            consolidated_chords.append((chord_start, end, chord))
            chord_start = end
    return consolidated_chords


def align_chords(gt_chords: List[Tuple[float, float, str]], est_chords: List[Tuple[float, float, str]]) -> List[List[Tuple[float, float, str]]]:
    aligned_chords = []
    for gt_start, gt_end, _ in gt_chords:
        aligned_chords.append([])
        for est_start, est_end, est_chord in est_chords:
            if est_start > gt_end:
                break
            if est_end < gt_start:
                continue
            aligned_chords[-1].append((est_start, est_end, est_chord))
    return aligned_chords


def compute_chord_accuracy(
    gt_chords: List[Tuple[float, float, str]],
    pred_chords: List[Tuple[float, float, str]],
) -> List[float]:
    # If the first chord is a "N" chord, remove it and offset the rest of the chords
    if gt_chords[0][-1] == "N":
        offset = gt_chords[0][1]
        gt_chords = [(start - offset, end - offset, chord) for start, end, chord in gt_chords[1:]]
    if pred_chords[0][-1] == "N":
        offset = pred_chords[0][1]
        pred_chords = [(start - offset, end - offset, chord) for start, end, chord in pred_chords[1:]]

    if len(pred_chords) == 0 or len(gt_chords) == 0:
        if len(gt_chords) == len(pred_chords):
            return [1.0]
        else:
            return [0.0]

    # Normalize the times of estimated chords
    gt_length = gt_chords[-1][1]
    est_length = pred_chords[-1][1]
    scalar = gt_length / est_length
    adjusted_chords = [(start * scalar, end * scalar, chord) for start, end, chord in pred_chords]
    # adjusted_chords = pred_chords

    # For each ground truth chord, find all estimated chords that overlap with it
    aligned_chords = align_chords(gt_chords, adjusted_chords)
    
    # Compute the chord accuracy for each ground truth chord
    accuracies = []
    for (gt_start, gt_end, gt_chord), est_chords in zip(gt_chords, aligned_chords):
        scores = []
        for est_start, est_end, est_chord in est_chords:
            # Compute the chord accuracy weighted by percentage of overlap
            overlap = min(gt_end, est_end) - max(gt_start, est_start)
            weight = overlap / (gt_end - gt_start)
            score = weight * chord_accuracy(gt_chord, est_chord)
            scores.append(score)
        accuracies.append(np.sum(scores))
    return accuracies


def main():
    input_path = "data_cache/isophonics_annotated"

    annotation_files = glob.glob(os.path.join(input_path, "**", "chords.lab"), recursive=True)

    accuracies = []
    for annotation_file in tqdm.tqdm(annotation_files, smoothing=0):
        parent_dir = os.path.dirname(annotation_file)
        midi_files = glob.glob(os.path.join(parent_dir, "*.mid"))

        accs = []
        for midi_file in midi_files:
            try:
                pm = pretty_midi.PrettyMIDI(midi_file)
            except Exception as e:
                # print(f"Could not load {midi_file}: {e}")
                continue

            # Load the chord annotations
            gt_chords = load_chord_annotations(annotation_file)

            # Calculate the estimated chords using the extract_chords function
            pred_chords = extract_chords(pm)
            # pred_chords = extract_chords_legacy(pm)

            micro_accs = compute_chord_accuracy(gt_chords, pred_chords)
            accs.append(np.mean(micro_accs))
        if len(accs) > 0:
            accuracies.append(np.mean(accs))

    print(f"Chord accuracy: {np.mean(accuracies)}")

if __name__ == "__main__":
    main()
