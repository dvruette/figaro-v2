import pretty_midi
import music21
import numpy as np
from typing import List

CHORD_TEMPLATES = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7": [0, 4, 10],
    "maj7": [0, 4, 11],
    "min7": [0, 3, 10],
    "sus4": [0, 5, 7],
}


def jaccard_similarity(a: set, b: set) -> float:
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union != 0 else 0


def get_notes_per_bar(pm: pretty_midi.PrettyMIDI):
    # Get the total number of bars
    downbeats = pm.get_downbeats()

    # Initialize a list to hold notes in each bar
    notes_per_bar = [[] for _ in range(len(downbeats))]

    # Iterate through all instruments and their notes
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            # Calculate the bar index for the current note
            # by finding the idx such that downbeats[idx] <= note.start < downbeats[idx + 1]
            start_idx = np.searchsorted(downbeats, note.start, side="right") - 1
            end_idx = np.searchsorted(downbeats, note.end, side="right") - 1
            
            # Add the note pitch to the corresponding bar list
            for i in range(start_idx, end_idx + 1):
                if i < len(downbeats):
                    notes_per_bar[i].append(note.pitch)

    # deduplicate notes
    notes_per_bar = [list(set(notes)) for notes in notes_per_bar]

    return notes_per_bar


def template_matching_chord_recognition(bars: List[List[int]]) -> List[str]:
    chords = []

    for bar in bars:
        best_similarity = 0
        best_chord = "N"  # Initialize with "no chord" (N)

        for root in range(12):  # Iterate over all 12 pitch classes
            for quality, template in CHORD_TEMPLATES.items():
                # Shift the template by the current root and wrap around using modulo 12
                shifted_template = {(pitch + root) % 12 for pitch in template}

                # Calculate the Jaccard similarity between the bar's pitch classes and the shifted template
                similarity = jaccard_similarity({p % 12 for p in bar}, shifted_template)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_chord = f"{music21.pitch.Pitch(root).name}:{quality}"

        chords.append(best_chord)

    return chords


def barwise_chords(pm: pretty_midi.PrettyMIDI) -> List[str]:
    # Get notes for each bar
    notes_per_bar = get_notes_per_bar(pm)

    # Get chords for each bar
    chords_per_bar = template_matching_chord_recognition(notes_per_bar)

    return chords_per_bar