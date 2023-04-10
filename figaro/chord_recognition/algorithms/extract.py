from typing import List, Tuple, Literal

import pretty_midi
import numpy as np

import figaro.chord_recognition.algorithms.chroma_scores as chroma
import figaro.chord_recognition.algorithms.heuristic_scores as heuristic


def extract_chords(
    pm: pretty_midi.PrettyMIDI,
    fs: str | float = "auto",
    score_type: Literal["chroma", "heuristic"] = "chroma",
) -> List[Tuple[float, float, str]]:
    # get chromagram
    if fs == "auto":
        median_beat_len = np.median(np.diff(pm.get_beats()))
        fs = 2 / median_beat_len
    chromagram = pm.get_chroma(fs=fs)

    if score_type == "chroma":
        scoring = chroma
    elif score_type == "heuristic":
        scoring = heuristic
    else:
        raise ValueError(f"Unknown score_type: {score_type}")
    
    scores = scoring.compute_scores(chromagram)

    # get most likely chord for each time step
    chord_indices = np.argmax(scores, axis=1)
    chord_indices[scores.sum(axis=1) == 0] = scoring.chord_to_idx("N")
    chords = [scoring.idx_to_chord(i) for i in chord_indices]

    # aggregate neighboring time steps with the same chord
    chord_times = []
    chord_start = 0
    for i, chord in enumerate(chords):
        if i == len(chords) - 1 or chord != chords[i + 1]:
            chord_times.append((chord_start / fs, (i+1) / fs, chord))
            chord_start = i + 1

    return chord_times