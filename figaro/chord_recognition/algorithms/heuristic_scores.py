import functools
from typing import List, Dict

import numpy as np

# define chord maps (required)
CHORD_MAPS = {
    'maj': [0, 4],
    'min': [0, 3],
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    'dom7': [0, 4, 10],
    'maj7': [0, 4, 11],
    'min7': [0, 3, 10],
}
# define chord insiders (+10)
CHORD_INSIDERS = {
    'maj': [7],
    'min': [7],
    'dim': [9],
    'aug': [],
    'dom7': [7],
    'maj7': [7],
    'min7': [7],
}
# define chord outsiders (-1)
CHORD_OUTSIDERS_1 = {
    'maj': [2, 5, 9],
    'min': [2, 5, 8],
    'dim': [2, 5, 10],
    'aug': [2, 5, 9],
    'dom7': [2, 5, 9],
    'maj7': [2, 5, 9],
    'maj7': [2, 5, 9],
    'min7': [2, 5, 8],
}
# define chord outsiders (-2)
CHORD_OUTSIDERS_2 = {
    'maj': [1, 3, 6, 8, 10, 11],
    'min': [1, 4, 6, 9, 11],
    'dim': [1, 4, 7, 8, 11],
    'aug': [1, 3, 6, 7, 10],
    'dom7': [1, 3, 6, 8, 11],
    'maj7': [1, 3, 6, 8, 10],
    'min7': [1, 4, 6, 9, 11],
}

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@functools.lru_cache(maxsize=1)
def _idx2chord() -> List[str]:
    return list(_chord2idx().keys())

@functools.lru_cache(maxsize=1)
def _chord2idx() -> Dict[str, int]:
    chords = {}
    for root in ROOTS:
        for quality in CHORD_MAPS.keys():
            chord = f"{root}:{quality}"
            chords[chord] = len(chords)
    chords["N"] = len(chords)
    return chords

def idx_to_chord(idx: int) -> str:
    return _idx2chord()[idx]

def chord_to_idx(chord: str) -> int:
    return _chord2idx()[chord]


@functools.lru_cache(maxsize=1)
def get_scoring_templates(insider_weight: float = 10, outsider_weight: float = -1) -> dict:
    templates = []
    for i in range(len(ROOTS)):
        for quality in CHORD_MAPS.keys():
            template = np.zeros(12)
            for interval in CHORD_INSIDERS[quality]:
                template[(i + interval) % 12] += insider_weight
            for interval in CHORD_OUTSIDERS_1[quality]:
                template[(i + interval) % 12] += outsider_weight
            for interval in CHORD_OUTSIDERS_2[quality]:
                template[(i + interval) % 12] += 2*outsider_weight
            templates.append(template)
    templates.append(np.ones(12))
    return np.stack(templates)


@functools.lru_cache(maxsize=1)
def get_masking_templates() -> dict:
    templates = []
    for i in range(len(ROOTS)):
        for quality in CHORD_MAPS.keys():
            template = np.zeros(12)
            for interval in CHORD_MAPS[quality]:
                template[(i + interval) % 12] = 1
            templates.append(template)
    templates.append(np.ones(12))
    return np.stack(templates)


def compute_scores(chroma: np.ndarray) -> np.ndarray:
    masking_templates = get_masking_templates()
    scoring_templates = get_scoring_templates()

    chroma = (chroma > 0).astype(int)

    mask = (np.dot(masking_templates, chroma) > 0).astype(int)
    scores = np.dot(scoring_templates, chroma) * mask
    return scores.T
