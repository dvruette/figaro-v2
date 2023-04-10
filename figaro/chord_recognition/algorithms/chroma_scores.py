import functools
from typing import Dict, List

import numpy as np


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

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@functools.lru_cache(maxsize=1)
def _idx2chord() -> List[str]:
    return list(_chord2idx().keys())

@functools.lru_cache(maxsize=1)
def _chord2idx() -> Dict[str, int]:
    chords = {}
    for root in ROOTS:
        for quality in CHORD_TEMPLATES.keys():
            chord = f"{root}:{quality}"
            chords[chord] = len(chords)
    chords["N"] = len(chords)
    return chords

def idx_to_chord(idx: int) -> str:
    return _idx2chord()[idx]

def chord_to_idx(chord: str) -> int:
    return _chord2idx()[chord]


@functools.lru_cache(maxsize=1)
def get_templates() -> Dict[str, np.ndarray]:
    templates = {}
    for i, root in enumerate(ROOTS):
        for quality, template in CHORD_TEMPLATES.items():
            chord = f"{root}:{quality}"
            templates[chord] = np.zeros(12)
            for pitch in template:
                templates[chord][(pitch + i) % 12] = 1
    templates["N"] = np.ones(12)
    return templates


def compute_scores(chroma: np.ndarray) -> np.ndarray:
    templates = get_templates()
    template_array = np.array(list(templates.values()))

    template_array = template_array / np.linalg.norm(template_array)
    chroma = (chroma > 0).astype(int)
    chroma = chroma / np.linalg.norm(chroma)

    # get cosine similarity between templates and chroma
    scores = np.dot(chroma.T, template_array.T)
    return scores
