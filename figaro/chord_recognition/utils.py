import re


CHORD_LOOKUP = {
    "CB": "Cb",
    "FB": "Fb",
    "DB": "Db",
    "EB": "Eb",
    "GB": "Gb",
    "AB": "Ab",
    "BB": "Bb",
    "C-": "Cb",
    "D-": "Db",
    "E-": "Eb",
    "F-": "Fb",
    "G-": "Gb",
    "A-": "Ab",
    "B-": "Bb",
}

CHROMA_LOOKUP = {
    "B#": 0,
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

CHORD_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_QUALITIES = ["maj", "min", "dim", "aug", "7", "maj7", "min7", "sus4"]

def normalize_chord(chord: str) -> str:
    if chord == "N":
        return chord

    # remove modifiers
    chord = re.split(r"[/(]", chord)[0]

    if ":" not in chord:
        root, quality = chord, "maj"
    else:
        root, quality = chord.split(":")
    quality = quality.lower()
    root = root[0].upper() + root[1:].lower()
    root = CHORD_LOOKUP.get(root, root)

    if quality == "dom7":
        quality = "7"

    return f"{root}:{quality}"


def root_to_chroma(root: str) -> int:
    root = root[0].upper() + root[1:].lower()
    if not root in CHROMA_LOOKUP:
        raise ValueError(f"Invalid root: {root}")
    return CHROMA_LOOKUP[root]
