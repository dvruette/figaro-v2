from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union, Dict

import numpy as np
import pretty_midi
from transformers.feature_extraction_utils import BatchFeature

from figaro.preprocessing.utils import normalize_time_sig
from figaro.chord_recognition.algorithms.extract import extract_chords
from figaro.chord_recognition.utils import normalize_chord
from figaro.preprocessing.base import BaseMidiProcessor


DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 33, dtype=int).tolist()
DEFAULT_DURATION_BINS = [
    1, 2, 3, 4, 6, 9,  # sub 1/4th
    12, 15, 18, 21,  # 1/16th steps until 2/4ths
    24, 30, 36, 42,  # 1/8th steps until 4/4ths
    48, 60, 72, 84,  # 1/4th steps until 8/4ths
    96, 120, 144, 168, 192,  # 1/2 steps until 16/4ths
]
DEFAULT_TEMPO_BINS = np.linspace(10, 240, 24, dtype=int).tolist()
DEFAULT_TIME_SIGNATURES = [(i, 4) for i in range(1, 9)]


def _find_closest_value(x, bins):
    return bins[np.argmin(np.abs(np.array(bins) - x))]

def _find_closest_idx(x, bins):
    return np.argmin(np.abs(np.array(bins) - x))


@dataclass
class RemiConfig:
    positions_per_beat: int = 12  # "beat" meaning a quarter note
    max_bars: int = 1024
    max_beats_per_bar: int = 16
    velocity_bins: List[int] = field(default_factory=lambda: DEFAULT_VELOCITY_BINS.copy())
    duration_bins: List[int] = field(default_factory=lambda: DEFAULT_DURATION_BINS.copy())
    tempo_bins: List[int] = field(default_factory=lambda: DEFAULT_TEMPO_BINS.copy())
    valid_time_signatures: List[Tuple[int, int]] = field(default_factory=lambda: DEFAULT_TIME_SIGNATURES.copy())
    extract_chords: bool = True


@dataclass
class RemiNote:
    position: int
    program: int
    is_drum: bool
    pitch: int
    velocity: int
    duration: int

    def __repr__(self):
        return f"<Pitch:{self.pitch}> <Velocity:{self.velocity}> <Duration:{self.duration}>"


@dataclass
class RemiChord:
    chord: str
    position: int

    def __repr__(self):
        return f"<Chord:{normalize_chord(self.chord)}>"


@dataclass
class RemiBar:
    tempo: int
    time_signature: Tuple[int, int]
    chords: List[RemiChord]
    notes: List[RemiNote]

    @property
    def instruments(self) -> Dict[int, List[RemiNote]]:
        instruments = {}
        for note in self.notes:
            if note.program not in instruments:
                instruments[note.program] = []
            instruments[note.program].append(note)
        return instruments

    def __repr__(self):
        all_events = sorted(self.chords + self.notes, key=lambda x: x.position)
        all_positions = [event.position for event in self.chords + self.notes]
        grouped_events = {pos: {"chords": [], "notes": []} for pos in all_positions}
        for event in all_events:
            if isinstance(event, RemiChord):
                grouped_events[event.position]["chords"].append(event)
            else:
                grouped_events[event.position]["notes"].append(event)

        norm_time_sig = normalize_time_sig(self.time_signature)
        tokens = [
            f"<TimeSig:{norm_time_sig[0]}/{norm_time_sig[1]}>",
            f"<Tempo:{self.tempo}>",
        ]

        for pos in sorted(grouped_events.keys()):
            events = grouped_events[pos]
            if len(events["notes"]) == 0:
                continue
            tokens.append(f"<Position:{pos}>")

            for chord in sorted(events["chords"], key=lambda x: x.chord):
                tokens.append(str(chord))

            prev_program = None
            for note in sorted(events["notes"], key=lambda x: (x.program, x.pitch, x.velocity, x.duration)):
                if note.program != prev_program:
                    instrument = "Drum" if note.is_drum else pretty_midi.program_to_instrument_name(note.program)
                    instrument = instrument.replace(" ", "_")
                    tokens.append(f"<Instrument:{instrument}>")
                tokens.append(str(note))
                prev_program = note.program
        
        return " ".join(tokens)

@dataclass
class RemiObject:
    bars: List[RemiBar]

    def __repr__(self):
        return " ".join(f"<Bar:{i}> {bar}" for i, bar in enumerate(self.bars))


class RemiProcessor(BaseMidiProcessor):
    def __init__(
        self,
        positions_per_beat: int = 12,  # "beat" meaning a quarter note
        max_bars: int = 1024,
        max_beats_per_bar: int = 16,
        velocity_bins: List[int] = DEFAULT_VELOCITY_BINS,
        duration_bins: List[int] = DEFAULT_DURATION_BINS,
        tempo_bins: List[int] = DEFAULT_TEMPO_BINS,
        valid_time_signatures: List[Tuple[int, int]] = DEFAULT_TIME_SIGNATURES,
        extract_chords: bool = True,
        **kwargs,
    ):  
        for ts in valid_time_signatures:
            if not len(ts) == 2:
                raise ValueError(f"All time signatures must be of the form (num, denom): found {ts}")
        valid_time_signatures = [normalize_time_sig(tuple(ts)) for ts in valid_time_signatures]

        self.positions_per_beat = positions_per_beat
        self.max_bars = max_bars
        self.max_beats_per_bar = max_beats_per_bar
        self.velocity_bins = velocity_bins
        self.duration_bins = duration_bins
        self.tempo_bins = tempo_bins
        self.valid_time_signatures = valid_time_signatures
        self.extract_chords = extract_chords

    @staticmethod
    def from_config(config: RemiConfig) -> "RemiProcessor":
        return RemiProcessor(
            positions_per_beat=config.positions_per_beat,
            max_bars=config.max_bars,
            max_beats_per_bar=config.max_beats_per_bar,
            velocity_bins=config.velocity_bins,
            duration_bins=config.duration_bins,
            tempo_bins=config.tempo_bins,
            valid_time_signatures=config.valid_time_signatures,
            extract_chords=config.extract_chords,
        )

    def preprocess(self, midi: Union[str, Path, pretty_midi.PrettyMIDI]) -> RemiObject:
        if isinstance(midi, (str, Path)):
            return self.preprocess_midi(midi)
        elif isinstance(midi, pretty_midi.PrettyMIDI):
            return self.preprocess_pretty_midi(midi)
        else:
            raise ValueError(f"Invalid type for midi: {type(midi)}")

    def preprocess_midi(self, midi_path: Union[str, Path]) -> RemiObject:
        pm = pretty_midi.PrettyMIDI(midi_path)
        return self.preprocess_pretty_midi(pm)

    def preprocess_pretty_midi(self, pm: pretty_midi.PrettyMIDI) -> RemiObject:
        if self.extract_chords:
            all_chords = extract_chords(pm)
        else:
            all_chords = []
        downbeats = pm.get_downbeats()
        time_sigs = pm.time_signature_changes
        tempo_change_times, tempi = pm.get_tempo_changes()
        tempo_change_times, tempi = list(tempo_change_times), list(tempi)
        curr_time_sig = None
        curr_tempo = None
        bars = []
        for i in range(len(downbeats)):
            if len(time_sigs) and (curr_time_sig is None or downbeats[i] >= time_sigs[0].time):
                curr_time_sig = time_sigs.pop(0)
            if len(tempi) and len(tempo_change_times) and (curr_tempo is None or downbeats[i] >= tempo_change_times[0]):
                curr_tempo = tempi.pop(0)
                tempo_change_times.pop(0)
            start = downbeats[i]
            end = downbeats[i + 1] if i + 1 < len(downbeats) else start + (downbeats[i - 1] - downbeats[i])  # extrapolate length of second-to-last bar
            bars.append(self._extract_bar(pm, start, end, curr_time_sig, curr_tempo, all_chords))
        return str(RemiObject(bars))
    
    def _extract_bar(
        self,
        pm: pretty_midi.PrettyMIDI,
        start: float,
        end: float,
        time_sig: pretty_midi.TimeSignature,
        tempo: float,
        all_chords: List[Tuple[float, float, str]],
    ) -> RemiBar:
        time_sig = time_sig.numerator, time_sig.denominator
        time_sig = normalize_time_sig(time_sig)
        if time_sig not in self.valid_time_signatures:
            raise ValueError(f"Time signature {time_sig} is not allowed.")
        
        tempo = _find_closest_value(tempo, self.tempo_bins)
        position_grid = self._get_position_grid(start, end, time_sig)
        chords = self._extract_chords(all_chords, start, end, position_grid)
        notes = self._extract_notes(pm, start, end, position_grid)
        
        return RemiBar(
            tempo=tempo,
            time_signature=time_sig,
            chords=chords,
            notes=notes,
        )
    
    def _get_position_grid(self, start: float, end: float, time_sig: Tuple[int, int]) -> np.ndarray:
        """Get a grid of positions for the given bar."""
        num_beats = time_sig[0] * 4 / time_sig[1]
        num_positions = num_beats * self.positions_per_beat
        if not num_positions == int(num_positions):
            raise ValueError(f"Could not evenly divide bar into positions for time signature: {'/'.join(time_sig)}")
        num_positions = int(num_positions)
        
        position_grid = np.linspace(start, end, num_positions, endpoint=False)
        return position_grid

    def _extract_chords(
        self,
        all_chords: List[Tuple[float, float, str]],
        start: float,
        end: float,
        position_grid: np.ndarray,
    ) -> List[RemiChord]:
        """Extract chords from the given bar."""
        chords = []
        for chord_start, chord_end, chord in all_chords:
            if chord_start >= end:
                break
            if chord_end <= start:
                continue
            chord_start = max(chord_start, start)
            chord_start_pos = _find_closest_idx(chord_start, position_grid)
            chords.append(RemiChord(chord, chord_start_pos))
        return chords
    
    def _extract_notes(
        self,
        pm: pretty_midi.PrettyMIDI,
        start: float,
        end: float,
        position_grid: np.ndarray,
    ) -> List[RemiNote]:
        """Extract notes from the given bar."""
        notes = []
        for instrument in pm.instruments:
            if instrument.is_drum:
                program = -1
            else:
                program = instrument.program
            for note in instrument.notes:
                if note.start >= end:
                    break
                if note.start <= start:
                    continue
                note_start_pos = _find_closest_idx(note.start, position_grid)
                # for simplicity, we assume that subsequent bars have the same length
                duration = (note.end - note.start) / (end - start) * len(position_grid)
                duration = _find_closest_value(duration, self.duration_bins)
                note_velocity = _find_closest_value(note.velocity, self.velocity_bins)
                if duration == 0:
                    continue
                notes.append(RemiNote(
                    program=program,
                    is_drum=instrument.is_drum,
                    pitch=note.pitch,
                    velocity=note_velocity,
                    duration=duration,
                    position=note_start_pos,
                ))
        return notes
