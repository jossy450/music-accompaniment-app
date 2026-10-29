# In app/chord_progression_generator.py

import numpy as np
import librosa
from typing import List, Dict

class ChordProgressionGenerator:
    """
    Generates a dynamic chord progression based on the vocal melody's contour and harmonic implications.
    """

    def __init__(self, key: str, scale: str, tempo: float):
        self.key = key
        self.scale = scale
        self.tempo = tempo

    def generate(self, vocal_melody: np.ndarray, sr: int) -> List[str]:
        """
        Analyzes the vocal melody and generates a suitable chord progression.
        The progression is a list of Roman numeral chord symbols.
        """
        # 1. Estimate pitch (F0) contour
        f0, _, _ = librosa.pyin(vocal_melody, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        
        # 2. Convert F0 to MIDI notes, ignoring silence (NaNs)
        midi_notes = librosa.hz_to_midi(f0[~np.isnan(f0)])
        
        if len(midi_notes) == 0:
            return self._get_static_progression()

        # 3. Quantize to beats (assuming 4 beats per bar for simplicity)
        beats_per_bar = 4
        beat_duration_samples = int((60 / self.tempo) * sr)
        
        # 4. Determine the root note of the key
        key_root_midi = int(np.ravel(librosa.note_to_midi(self.key + '4'))[0])
        
        # 5. Define Roman numeral mapping (simplified for major/minor)
        if self.scale == "major":
            roman_map = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii°"}
        elif self.scale == "minor":
            roman_map = {0: "i", 2: "ii°", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}
        else:
            return self._get_static_progression()

        progression = []
        # Iterate over 4-beat segments (bars)
        for i in range(0, len(vocal_melody), beat_duration_samples * beats_per_bar):
            segment = vocal_melody[i : i + beat_duration_samples * beats_per_bar]
            
            if len(segment) < beat_duration_samples * 2: # Skip very short tail segments
                break

            # Re-estimate pitch for the segment
            f0_seg, _, _ = librosa.pyin(segment, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            midi_notes_seg = librosa.hz_to_midi(f0_seg[~np.isnan(f0_seg)])
            
            if len(midi_notes_seg) == 0:
                # If no vocal in this bar, use the previous chord or default to I
                progression.append(progression[-1] if progression else roman_map.get(0))
                continue

            # Find the most common note in the segment (melody root)
            unique, counts = np.unique(np.round(midi_notes_seg) % 12, return_counts=True)
            melody_root_chroma = unique[np.argmax(counts)]
            
            # Calculate interval from key root
            interval = (melody_root_chroma - (key_root_midi % 12)) % 12
            
            # Simple rule: use the chord whose root is the most common melody note
            # This is a simplification of harmonic analysis (e.g., ignoring V7, secondary dominants)
            chord_symbol = roman_map.get(interval, roman_map.get(0))
            
            # Basic progression smoothing (avoid too many rapid changes)
            if progression and progression[-1] == chord_symbol:
                # Extend the current chord
                pass
            elif progression and len(progression) > 1 and progression[-2] == chord_symbol:
                # Avoid simple back-and-forth
                progression.append(progression[-1])
            else:
                progression.append(chord_symbol)

        # Ensure a minimum length for the progression
        if not progression:
            return self._get_static_progression()
            
        # Post-process: ensure progression ends on a tonic (I or i)
        if progression[-1] not in ["I", "i"]:
            progression.append(roman_map.get(0))
            
        return progression

    def _get_static_progression(self) -> List[str]:
        """
        Returns a static chord progression for the given key and scale.
        """
        if self.scale == "major":
            return ["I", "V", "vi", "IV"] * 2
        elif self.scale == "minor":
            return ["i", "VI", "III", "VII"] * 2
        else:
            return ["I", "V", "vi", "IV"] * 2
