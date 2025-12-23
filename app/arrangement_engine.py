# In app/arrangement_engine.py - PROFESSIONAL VERSION

import numpy as np
import librosa
from typing import Dict, List, Tuple
import pretty_midi
from .chord_progression_generator import ChordProgressionGenerator

class VocalFollowingArrangementEngine:
    """
    Advanced arrangement engine that listens to the vocal and generates
    accompaniment that follows its timing, phrasing, and emotional arc.
    The band adapts to the vocal, not the other way around.
    """
    
    def __init__(self, vocal_analysis: Dict, genre: str, sr: int = 44100):
        self.vocal_analysis = vocal_analysis
        self.genre = genre
        self.sr = sr
        self.tempo = vocal_analysis.get('rhythm', {}).get('tempo', 120)
        self.key = vocal_analysis.get('pitch', {}).get('key', 'C')
        self.scale = vocal_analysis.get('pitch', {}).get('scale', 'major')
        self.phrases = vocal_analysis.get('phrases', [])
        self.chord_generator = ChordProgressionGenerator(self.key, self.scale, self.tempo)
    
    def _analyze_vocal_phrasing(self, vocal: np.ndarray) -> List[Dict]:
        """
        Analyze the vocal to understand phrase structure, dynamics, and breathing.
        This tells the band when to play and when to give space.
        """
        # Detect phrase boundaries (where the vocal pauses for breath)
        S = librosa.feature.melspectrogram(y=vocal, sr=self.sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        energy = np.mean(S_db, axis=0)
        
        # Normalize energy
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
        
        # Detect phrase boundaries (sudden drops in energy)
        energy_diff = np.diff(energy_norm)
        phrase_boundaries = np.where(energy_diff < -0.3)[0]
        
        phrases = []
        prev_boundary = 0
        for boundary in phrase_boundaries:
            phrase_start = prev_boundary / self.sr
            phrase_end = boundary / self.sr
            phrase_energy = np.mean(energy_norm[prev_boundary:boundary])
            phrases.append({
                'start': phrase_start,
                'end': phrase_end,
                'duration': phrase_end - phrase_start,
                'energy': phrase_energy,
                'type': 'vocal_phrase'
            })
            prev_boundary = boundary
        
        # Add final phrase
        if prev_boundary < len(energy_norm):
            phrases.append({
                'start': prev_boundary / self.sr,
                'end': len(vocal) / self.sr,
                'duration': (len(vocal) - prev_boundary) / self.sr,
                'energy': np.mean(energy_norm[prev_boundary:]),
                'type': 'vocal_phrase'
            })
        
        # Store the phrases in the instance for later use
        self.phrases = phrases
        return phrases
    
    def _generate_accompaniment_for_phrase(self, phrase: Dict, 
                                          instruments: List[str],
                                          chord_progression: List[str]) -> pretty_midi.PrettyMIDI:
        """
        Generate MIDI accompaniment for a single vocal phrase.
        The band responds to the vocal's energy and timing.
        """
        midi = pretty_midi.PrettyMIDI()
        midi.instruments.append(pretty_midi.Instrument(program=0, name='Piano'))
        
        # Chord progression is now passed in, ensuring musical consistency across phrases.
        # We will cycle through the progression for the duration of the phrase.
        
        phrase_duration = phrase['end'] - phrase['start']
        beat_duration = 60.0 / self.tempo  # Duration of one beat
        
        # Generate chords that respond to vocal energy
        chord_index = 0
        current_time = phrase['start']
        
        while current_time < phrase['end']:
            chord = chord_progression[chord_index % len(chord_progression)]
            chord_notes = self._chord_to_notes(chord, self.key)
            
            # We want the chords to change musically, e.g., every 2 or 4 beats.
            # Let's default to 4 beats (a bar) for musical progress.
            chord_duration = beat_duration * 4 
            
            # If the phrase is very short, use a shorter chord duration
            if phrase_duration < beat_duration * 4:
                chord_duration = beat_duration * 2
            
            chord_end = min(current_time + chord_duration, phrase['end'])
            
            # Add chord notes
            for note_pitch in chord_notes:
                note = pretty_midi.Note(velocity=80, pitch=note_pitch, 
                                       start=current_time, end=chord_end)
                midi.instruments[0].notes.append(note)
            
            current_time = chord_end
            chord_index += 1
        
        return midi
    
    def _get_chord_progression_for_vocal(self, vocal: np.ndarray) -> List[str]:
        """Generate a chord progression based on vocal analysis."""
        # For now, we use the placeholder generator. This will be enhanced in Phase 3.
        return self.chord_generator.generate(vocal, self.sr)
    
    def _chord_to_notes(self, chord: str, key: str) -> List[int]:
        """Convert a chord symbol to MIDI note numbers."""
        # Note: This is a simplified version. A full implementation would handle
        # all chord types (maj7, min7, etc.)
        # The full chord logic will be moved to ChordProgressionGenerator in Phase 3.
        
        note_map = {
            'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65,
            'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71
        }
        
        root = note_map.get(key, 60)
        
        # Determine chord intervals based on chord type
        if chord == 'I':
            intervals = [0, 4, 7]  # Major triad
        elif chord == 'ii':
            intervals = [2, 5, 9]  # Minor triad
        elif chord == 'IV':
            intervals = [5, 9, 12]  # Major triad
        elif chord == 'V':
            intervals = [7, 11, 14]  # Major triad
        elif chord == 'vi':
            intervals = [9, 12, 16]  # Minor triad
        else:
            intervals = [0, 4, 7]  # Default to major
        
        return [root + interval for interval in intervals]
    
    def create_vocal_following_arrangement(self, vocal: np.ndarray, 
                                          instruments: List[str],
                                          complexity: str = 'balanced') -> pretty_midi.PrettyMIDI:
        """
        Create a complete arrangement that follows the vocal's phrasing and energy.
        This is the core of the "band follows vocal" approach.
        """
        midi = pretty_midi.PrettyMIDI()
        
        # Analyze vocal phrasing
        phrases = self._analyze_vocal_phrasing(vocal)
        
        # Add instruments
        instrument_programs = {
            'drums': 0,
            'bass': 33,
            'piano': 0,
            'guitar': 24,
            'strings': 48,
            'brass': 60,
            'organ': 19,
            'sax': 64,
            'synth_pad': 88,
            'percussion': 0,
        }
        
        for inst in instruments:
            program = instrument_programs.get(inst, 0)
            midi.instruments.append(pretty_midi.Instrument(program=program, name=inst))
        
        # Generate accompaniment for each vocal phrase
        
        # Generate a global chord progression based on the entire vocal
        # This progression is now dynamic and follows the vocal melody.
        global_progression = self._get_chord_progression_for_vocal(vocal)
        
        for phrase_idx, phrase in enumerate(phrases):
            # Generate bass line
            if 'bass' in instruments:
                bass_midi = self._generate_bass_line(phrase, self.key, self.tempo)
                for note in bass_midi.instruments[0].notes:
                    midi.instruments[instruments.index('bass')].notes.append(note)
            
            # Generate drum pattern
            if 'drums' in instruments:
                drum_midi = self._generate_drum_pattern(phrase, self.tempo, phrase['energy'])
                for note in drum_midi.instruments[0].notes:
                    midi.instruments[instruments.index('drums')].notes.append(note)
            
            # Generate harmonic accompaniment (piano, strings, etc.)
            for inst in ['piano', 'strings', 'organ']:
                if inst in instruments:
                    # Pass the global progression to the phrase generator
                    harm_midi = self._generate_accompaniment_for_phrase(phrase, [inst], global_progression)
                    for note in harm_midi.instruments[0].notes:
                        midi.instruments[instruments.index(inst)].notes.append(note)
        
        return midi
    
    def _generate_bass_line(self, phrase: Dict, key: str, tempo: float) -> pretty_midi.PrettyMIDI:
        """Generate a bass line that follows the vocal phrase."""
        midi = pretty_midi.PrettyMIDI()
        midi.instruments.append(pretty_midi.Instrument(program=33, name='Bass'))
        
        note_map = {
            'C': 36, 'C#': 37, 'D': 38, 'D#': 39, 'E': 40, 'F': 41,
            'F#': 42, 'G': 43, 'G#': 44, 'A': 45, 'A#': 46, 'B': 47
        }
        
        root_note = note_map.get(key, 36)
        beat_duration = 60.0 / tempo
        
        current_time = phrase['start']
        while current_time < phrase['end']:
            note = pretty_midi.Note(velocity=70, pitch=root_note,
                                   start=current_time, end=current_time + beat_duration)
            midi.instruments[0].notes.append(note)
            current_time += beat_duration
        
        return midi
    
    def _generate_drum_pattern(self, phrase: Dict, tempo: float, 
                              energy: float) -> pretty_midi.PrettyMIDI:
        """Generate a drum pattern that responds to vocal energy."""
        midi = pretty_midi.PrettyMIDI()
        midi.instruments.append(pretty_midi.Instrument(program=0, name='Drums'))
        
        beat_duration = 60.0 / tempo
        kick_pitch = 36  # Kick drum
        snare_pitch = 38  # Snare drum
        
        current_time = phrase['start']
        beat_count = 0
        
        while current_time < phrase['end']:
            # Kick on beats 1 and 3
            if beat_count % 4 in [0, 2]:
                kick = pretty_midi.Note(velocity=90, pitch=kick_pitch,
                                       start=current_time, end=current_time + beat_duration * 0.9)
                midi.instruments[0].notes.append(kick)
            
            # Snare on beats 2 and 4 (if energy is high)
            if beat_count % 4 in [1, 3] and energy > 0.5:
                snare = pretty_midi.Note(velocity=80, pitch=snare_pitch,
                                        start=current_time, end=current_time + beat_duration * 0.9)
                midi.instruments[0].notes.append(snare)
            
            current_time += beat_duration
            beat_count += 1
        
        return midi
