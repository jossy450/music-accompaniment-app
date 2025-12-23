# FIXED ARRANGEMENT ENGINE - Correct MIDI programs and proper instrument selection

import numpy as np
import pretty_midi
from typing import Dict, List

class VocalResponsiveArrangementEngine:
    """
    Generates MIDI accompaniment with CORRECT instrument programs.
    """
    
    # MIDI Program numbers for instruments
    MIDI_PROGRAMS = {
        'drums': (0, True),  # (program, is_drum)
        'bass': (33, False),  # Electric bass
        'piano': (0, False),  # Acoustic piano
        'strings': (48, False),  # String ensemble
        'guitar': (25, False),  # Acoustic guitar
        'organ': (19, False),  # Church organ
        'synth': (80, False),  # Square wave synth
        'flute': (73, False),  # Flute
        'sax': (64, False),  # Alto sax
        'trumpet': (56, False),  # Trumpet
    }
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Chord progressions
        self.progressions = {
            'gospel': [['I', 'IV', 'I', 'V'], ['I', 'vi', 'IV', 'V']],
            'pop': [['I', 'V', 'vi', 'IV'], ['vi', 'IV', 'I', 'V']],
            'rock': [['I', 'IV', 'V', 'IV'], ['I', 'V', 'vi', 'IV']],
            'r&b': [['I', 'vi', 'IV', 'V'], ['vi', 'IV', 'I', 'V']],
            'jazz': [['ii', 'V', 'I', 'IV'], ['I', 'vi', 'ii', 'V']],
            'ballad': [['I', 'V', 'vi', 'IV'], ['vi', 'IV', 'I', 'V']],
        }
    
    def generate_vocal_responsive_arrangement(
        self, 
        vocal_analysis: Dict,
        instruments: List[str],
        genre: str = 'gospel',
        complexity: str = 'medium'
    ) -> pretty_midi.PrettyMIDI:
        """Generate MIDI with correct instrument programs."""
        print(f"[ARRANGEMENT] Generating accompaniment for {genre}...")
        
        # Get vocal characteristics
        key = vocal_analysis['key']
        scale = vocal_analysis['scale']
        phrases = vocal_analysis['timing']['phrases']
        intensity = vocal_analysis['intensity']['intensity']
        
        # Calculate tempo
        if phrases:
            avg_phrase_duration = vocal_analysis['timing']['average_phrase_duration']
            tempo = self._calculate_tempo(avg_phrase_duration)
        else:
            tempo = 120
        
        # Create MIDI with correct tempo - FIXED HERE
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)  # Set tempo during creation
        
        print(f"  ✓ Tempo: {tempo:.0f} BPM, Key: {key} {scale}")
        
        # Generate chord progression
        chord_progression = self._generate_chord_progression(len(phrases), genre)
        
        # Add instruments with CORRECT MIDI programs
        if not instruments:
            instruments = ['drums', 'bass', 'piano']
        
        for instrument_name in instruments:
            if instrument_name.lower() == 'drums':
                self._add_drums(midi, phrases, intensity, tempo, complexity)
            elif instrument_name.lower() == 'bass':
                self._add_bass(midi, chord_progression, key, phrases, tempo)
            elif instrument_name.lower() == 'piano':
                self._add_piano(midi, chord_progression, key, phrases, intensity, tempo)
            elif instrument_name.lower() == 'strings':
                self._add_strings(midi, chord_progression, key, phrases, intensity, tempo)
            elif instrument_name.lower() == 'guitar':
                self._add_guitar(midi, chord_progression, key, phrases, intensity, tempo)
            elif instrument_name.lower() == 'organ':
                self._add_organ(midi, chord_progression, key, phrases, intensity, tempo)
            elif instrument_name.lower() == 'synth':
                self._add_synth(midi, chord_progression, key, phrases, intensity, tempo)
            elif instrument_name.lower() == 'flute':
                self._add_flute(midi, chord_progression, key, phrases, intensity, tempo)
            elif instrument_name.lower() == 'sax':
                self._add_sax(midi, chord_progression, key, phrases, intensity, tempo)
            elif instrument_name.lower() == 'trumpet':
                self._add_trumpet(midi, chord_progression, key, phrases, intensity, tempo)
        
        print(f"  ✓ Generated {len(midi.instruments)} instruments")
        return midi
    
    def _calculate_tempo(self, avg_phrase_duration: float) -> float:
        """Calculate tempo from phrase duration."""
        if avg_phrase_duration < 2:
            return 140
        elif avg_phrase_duration < 3:
            return 120
        elif avg_phrase_duration < 5:
            return 100
        else:
            return 80
    
    def _generate_chord_progression(self, num_phrases: int, genre: str) -> List[str]:
        """Generate chord progression."""
        progressions = self.progressions.get(genre, self.progressions['gospel'])
        selected = progressions[num_phrases % len(progressions)]
        
        # Repeat to cover all phrases
        full_progression = []
        for i in range(num_phrases):
            full_progression.append(selected[i % len(selected)])
        
        return full_progression
    
    def _add_drums(self, midi: pretty_midi.PrettyMIDI, phrases: List[Dict], 
                   intensity: np.ndarray, tempo: float, complexity: str):
        """Add drum track with CORRECT MIDI program."""
        print("  [DRUMS] Adding drums...")
        
        # Create drum track with is_drum=True
        drum_track = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')
        
        beat_duration = 60 / tempo
        bar_duration = beat_duration * 4
        
        for phrase in phrases:
            phrase_start = phrase['start']
            phrase_duration = phrase['duration']
            
            # Get intensity for this phrase
            if phrase_duration > 0 and len(intensity) > 0:
                start_idx = int(phrase_start * self.sr / self.hop_length)
                end_idx = int((phrase_start + phrase_duration) * self.sr / self.hop_length)
                if start_idx < len(intensity) and end_idx < len(intensity):
                    phrase_intensity = np.mean(intensity[start_idx:end_idx])
                else:
                    phrase_intensity = 0.5
            else:
                phrase_intensity = 0.5
            
            num_bars = int(phrase_duration / bar_duration) + 1
            
            for bar in range(num_bars):
                # Kick drum (MIDI 36)
                for beat in range(4):
                    note_time = phrase_start + bar * bar_duration + beat * beat_duration
                    if note_time < phrase_start + phrase_duration:
                        velocity = int(80 + phrase_intensity * 40)
                        note = pretty_midi.Note(velocity, 36, note_time, note_time + 0.1)
                        drum_track.notes.append(note)
                
                # Hi-hat (MIDI 42)
                for beat in [0.5, 1.5, 2.5, 3.5]:
                    note_time = phrase_start + bar * bar_duration + beat * beat_duration
                    if note_time < phrase_start + phrase_duration:
                        velocity = int(60 + phrase_intensity * 30)
                        note = pretty_midi.Note(velocity, 42, note_time, note_time + 0.08)
                        drum_track.notes.append(note)
        
        midi.instruments.append(drum_track)
    
    def _add_bass(self, midi: pretty_midi.PrettyMIDI, chord_progression: List[str],
                  key: str, phrases: List[Dict], tempo: float):
        """Add bass with CORRECT MIDI program (33 = Electric Bass)."""
        print("  [BASS] Adding bass...")
        
        bass_track = pretty_midi.Instrument(program=33, name='Bass')
        
        beat_duration = 60 / tempo
        bar_duration = beat_duration * 4
        
        for phrase_idx, phrase in enumerate(phrases):
            phrase_start = phrase['start']
            phrase_duration = phrase['duration']
            chord = chord_progression[phrase_idx % len(chord_progression)]
            
            root_midi = self._chord_to_midi(chord, key)
            bass_note = root_midi - 12
            
            num_bars = int(phrase_duration / bar_duration) + 1
            
            for bar in range(num_bars):
                # Root on beat 1
                note_time = phrase_start + bar * bar_duration
                if note_time < phrase_start + phrase_duration:
                    note = pretty_midi.Note(100, bass_note, note_time, note_time + beat_duration * 0.9)
                    bass_track.notes.append(note)
                
                # Fifth on beat 3
                note_time = phrase_start + bar * bar_duration + beat_duration * 2
                if note_time < phrase_start + phrase_duration:
                    fifth = bass_note + 7
                    note = pretty_midi.Note(90, fifth, note_time, note_time + beat_duration * 0.9)
                    bass_track.notes.append(note)
        
        midi.instruments.append(bass_track)
    
    def _add_piano(self, midi: pretty_midi.PrettyMIDI, chord_progression: List[str],
                   key: str, phrases: List[Dict], intensity: np.ndarray, tempo: float):
        """Add piano with CORRECT MIDI program (0 = Acoustic Piano)."""
        print("  [PIANO] Adding piano...")
        
        piano_track = pretty_midi.Instrument(program=0, name='Piano')
        
        beat_duration = 60 / tempo
        bar_duration = beat_duration * 4
        
        for phrase_idx, phrase in enumerate(phrases):
            phrase_start = phrase['start']
            phrase_duration = phrase['duration']
            chord = chord_progression[phrase_idx % len(chord_progression)]
            
            chord_notes = self._chord_to_midi_notes(chord, key)
            
            if phrase_duration > 0 and len(intensity) > 0:
                start_idx = int(phrase_start * self.sr / self.hop_length)
                end_idx = int((phrase_start + phrase_duration) * self.sr / self.hop_length)
                if start_idx < len(intensity) and end_idx < len(intensity):
                    phrase_intensity = np.mean(intensity[start_idx:end_idx])
                else:
                    phrase_intensity = 0.5
            else:
                phrase_intensity = 0.5
            
            num_bars = int(phrase_duration / bar_duration) + 1
            
            for bar in range(num_bars):
                for beat in range(4):
                    note_time = phrase_start + bar * bar_duration + beat * beat_duration
                    if note_time < phrase_start + phrase_duration:
                        note_pitch = chord_notes[beat % len(chord_notes)]
                        velocity = int(70 + phrase_intensity * 40)
                        note = pretty_midi.Note(velocity, note_pitch, note_time, note_time + beat_duration * 0.8)
                        piano_track.notes.append(note)
        
        midi.instruments.append(piano_track)
    
    def _add_strings(self, midi: pretty_midi.PrettyMIDI, chord_progression: List[str],
                     key: str, phrases: List[Dict], intensity: np.ndarray, tempo: float):
        """Add strings with CORRECT MIDI program (48 = String Ensemble)."""
        print("  [STRINGS] Adding strings...")
        
        strings_track = pretty_midi.Instrument(program=48, name='Strings')
        
        for phrase_idx, phrase in enumerate(phrases):
            phrase_start = phrase['start']
            phrase_duration = phrase['duration']
            chord = chord_progression[phrase_idx % len(chord_progression)]
            
            chord_notes = self._chord_to_midi_notes(chord, key, octave_offset=2)
            
            if phrase_duration > 0 and len(intensity) > 0:
                start_idx = int(phrase_start * self.sr / self.hop_length)
                end_idx = int((phrase_start + phrase_duration) * self.sr / self.hop_length)
                if start_idx < len(intensity) and end_idx < len(intensity):
                    phrase_intensity = np.mean(intensity[start_idx:end_idx])
                else:
                    phrase_intensity = 0.5
            else:
                phrase_intensity = 0.5
            
            for note_pitch in chord_notes:
                velocity = int(60 + phrase_intensity * 30)
                note = pretty_midi.Note(velocity, note_pitch, phrase_start, phrase_start + phrase_duration)
                strings_track.notes.append(note)
        
        midi.instruments.append(strings_track)
    
    def _add_guitar(self, midi: pretty_midi.PrettyMIDI, chord_progression: List[str],
                    key: str, phrases: List[Dict], intensity: np.ndarray, tempo: float):
        """Add guitar with CORRECT MIDI program (25 = Acoustic Guitar)."""
        print("  [GUITAR] Adding guitar...")
        
        guitar_track = pretty_midi.Instrument(program=25, name='Guitar')
        
        beat_duration = 60 / tempo
        bar_duration = beat_duration * 4
        
        for phrase_idx, phrase in enumerate(phrases):
            phrase_start = phrase['start']
            phrase_duration = phrase['duration']
            chord = chord_progression[phrase_idx % len(chord_progression)]
            
            chord_notes = self._chord_to_midi_notes(chord, key, octave_offset=1)
            
            if phrase_duration > 0 and len(intensity) > 0:
                start_idx = int(phrase_start * self.sr / self.hop_length)
                end_idx = int((phrase_start + phrase_duration) * self.sr / self.hop_length)
                if start_idx < len(intensity) and end_idx < len(intensity):
                    phrase_intensity = np.mean(intensity[start_idx:end_idx])
                else:
                    phrase_intensity = 0.5
            else:
                phrase_intensity = 0.5
            
            num_bars = int(phrase_duration / bar_duration) + 1
            
            for bar in range(num_bars):
                for beat in range(4):
                    note_time = phrase_start + bar * bar_duration + beat * beat_duration
                    if note_time < phrase_start + phrase_duration:
                        velocity = int(75 + phrase_intensity * 35)
                        for note_pitch in chord_notes:
                            note = pretty_midi.Note(velocity, note_pitch, note_time, note_time + beat_duration * 0.7)
                            guitar_track.notes.append(note)
        
        midi.instruments.append(guitar_track)
    
    def _add_organ(self, midi: pretty_midi.PrettyMIDI, chord_progression: List[str],
                   key: str, phrases: List[Dict], intensity: np.ndarray, tempo: float):
        """Add organ with CORRECT MIDI program (19 = Church Organ)."""
        print("  [ORGAN] Adding organ...")
        
        organ_track = pretty_midi.Instrument(program=19, name='Organ')
        
        for phrase_idx, phrase in enumerate(phrases):
            phrase_start = phrase['start']
            phrase_duration = phrase['duration']
            chord = chord_progression[phrase_idx % len(chord_progression)]
            
            chord_notes = self._chord_to_midi_notes(chord, key, octave_offset=1)
            
            if phrase_duration > 0 and len(intensity) > 0:
                start_idx = int(phrase_start * self.sr / self.hop_length)
                end_idx = int((phrase_start + phrase_duration) * self.sr / self.hop_length)
                if start_idx < len(intensity) and end_idx < len(intensity):
                    phrase_intensity = np.mean(intensity[start_idx:end_idx])
                else:
                    phrase_intensity = 0.5
            else:
                phrase_intensity = 0.5
            
            for note_pitch in chord_notes:
                velocity = int(70 + phrase_intensity * 35)
                note = pretty_midi.Note(velocity, note_pitch, phrase_start, phrase_start + phrase_duration)
                organ_track.notes.append(note)
        
        midi.instruments.append(organ_track)
    
    def _add_synth(self, midi: pretty_midi.PrettyMIDI, chord_progression: List[str],
                   key: str, phrases: List[Dict], intensity: np.ndarray, tempo: float):
        """Add synth with CORRECT MIDI program (80 = Square wave synth)."""
        print("  [SYNTH] Adding synth...")
        
        synth_track = pretty_midi.Instrument(program=80, name='Synth')
        
        beat_duration = 60 / tempo
        bar_duration = beat_duration * 4
        
        for phrase_idx, phrase in enumerate(phrases):
            phrase_start = phrase['start']
            phrase_duration = phrase['duration']
            chord = chord_progression[phrase_idx % len(chord_progression)]
            
            chord_notes = self._chord_to_midi_notes(chord, key, octave_offset=2)
            
            if phrase_duration > 0 and len(intensity) > 0:
                start_idx = int(phrase_start * self.sr / self.hop_length)
                end_idx = int((phrase_start + phrase_duration) * self.sr / self.hop_length)
                if start_idx < len(intensity) and end_idx < len(intensity):
                    phrase_intensity = np.mean(intensity[start_idx:end_idx])
                else:
                    phrase_intensity = 0.5
            else:
                phrase_intensity = 0.5
            
            num_bars = int(phrase_duration / bar_duration) + 1
            
            for bar in range(num_bars):
                note_time = phrase_start + bar * bar_duration
                if note_time < phrase_start + phrase_duration:
                    velocity = int(75 + phrase_intensity * 35)
                    for note_pitch in chord_notes[:2]:  # Just root and third
                        note = pretty_midi.Note(velocity, note_pitch, note_time, note_time + bar_duration * 0.9)
                        synth_track.notes.append(note)
        
        midi.instruments.append(synth_track)
    
    def _add_flute(self, midi: pretty_midi.PrettyMIDI, chord_progression: List[str],
                   key: str, phrases: List[Dict], intensity: np.ndarray, tempo: float):
        """Add flute with CORRECT MIDI program (73 = Flute)."""
        print("  [FLUTE] Adding flute...")
        
        flute_track = pretty_midi.Instrument(program=73, name='Flute')
        
        beat_duration = 60 / tempo
        bar_duration = beat_duration * 4
        
        for phrase_idx, phrase in enumerate(phrases):
            phrase_start = phrase['start']
            phrase_duration = phrase['duration']
            chord = chord_progression[phrase_idx % len(chord_progression)]
            
            root_midi = self._chord_to_midi(chord, key) + 12  # One octave higher
            
            if phrase_duration > 0 and len(intensity) > 0:
                start_idx = int(phrase_start * self.sr / self.hop_length)
                end_idx = int((phrase_start + phrase_duration) * self.sr / self.hop_length)
                if start_idx < len(intensity) and end_idx < len(intensity):
                    phrase_intensity = np.mean(intensity[start_idx:end_idx])
                else:
                    phrase_intensity = 0.5
            else:
                phrase_intensity = 0.5
            
            num_bars = int(phrase_duration / bar_duration) + 1
            
            for bar in range(num_bars):
                for beat in [0, 2]:  # Play on beats 1 and 3
                    note_time = phrase_start + bar * bar_duration + beat * beat_duration
                    if note_time < phrase_start + phrase_duration:
                        velocity = int(70 + phrase_intensity * 35)
                        note = pretty_midi.Note(velocity, root_midi, note_time, note_time + beat_duration * 2)
                        flute_track.notes.append(note)
        
        midi.instruments.append(flute_track)
    
    def _add_sax(self, midi: pretty_midi.PrettyMIDI, chord_progression: List[str],
                 key: str, phrases: List[Dict], intensity: np.ndarray, tempo: float):
        """Add sax with CORRECT MIDI program (64 = Alto Sax)."""
        print("  [SAX] Adding sax...")
        
        sax_track = pretty_midi.Instrument(program=64, name='Sax')
        
        for phrase_idx, phrase in enumerate(phrases):
            phrase_start = phrase['start']
            phrase_duration = phrase['duration']
            chord = chord_progression[phrase_idx % len(chord_progression)]
            
            root_midi = self._chord_to_midi(chord, key) + 12  # One octave higher
            
            if phrase_duration > 0 and len(intensity) > 0:
                start_idx = int(phrase_start * self.sr / self.hop_length)
                end_idx = int((phrase_start + phrase_duration) * self.sr / self.hop_length)
                if start_idx < len(intensity) and end_idx < len(intensity):
                    phrase_intensity = np.mean(intensity[start_idx:end_idx])
                else:
                    phrase_intensity = 0.5
            else:
                phrase_intensity = 0.5
            
            # Play the root note for the entire phrase
            velocity = int(80 + phrase_intensity * 35)
            note = pretty_midi.Note(velocity, root_midi, phrase_start, phrase_start + phrase_duration)
            sax_track.notes.append(note)
        
        midi.instruments.append(sax_track)
    
    def _add_trumpet(self, midi: pretty_midi.PrettyMIDI, chord_progression: List[str],
                     key: str, phrases: List[Dict], intensity: np.ndarray, tempo: float):
        """Add trumpet with CORRECT MIDI program (56 = Trumpet)."""
        print("  [TRUMPET] Adding trumpet...")
        
        trumpet_track = pretty_midi.Instrument(program=56, name='Trumpet')
        
        beat_duration = 60 / tempo
        bar_duration = beat_duration * 4
        
        for phrase_idx, phrase in enumerate(phrases):
            phrase_start = phrase['start']
            phrase_duration = phrase['duration']
            chord = chord_progression[phrase_idx % len(chord_progression)]
            
            root_midi = self._chord_to_midi(chord, key) + 12  # One octave higher
            
            if phrase_duration > 0 and len(intensity) > 0:
                start_idx = int(phrase_start * self.sr / self.hop_length)
                end_idx = int((phrase_start + phrase_duration) * self.sr / self.hop_length)
                if start_idx < len(intensity) and end_idx < len(intensity):
                    phrase_intensity = np.mean(intensity[start_idx:end_idx])
                else:
                    phrase_intensity = 0.5
            else:
                phrase_intensity = 0.5
            
            num_bars = int(phrase_duration / bar_duration) + 1
            
            for bar in range(num_bars):
                note_time = phrase_start + bar * bar_duration + beat_duration * 1  # Beat 2
                if note_time < phrase_start + phrase_duration:
                    velocity = int(85 + phrase_intensity * 30)
                    note = pretty_midi.Note(velocity, root_midi, note_time, note_time + beat_duration)
                    trumpet_track.notes.append(note)
        
        midi.instruments.append(trumpet_track)
    
    def _chord_to_midi(self, chord: str, key: str) -> int:
        """Convert chord to MIDI note."""
        roman_to_interval = {
            'I': 0, 'ii': 2, 'III': 4, 'iii': 3, 'IV': 5, 'V': 7, 
            'VI': 9, 'vi': 8, 'VII': 11, 'vii': 10
        }
        
        interval = roman_to_interval.get(chord, 0)
        key_midi = self._note_to_midi(key)
        
        return key_midi + interval
    
    def _chord_to_midi_notes(self, chord: str, key: str, octave_offset: int = 0) -> List[int]:
        """Convert chord to MIDI notes."""
        root = self._chord_to_midi(chord, key)
        third = root + 4
        fifth = root + 7
        
        return [root + (octave_offset * 12), third + (octave_offset * 12), fifth + (octave_offset * 12)]
    
    def _note_to_midi(self, note_name: str) -> int:
        """Convert note name to MIDI."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        if note_name in notes:
            note_idx = notes.index(note_name)
        else:
            note_idx = 0  # Default to C if not found
        return 60 + note_idx  # Middle C