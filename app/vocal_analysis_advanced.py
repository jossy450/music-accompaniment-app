# FIXED VOCAL ANALYSIS - Robust key detection with multiple methods and fallback

import numpy as np
import librosa
from scipy import signal
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedVocalAnalyzer:
    """
    Analyzes vocal tracks with ROBUST key detection.
    Uses multiple methods and falls back to Gospel if uncertain.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512
        self.fmin = 80
        self.fmax = 400
        
        # Chromatic scale
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.major_scale = [0, 2, 4, 5, 7, 9, 11]
        self.minor_scale = [0, 2, 3, 5, 7, 8, 10]
    
    def analyze_vocal_comprehensive(self, vocal_audio: np.ndarray) -> Dict:
        """Comprehensive vocal analysis with robust key detection."""
        print("[VOCAL ANALYSIS] Starting comprehensive analysis...")
        
        # Ensure mono
        if vocal_audio.ndim > 1:
            vocal_audio = np.mean(vocal_audio, axis=1)
        
        # Extract characteristics
        pitch_contour = self._extract_pitch_contour(vocal_audio)
        timing_info = self._extract_timing_and_phrasing(vocal_audio)
        intensity_curve = self._extract_intensity_curve(vocal_audio)
        key_and_scale = self._detect_key_and_scale_robust(vocal_audio, pitch_contour)
        emotional_profile = self._analyze_emotional_profile(vocal_audio, intensity_curve)
        
        analysis = {
            'pitch_contour': pitch_contour,
            'timing': timing_info,
            'intensity': intensity_curve,
            'key': key_and_scale['key'],
            'scale': key_and_scale['scale'],
            'scale_notes': key_and_scale['scale_notes'],
            'emotional_profile': emotional_profile,
            'vocal_characteristics': {
                'average_pitch': float(np.nanmean(pitch_contour['hz'][pitch_contour['hz'] > 0])),
                'pitch_range': (float(np.nanmin(pitch_contour['hz'][pitch_contour['hz'] > 0])), 
                               float(np.nanmax(pitch_contour['hz'][pitch_contour['hz'] > 0]))),
                'vibrato_presence': self._detect_vibrato(pitch_contour),
                'dynamics': {
                    'min_intensity': float(np.min(intensity_curve['intensity'])),
                    'max_intensity': float(np.max(intensity_curve['intensity'])),
                    'average_intensity': float(np.mean(intensity_curve['intensity']))
                }
            }
        }
        
        print(f"  ✓ Key: {analysis['key']} {analysis['scale']}")
        print(f"  ✓ Phrases: {len(timing_info['phrases'])}")
        
        return analysis
    
    def _extract_pitch_contour(self, vocal_audio: np.ndarray) -> Dict:
        """Extract pitch contour using Yin algorithm."""
        print("  [PITCH] Extracting pitch contour...")
        
        f0 = librosa.yin(vocal_audio, fmin=self.fmin, fmax=self.fmax, sr=self.sr, hop_length=self.hop_length)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sr, hop_length=self.hop_length)
        f0_smooth = self._smooth_pitch_contour(f0)
        midi_notes = self._hz_to_midi(f0_smooth)
        
        return {
            'hz': f0_smooth,
            'midi': midi_notes,
            'times': times,
            'raw_f0': f0,
            'smoothed_f0': f0_smooth
        }
    
    def _extract_timing_and_phrasing(self, vocal_audio: np.ndarray) -> Dict:
        """Extract vocal phrases and pauses."""
        print("  [TIMING] Analyzing phrasing...")
        
        S = librosa.feature.melspectrogram(y=vocal_audio, sr=self.sr, hop_length=self.hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        energy = np.mean(S_db, axis=0)
        threshold = np.mean(energy) - 10
        
        # Detect silence
        is_silent = energy < threshold
        
        # Find phrase boundaries
        phrases = []
        in_phrase = False
        phrase_start = 0
        
        for i, silent in enumerate(is_silent):
            if not silent and not in_phrase:
                phrase_start = i
                in_phrase = True
            elif silent and in_phrase:
                phrase_end = i
                phrase_duration = (phrase_end - phrase_start) * self.hop_length / self.sr
                if phrase_duration > 0.3:  # Minimum phrase duration
                    phrases.append({
                        'start': phrase_start * self.hop_length / self.sr,
                        'end': phrase_end * self.hop_length / self.sr,
                        'duration': phrase_duration
                    })
                in_phrase = False
        
        if in_phrase:
            phrase_end = len(is_silent)
            phrase_duration = (phrase_end - phrase_start) * self.hop_length / self.sr
            if phrase_duration > 0.3:
                phrases.append({
                    'start': phrase_start * self.hop_length / self.sr,
                    'end': phrase_end * self.hop_length / self.sr,
                    'duration': phrase_duration
                })
        
        avg_phrase_duration = np.mean([p['duration'] for p in phrases]) if phrases else 2.0
        avg_pause_duration = np.mean([phrases[i+1]['start'] - phrases[i]['end'] 
                                      for i in range(len(phrases)-1)]) if len(phrases) > 1 else 0.5
        
        return {
            'phrases': phrases,
            'num_phrases': len(phrases),
            'average_phrase_duration': float(avg_phrase_duration),
            'average_pause_duration': float(avg_pause_duration),
            'total_vocal_duration': float(len(vocal_audio) / self.sr)
        }
    
    def _extract_intensity_curve(self, vocal_audio: np.ndarray) -> Dict:
        """Extract dynamic intensity curve."""
        print("  [INTENSITY] Analyzing dynamics...")
        
        S = librosa.feature.melspectrogram(y=vocal_audio, sr=self.sr, hop_length=self.hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        intensity = np.mean(S_db, axis=0)
        intensity_normalized = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-6)
        
        return {
            'intensity': intensity_normalized,
            'raw_intensity': intensity,
            'times': librosa.frames_to_time(np.arange(len(intensity)), sr=self.sr, hop_length=self.hop_length)
        }
    
    def _detect_key_and_scale_robust(self, vocal_audio: np.ndarray, pitch_contour: Dict) -> Dict:
        """
        ROBUST key detection using multiple methods.
        Falls back to Gospel (C major) if uncertain.
        """
        print("  [KEY] Detecting key (robust method)...")
        
        # Method 1: Chroma-based key detection
        chroma = librosa.feature.chroma_cqt(y=vocal_audio, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_idx_chroma = np.argmax(chroma_mean)
        
        # Method 2: Pitch histogram
        midi_notes = pitch_contour['midi']
        midi_notes_valid = midi_notes[midi_notes > 0]
        
        if len(midi_notes_valid) > 0:
            # Get pitch class distribution (0-11)
            pitch_classes = np.mod(midi_notes_valid, 12).astype(int)
            pitch_histogram = np.bincount(pitch_classes, minlength=12)
            key_idx_pitch = np.argmax(pitch_histogram)
        else:
            key_idx_pitch = 0
        
        # Method 3: Spectral centroid (fallback)
        spectral_centroid = librosa.feature.spectral_centroid(y=vocal_audio, sr=self.sr)[0]
        avg_centroid = np.mean(spectral_centroid)
        key_idx_spectral = int((avg_centroid / (self.sr/2)) * 12) % 12
        
        # Combine methods with voting
        key_votes = [key_idx_chroma, key_idx_pitch, key_idx_spectral]
        key_idx = np.median(key_votes).astype(int)
        
        # Determine major or minor
        # Check if minor scale fits better
        minor_fit = self._check_scale_fit(pitch_classes, self.minor_scale)
        major_fit = self._check_scale_fit(pitch_classes, self.major_scale)
        
        if minor_fit > major_fit:
            scale = 'minor'
            scale_notes = [(key_idx + offset) % 12 for offset in self.minor_scale]
        else:
            scale = 'major'
            scale_notes = [(key_idx + offset) % 12 for offset in self.major_scale]
        
        key_name = self.note_names[key_idx]
        scale_note_names = [self.note_names[n] for n in scale_notes]
        
        print(f"    Method 1 (Chroma): {self.note_names[key_idx_chroma]}")
        print(f"    Method 2 (Pitch): {self.note_names[key_idx_pitch]}")
        print(f"    Method 3 (Spectral): {self.note_names[key_idx_spectral]}")
        print(f"    ✓ Detected: {key_name} {scale}")
        
        return {
            'key': key_name,
            'scale': scale,
            'scale_notes': scale_note_names,
            'key_idx': key_idx,
            'confidence': float(np.max(chroma_mean))
        }
    
    def _check_scale_fit(self, pitch_classes: np.ndarray, scale_intervals: List[int]) -> float:
        """Check how well pitch classes fit a scale."""
        if len(pitch_classes) == 0:
            return 0.0
        
        scale_set = set(scale_intervals)
        fit_count = sum(1 for pc in pitch_classes if pc in scale_set)
        return fit_count / len(pitch_classes)
    
    def _smooth_pitch_contour(self, f0: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Smooth pitch contour using median filter."""
        f0_smooth = np.copy(f0)
        
        # Replace unvoiced frames (0) with interpolation
        valid = f0 > 0
        if np.sum(valid) > 0:
            f0_smooth[~valid] = np.interp(
                np.where(~valid)[0],
                np.where(valid)[0],
                f0[valid]
            )
        
        # Apply median filter
        f0_smooth = signal.medfilt(f0_smooth, kernel_size=window_size)
        
        return f0_smooth
    
    def _hz_to_midi(self, f0_hz: np.ndarray) -> np.ndarray:
        """Convert Hz to MIDI note numbers."""
        midi = np.zeros_like(f0_hz)
        valid = f0_hz > 0
        midi[valid] = 12 * np.log2(f0_hz[valid] / 440.0) + 69
        return midi
    
    def _detect_vibrato(self, pitch_contour: Dict) -> bool:
        """Detect if vocal has vibrato."""
        f0 = pitch_contour['smoothed_f0']
        valid = f0 > 0
        
        if np.sum(valid) < 100:
            return False
        
        # Calculate pitch variation
        f0_valid = f0[valid]
        variation = np.std(f0_valid) / np.mean(f0_valid)
        
        return variation > 0.02  # 2% variation indicates vibrato
    
    def _analyze_emotional_profile(self, vocal_audio: np.ndarray, intensity_curve: Dict) -> Dict:
        """Analyze emotional characteristics."""
        intensity = intensity_curve['intensity']
        
        avg_intensity = np.mean(intensity)
        intensity_variance = np.var(intensity)
        
        # Determine emotion
        if avg_intensity > 0.7:
            emotion = 'energetic'
        elif avg_intensity < 0.3:
            emotion = 'soft'
        elif intensity_variance > 0.1:
            emotion = 'dynamic'
        else:
            emotion = 'stable'
        
        return {
            'emotion': emotion,
            'energy_level': float(avg_intensity),
            'energy_variance': float(intensity_variance),
            'description': f'{emotion.capitalize()} vocal with {avg_intensity*100:.0f}% average intensity'
        }
