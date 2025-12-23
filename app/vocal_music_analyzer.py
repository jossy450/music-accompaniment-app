# In app/vocal_musical_analyzer.py

import numpy as np
import librosa
import scipy.signal
from typing import Dict, List, Tuple

class VocalMusicalAnalyzer:
    """Analyzes vocal audio to extract musical characteristics for arrangement"""
    
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
    def analyze_vocal(self, audio_path: str) -> Dict:
        """
        Analyze vocal audio file and extract:
        - Key and scale
        - Tempo
        - Phrase segmentation
        - Intensity/energy
        """
        
        print(f"[MUSICAL ANALYZER] Loading audio from {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        
        analysis = {
            'key': 'C',
            'scale': 'major',
            'tempo': 120.0,
            'timing': {
                'phrases': [],
                'average_phrase_duration': 0.0
            },
            'intensity': {
                'intensity': np.array([])
            }
        }
        
        try:
            # 1. Detect key and scale
            print("[MUSICAL ANALYZER] Detecting key...")
            key, scale = self.detect_key(y)
            analysis['key'] = key
            analysis['scale'] = scale
            
            # 2. Detect tempo
            print("[MUSICAL ANALYZER] Detecting tempo...")
            tempo = self.detect_tempo(y)
            analysis['tempo'] = tempo
            
            # 3. Detect phrases (vocal segments)
            print("[MUSICAL ANALYZER] Detecting phrases...")
            phrases = self.detect_phrases(y)
            analysis['timing']['phrases'] = phrases
            
            if phrases:
                durations = [p['duration'] for p in phrases]
                analysis['timing']['average_phrase_duration'] = float(np.mean(durations))
            
            # 4. Calculate intensity/energy
            print("[MUSICAL ANALYZER] Calculating intensity...")
            intensity = self.calculate_intensity(y)
            analysis['intensity']['intensity'] = intensity
            
            print(f"[MUSICAL ANALYZER] Analysis complete: Key={key} {scale}, Tempo={tempo:.1f} BPM")
            
        except Exception as e:
            print(f"[MUSICAL ANALYZER] Error in analysis: {e}")
            # Return fallback analysis
            analysis = self.get_fallback_analysis(y)
        
        return analysis
    
    def detect_key(self, y: np.ndarray) -> Tuple[str, str]:
        """Detect musical key using chroma features"""
        try:
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr)
            
            # Average over time
            chroma_mean = np.mean(chroma, axis=1)
            
            # Use Krumhansl-Schmuckler key-finding algorithm approximation
            # Major and minor key profiles
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 
                                      5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                                      4.75, 3.98, 2.69, 3.34, 3.17])
            
            # Normalize profiles
            major_profile = major_profile / np.sum(major_profile)
            minor_profile = minor_profile / np.sum(minor_profile)
            
            # Find best matching key
            best_correlation = -1
            best_key = 0  # C
            best_scale = 'major'
            
            for shift in range(12):
                # Shift profiles
                major_shifted = np.roll(major_profile, shift)
                minor_shifted = np.roll(minor_profile, shift)
                
                # Correlate with chroma
                major_corr = np.corrcoef(chroma_mean, major_shifted)[0, 1]
                minor_corr = np.corrcoef(chroma_mean, minor_shifted)[0, 1]
                
                if major_corr > best_correlation:
                    best_correlation = major_corr
                    best_key = shift
                    best_scale = 'major'
                
                if minor_corr > best_correlation:
                    best_correlation = minor_corr
                    best_key = shift
                    best_scale = 'minor'
            
            # Convert key index to note name
            key_name = self.note_names[best_key]
            
            return key_name, best_scale
            
        except Exception as e:
            print(f"[MUSICAL ANALYZER] Key detection failed: {e}")
            return 'C', 'major'
    
    def detect_tempo(self, y: np.ndarray) -> float:
        """Detect tempo using onset detection"""
        try:
            # Use onset detection to find tempo
            onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
            
            if isinstance(tempo, np.ndarray) and len(tempo) > 0:
                return float(tempo[0])
            else:
                # Fallback: estimate tempo from energy peaks
                hop_length = self.hop_length
                rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
                
                # Find peaks in RMS
                peaks, _ = scipy.signal.find_peaks(rms, distance=20)
                
                if len(peaks) > 1:
                    # Calculate average time between peaks
                    peak_times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=hop_length)
                    intervals = np.diff(peak_times)
                    avg_interval = np.mean(intervals)
                    
                    if avg_interval > 0:
                        tempo = 60 / avg_interval
                        # Round to nearest typical tempo
                        typical_tempos = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
                        tempo = min(typical_tempos, key=lambda x: abs(x - tempo))
                        return float(tempo)
            
            return 120.0  # Default tempo
            
        except Exception as e:
            print(f"[MUSICAL ANALYZER] Tempo detection failed: {e}")
            return 120.0
    
    def detect_phrases(self, y: np.ndarray) -> List[Dict]:
        """Detect vocal phrases using energy and silence detection"""
        try:
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            
            # Normalize RMS
            rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-10)
            
            # Find energy peaks (potential phrase starts)
            peaks, properties = scipy.signal.find_peaks(
                rms_normalized, 
                height=0.3,  # Minimum energy threshold
                distance=10,  # Minimum frames between peaks
                prominence=0.2  # Minimum prominence
            )
            
            # Convert peaks to time
            peak_times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
            
            if len(peak_times) == 0:
                # No clear peaks found, use whole audio as one phrase
                duration = len(y) / self.sr
                return [{'start': 0.0, 'duration': duration, 'confidence': 0.8}]
            
            # Create phrases from peaks
            phrases = []
            total_duration = len(y) / self.sr
            
            # First phrase: from start to first peak
            if peak_times[0] > 0.5:  # If first peak is not at the very beginning
                phrases.append({
                    'start': 0.0,
                    'duration': peak_times[0],
                    'confidence': 0.7
                })
            
            # Middle phrases: between peaks
            for i in range(len(peak_times) - 1):
                start = peak_times[i]
                end = peak_times[i + 1]
                duration = end - start
                
                if duration > 0.5:  # Only include phrases longer than 0.5 seconds
                    phrases.append({
                        'start': start,
                        'duration': duration,
                        'confidence': 0.9
                    })
            
            # Last phrase: from last peak to end
            if peak_times[-1] < total_duration - 0.5:
                phrases.append({
                    'start': peak_times[-1],
                    'duration': total_duration - peak_times[-1],
                    'confidence': 0.7
                })
            
            # If no phrases were created (shouldn't happen), create one
            if not phrases:
                phrases.append({
                    'start': 0.0,
                    'duration': total_duration,
                    'confidence': 0.8
                })
            
            # Ensure phrases don't overlap and are in order
            phrases = sorted(phrases, key=lambda x: x['start'])
            
            # Adjust durations to prevent overlap
            for i in range(len(phrases) - 1):
                current_end = phrases[i]['start'] + phrases[i]['duration']
                next_start = phrases[i + 1]['start']
                
                if current_end > next_start:
                    # Adjust current phrase duration
                    phrases[i]['duration'] = next_start - phrases[i]['start']
            
            # Remove very short phrases (< 0.3 seconds)
            phrases = [p for p in phrases if p['duration'] >= 0.3]
            
            return phrases
            
        except Exception as e:
            print(f"[MUSICAL ANALYZER] Phrase detection failed: {e}")
            # Return single phrase covering entire audio
            duration = len(y) / self.sr
            return [{'start': 0.0, 'duration': duration, 'confidence': 0.5}]
    
    def calculate_intensity(self, y: np.ndarray) -> np.ndarray:
        """Calculate intensity/energy over time"""
        try:
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            
            # Normalize to 0-1 range
            if np.max(rms) > np.min(rms):
                intensity = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
            else:
                intensity = np.ones_like(rms) * 0.5
            
            return intensity
            
        except Exception as e:
            print(f"[MUSICAL ANALYZER] Intensity calculation failed: {e}")
            # Return default intensity
            num_frames = len(y) // self.hop_length + 1
            return np.ones(num_frames) * 0.7
    
    def get_fallback_analysis(self, y: np.ndarray) -> Dict:
        """Provide fallback analysis when automatic analysis fails"""
        duration = len(y) / self.sr
        
        # Create phrases based on duration
        phrase_duration = 4.0  # Default phrase length
        num_phrases = max(1, int(duration / phrase_duration))
        
        phrases = []
        for i in range(num_phrases):
            start = i * phrase_duration
            actual_duration = min(phrase_duration, duration - start)
            
            phrases.append({
                'start': start,
                'duration': actual_duration,
                'confidence': 0.5
            })
        
        # Calculate intensity
        intensity = self.calculate_intensity(y)
        
        return {
            'key': 'C',
            'scale': 'major',
            'tempo': 120.0,
            'timing': {
                'phrases': phrases,
                'average_phrase_duration': float(np.mean([p['duration'] for p in phrases]))
            },
            'intensity': {
                'intensity': intensity
            }
        }