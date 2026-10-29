# In app/vocal_analyzer.py

import numpy as np
import librosa
from typing import Dict

class AdvancedVocalGenreAnalyzer:
    """Analyzes vocal characteristics to determine genre and provide recommendations."""
    
    def __init__(self):
        self.genre_templates = {
            'pop': {'tempo_range': (90, 130), 'energy': 0.6, 'brightness': 0.7},
            'rock': {'tempo_range': (100, 140), 'energy': 0.8, 'brightness': 0.5},
            'r&b': {'tempo_range': (80, 110), 'energy': 0.5, 'brightness': 0.6},
            'jazz': {'tempo_range': (90, 180), 'energy': 0.4, 'brightness': 0.8},
            'gospel': {'tempo_range': (100, 130), 'energy': 0.9, 'brightness': 0.8},
            'ballad': {'tempo_range': (60, 90), 'energy': 0.3, 'brightness': 0.7},
            'afrobeat': {'tempo_range': (100, 130), 'energy': 0.85, 'brightness': 0.75},
            'reggae': {'tempo_range': (70, 100), 'energy': 0.5, 'brightness': 0.6},
        }
    
    def _extract_advanced_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract advanced audio features."""
        # Tempo estimation
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Energy
        rms = librosa.feature.rms(y=audio)[0]
        energy = np.mean(rms)
        
        # Spectral centroid (brightness)
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        brightness = np.mean(spec_cent) / (sr / 2)  # Normalize to 0-1
        
        # Zero crossing rate (articulation)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        articulation = np.mean(zcr)
        
        return {
            'tempo': float(np.atleast_1d(tempo)[0]),  # librosa >= 0.10 returns a 1-element array
            'energy': float(np.clip(energy, 0, 1)),
            'brightness': float(np.clip(brightness, 0, 1)),
            'articulation': float(articulation),
        }
    
    def detect_genre_from_vocal(self, audio: np.ndarray, sr: int) -> Dict:
        """Detect genre based on vocal characteristics."""
        features = self._extract_advanced_features(audio, sr)
        
        # Score each genre
        genre_scores = {}
        for genre, template in self.genre_templates.items():
            tempo_score = 1 - abs(features['tempo'] - np.mean(template['tempo_range'])) / 100
            energy_score = 1 - abs(features['energy'] - template['energy'])
            brightness_score = 1 - abs(features['brightness'] - template['brightness'])
            
            overall_score = (tempo_score + energy_score + brightness_score) / 3
            genre_scores[genre] = float(np.clip(overall_score, 0, 1))
        
        # Find best match
        best_genre = max(genre_scores, key=genre_scores.get)
        
        return {
            'detected_genre': best_genre,
            'genre_probabilities': genre_scores,
            'feature_analysis': features,
            'recommendation': {
                'genre': best_genre,
                'key_suggestion': 'C',
                'reason': f"Detected {best_genre} characteristics based on vocal performance."
            }
        }
