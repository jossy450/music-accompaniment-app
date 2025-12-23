# In app/mixing.py - PROFESSIONAL VERSION

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import librosa
import soundfile as sf
from typing import Tuple, Optional

class ProfessionalMixMaster:
    """
    Professional mixing and mastering engine that mimics real studio workflows.
    Includes multiband compression, EQ, limiting, and loudness normalization.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.target_loudness = -14.0  # LUFS (streaming standard)
        
    def _apply_eq(self, audio: np.ndarray, eq_type: str = 'presence') -> np.ndarray:
        """Apply parametric EQ to shape the tone."""
        # Design filters for different EQ profiles
        if eq_type == 'presence':  # Boost presence for vocals
            # Boost around 2-4kHz (presence peak)
            sos = signal.butter(4, [1500, 4000], btype='band', fs=self.sr, output='sos')
            filtered = signal.sosfilt(sos, audio)
            return audio + 0.3 * filtered  # 30% blend
        elif eq_type == 'warmth':  # Add warmth to band
            # Boost around 100-300Hz (warmth)
            sos = signal.butter(4, [80, 300], btype='band', fs=self.sr, output='sos')
            filtered = signal.sosfilt(sos, audio)
            return audio + 0.2 * filtered
        return audio
    
    def _apply_compression(self, audio: np.ndarray, ratio: float = 4.0, 
                          threshold: float = -20.0, attack_ms: float = 10.0,
                          release_ms: float = 100.0) -> np.ndarray:
        """Apply dynamic range compression (like a studio compressor)."""
        attack_samples = int(self.sr * attack_ms / 1000.0)
        release_samples = int(self.sr * release_ms / 1000.0)
        
        # Convert to dB
        eps = 1e-10
        audio_db = 20 * np.log10(np.abs(audio) + eps)
        
        # Calculate gain reduction
        gain_reduction = np.zeros_like(audio_db)
        for i in range(len(audio_db)):
            if audio_db[i] > threshold:
                gain_reduction[i] = (audio_db[i] - threshold) * (1 - 1/ratio)
        
        # Smooth gain reduction with attack/release envelope
        smoothed_gr = np.zeros_like(gain_reduction)
        for i in range(len(gain_reduction)):
            if i == 0:
                smoothed_gr[i] = gain_reduction[i]
            else:
                if gain_reduction[i] > smoothed_gr[i-1]:  # Attack
                    alpha = 1.0 / attack_samples if attack_samples > 0 else 1.0
                else:  # Release
                    alpha = 1.0 / release_samples if release_samples > 0 else 1.0
                smoothed_gr[i] = alpha * gain_reduction[i] + (1 - alpha) * smoothed_gr[i-1]
        
        # Apply gain reduction
        gain_linear = 10 ** (-smoothed_gr / 20.0)
        return audio * gain_linear
    
    def _apply_multiband_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply compression to different frequency bands independently."""
        # Split into 3 bands: low (0-250Hz), mid (250-4kHz), high (4kHz+)
        
        # Low band compression (tight, for control)
        sos_low = signal.butter(4, 250, btype='low', fs=self.sr, output='sos')
        low_band = signal.sosfilt(sos_low, audio)
        low_compressed = self._apply_compression(low_band, ratio=6.0, threshold=-18.0)
        
        # Mid band compression (moderate)
        sos_mid_low = signal.butter(4, 250, btype='high', fs=self.sr, output='sos')
        sos_mid_high = signal.butter(4, 4000, btype='low', fs=self.sr, output='sos')
        mid_band = signal.sosfilt(sos_mid_high, signal.sosfilt(sos_mid_low, audio))
        mid_compressed = self._apply_compression(mid_band, ratio=4.0, threshold=-20.0)
        
        # High band compression (light, preserve detail)
        sos_high = signal.butter(4, 4000, btype='high', fs=self.sr, output='sos')
        high_band = signal.sosfilt(sos_high, audio)
        high_compressed = self._apply_compression(high_band, ratio=2.0, threshold=-15.0)
        
        # Reconstruct
        return low_compressed + mid_compressed + high_compressed
    
    def _apply_limiter(self, audio: np.ndarray, threshold: float = -1.0) -> np.ndarray:
        """Apply a hard limiter to prevent clipping."""
        return self._apply_compression(audio, ratio=10.0, threshold=threshold, 
                                       attack_ms=2.0, release_ms=50.0)
    
    def _normalize_loudness(self, audio: np.ndarray, target_loudness: float = -14.0) -> np.ndarray:
        """Normalize audio to target loudness (LUFS)."""
        # Simple loudness approximation (RMS-based)
        rms = np.sqrt(np.mean(audio ** 2))
        current_loudness = 20 * np.log10(rms + 1e-10)
        gain_db = target_loudness - current_loudness
        gain_linear = 10 ** (gain_db / 20.0)
        return audio * gain_linear
    
    def _apply_stereo_widening(self, audio: np.ndarray, width: float = 0.5) -> np.ndarray:
        """Apply subtle stereo widening for a more spacious sound."""
        if len(audio.shape) == 1 or audio.shape[1] == 1:
            # Mono input - create stereo with slight delay
            audio_mono = audio if len(audio.shape) == 1 else audio[:, 0]
            delay_samples = int(0.02 * self.sr)  # 20ms delay
            delayed = np.concatenate([np.zeros(delay_samples), audio_mono[:-delay_samples]])
            
            left = audio_mono * (1 - width * 0.5) + delayed * width * 0.5
            right = audio_mono * (1 - width * 0.5) + audio_mono * width * 0.5
            return np.column_stack([left, right])
        return audio
    
    def mix_tracks(self, vocal: np.ndarray, accompaniment: np.ndarray, 
                   vocal_level_db: float = 0.0, music_level_db: float = -6.0,
                   mixing_profile: str = 'vocal_forward') -> np.ndarray:
        """
        Mix vocal and accompaniment with professional techniques.
        
        mixing_profile options:
        - 'vocal_forward': Vocal prominent, band supporting (default for most genres)
        - 'balanced': Equal presence of vocal and band
        """
        
        # Ensure both tracks are the same length
        max_len = max(len(vocal), len(accompaniment))
        vocal_padded = np.pad(vocal, (0, max_len - len(vocal)))
        accompaniment_padded = np.pad(accompaniment, (0, max_len - len(accompaniment)))
        
        # Apply EQ to shape tone
        vocal_eq = self._apply_eq(vocal_padded, eq_type='presence')
        accompaniment_eq = self._apply_eq(accompaniment_padded, eq_type='warmth')
        
        # Apply compression to individual tracks
        vocal_compressed = self._apply_compression(vocal_eq, ratio=3.0, threshold=-18.0)
        accompaniment_compressed = self._apply_compression(accompaniment_eq, ratio=2.5, threshold=-20.0)
        
        # Apply level adjustments
        vocal_gain = 10 ** (vocal_level_db / 20.0)
        music_gain = 10 ** (music_level_db / 20.0)
        
        vocal_leveled = vocal_compressed * vocal_gain
        accompaniment_leveled = accompaniment_compressed * music_gain
        
        # Mix based on profile
        if mixing_profile == 'vocal_forward':
            # Vocal at -3dB relative to band
            mixed = vocal_leveled + accompaniment_leveled * 0.7
        elif mixing_profile == 'balanced':
            # Equal mix
            mixed = (vocal_leveled + accompaniment_leveled) * 0.5
        else:
            mixed = vocal_leveled + accompaniment_leveled
        
        return mixed
    
    def master(self, audio: np.ndarray, apply_stereo_widening: bool = True) -> np.ndarray:
        """
        Apply professional mastering chain.
        Expert recommendation: Always apply for final output.
        """
        # 1. Multiband compression for tonal balance
        audio = self._apply_multiband_compression(audio)
        
        # 2. EQ for final tone shaping
        audio = self._apply_eq(audio, eq_type='presence')
        
        # 3. Limiter to prevent clipping
        audio = self._apply_limiter(audio, threshold=-1.0)
        
        # 4. Stereo widening for spaciousness (optional)
        if apply_stereo_widening:
            audio = self._apply_stereo_widening(audio, width=0.3)
        
        # 5. Loudness normalization to streaming standard
        audio = self._normalize_loudness(audio, target_loudness=self.target_loudness)
        
        # 6. Final limiter to catch any peaks
        audio = self._apply_limiter(audio, threshold=-0.5)
        
        return audio


class VocalTimeAligner:
    """
    Aligns the band accompaniment to the vocal's natural timing.
    This is the key to making the output sound like a real band playing together.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def detect_vocal_timing(self, vocal: np.ndarray) -> dict:
        """Analyze vocal to extract timing information."""
        # Detect onset times (when the vocal starts singing)
        S = librosa.feature.melspectrogram(y=vocal, sr=self.sr)
        onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max))
        onsets = librosa.onset.onset_detect(onset_env=onset_env, sr=self.sr, units='samples')
        
        # Detect vocal phrases (silent gaps)
        S_db = librosa.power_to_db(S, ref=np.max)
        energy = np.mean(S_db, axis=0)
        
        # Find silence regions
        silence_threshold = np.mean(energy) - 10
        is_silence = energy < silence_threshold
        
        return {
            'onsets': onsets,
            'energy': energy,
            'is_silence': is_silence,
            'duration': len(vocal) / self.sr
        }
    
    def align_accompaniment_to_vocal(self, accompaniment: np.ndarray, 
                                     vocal: np.ndarray) -> np.ndarray:
        """
        Time-stretch the accompaniment to match the vocal's timing.
        This ensures they stay synchronized throughout the song.
        """
        vocal_timing = self.detect_vocal_timing(vocal)
        
        # Calculate time-stretch factor
        vocal_duration = len(vocal) / self.sr
        accompaniment_duration = len(accompaniment) / self.sr
        
        if accompaniment_duration > 0:
            stretch_factor = vocal_duration / accompaniment_duration
            
            # Use librosa's time-stretching
            if 0.8 < stretch_factor < 1.2:  # Only stretch if within reasonable range
                accompaniment_stretched = librosa.effects.time_stretch(accompaniment, rate=stretch_factor)
            else:
                accompaniment_stretched = accompaniment
        else:
            accompaniment_stretched = accompaniment
        
        # Pad or trim to match vocal length
        if len(accompaniment_stretched) < len(vocal):
            accompaniment_stretched = np.pad(accompaniment_stretched, 
                                            (0, len(vocal) - len(accompaniment_stretched)))
        else:
            accompaniment_stretched = accompaniment_stretched[:len(vocal)]
        
        return accompaniment_stretched
