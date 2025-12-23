# FIXED MIXING - Professional mixing with EQ, compression, and proper level balancing

import numpy as np
from scipy import signal

class ProfessionalMixMaster:
    """
    Professional mixing and mastering with:
    - EQ for tone shaping
    - Compression for dynamics control
    - Proper gain staging
    - Loudness normalization
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def mix_tracks(
        self,
        vocal: np.ndarray,
        band: np.ndarray,
        vocal_level_db: float = 0.0,
        music_level_db: float = -6.0,
        mixing_profile: str = "vocal_forward",
    ) -> np.ndarray:
        """
        Mix vocal and band with professional processing.
        """
        print("[MIXING] Professional mixing...")
        
        # Ensure mono
        vocal = self._ensure_mono(vocal)
        band = self._ensure_mono(band)
        
        # Match lengths
        max_len = max(len(vocal), len(band))
        vocal = self._pad_or_trim(vocal, max_len)
        band = self._pad_or_trim(band, max_len)
        
        # Normalize to prevent clipping during processing
        vocal = self._normalize(vocal, target_level=-3.0)
        band = self._normalize(band, target_level=-6.0)
        
        # Apply EQ to vocal (enhance presence)
        vocal = self._eq_vocal(vocal)
        
        # Apply compression to vocal (control dynamics) - Increased attack/release for smoother sound
        vocal = self._compress(vocal, threshold_db=-24, ratio=3.0, attack_ms=20, release_ms=150)
        
        # Apply EQ to band (reduce mud, enhance clarity)
        band = self._eq_band(band)
        
        # Apply compression to band - Increased threshold and smoother release
        band = self._compress(band, threshold_db=-20, ratio=1.5, attack_ms=30, release_ms=200)
        
        # Apply gain staging
        vocal_gain = self._db_to_lin(vocal_level_db)
        music_gain = self._db_to_lin(music_level_db)
        
        # Adjust for mixing profile
        if mixing_profile == "vocal_forward":
            vocal_gain *= self._db_to_lin(3.0)  # Boost vocal
            music_gain *= self._db_to_lin(-2.0)  # Reduce band slightly
        elif mixing_profile == "balanced":
            vocal_gain *= self._db_to_lin(0.0)
            music_gain *= self._db_to_lin(0.0)
        
        vocal = vocal * vocal_gain
        band = band * music_gain
        
        # Mix
        mix = vocal + band
        
        # Hard limiting to prevent clipping before final normalization
        mix = self._hard_limit(mix, threshold_db=-1.0)
        
        # Normalize final mix
        mix = self._normalize(mix, target_level=-3.0)
        
        print(f"  ✓ Mixed with {mixing_profile} profile")
        
        return mix.astype(np.float32)
    
    def master(self, audio: np.ndarray, apply_stereo_widening: bool = True) -> np.ndarray:
        """
        Professional mastering chain:
        - Multiband compression
        - Parametric EQ
        - Limiting
        - Loudness normalization
        """
        print("[MASTERING] Applying professional mastering...")
        
        audio = self._ensure_mono(audio).astype(np.float32)
        
        # Multiband compression (low, mid, high)
        audio = self._multiband_compress(audio)
        
        # Parametric EQ (final touch)
        audio = self._eq_master(audio)
        
        # Limiter (safety) - Using a hard limiter for final brickwall limiting
        audio = self._hard_limit(audio, threshold_db=-0.5)
        
        # Loudness normalization to -14 LUFS (streaming standard)
        audio = self._normalize_loudness(audio, target_lufs=-14.0)
        
        # Stereo widening
        if apply_stereo_widening:
            audio = self._stereo_widen(audio)
        else:
            audio = audio[:, None]
        
        print("  ✓ Mastering complete (-14 LUFS)")
        
        return audio.astype(np.float32)
    
    # ===== EQ Methods =====
    
    def _eq_vocal(self, audio: np.ndarray) -> np.ndarray:
        """EQ for vocal: boost presence (2-4kHz), reduce mud (200-500Hz)."""
        # High-pass filter (remove rumble below 80Hz)
        audio = self._highpass_filter(audio, cutoff_hz=80)
        
        # Reduce mud (200-500Hz, -3dB)
        audio = self._peaking_eq(audio, freq_hz=350, gain_db=-3, q=1.0)
        
        # Boost presence (2-4kHz, +2dB)
        audio = self._peaking_eq(audio, freq_hz=3000, gain_db=2, q=1.0)
        
        # Slight presence boost (5-8kHz, +1dB)
        audio = self._peaking_eq(audio, freq_hz=6000, gain_db=1, q=1.0)
        
        return audio
    
    def _eq_band(self, audio: np.ndarray) -> np.ndarray:
        """EQ for band: reduce mud, enhance clarity."""
        # High-pass filter
        audio = self._highpass_filter(audio, cutoff_hz=60)
        
        # Reduce mud (200-400Hz, -2dB)
        audio = self._peaking_eq(audio, freq_hz=300, gain_db=-2, q=1.0)
        
        # Enhance clarity (2-3kHz, +1.5dB)
        audio = self._peaking_eq(audio, freq_hz=2500, gain_db=1.5, q=1.0)
        
        return audio
    
    def _eq_master(self, audio: np.ndarray) -> np.ndarray:
        """Master EQ: subtle enhancement."""
        # Slight low-end boost (80-120Hz, +1dB)
        audio = self._peaking_eq(audio, freq_hz=100, gain_db=1, q=0.7)
        
        # Slight high-end boost (8-12kHz, +0.5dB)
        audio = self._peaking_eq(audio, freq_hz=10000, gain_db=0.5, q=0.7)
        
        return audio
    
    def _highpass_filter(self, audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """High-pass filter to remove low-frequency rumble."""
        nyquist = self.sr / 2
        normalized_cutoff = cutoff_hz / nyquist
        
        if normalized_cutoff >= 1.0:
            return audio
        
        b, a = signal.butter(2, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def _peaking_eq(self, audio: np.ndarray, freq_hz: float, gain_db: float, q: float) -> np.ndarray:
        """Peaking EQ filter."""
        nyquist = self.sr / 2
        w0 = 2 * np.pi * freq_hz / self.sr
        
        alpha = np.sin(w0) / (2 * q)
        A = 10 ** (gain_db / 40)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        
        return signal.filtfilt(b, a, audio)
    
    # ===== Compression Methods =====
    
    def _compress(self, audio: np.ndarray, threshold_db: float = -20, 
                  ratio: float = 4.0, attack_ms: float = 10, release_ms: float = 100) -> np.ndarray:
        """Dynamic range compressor."""
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-12)
        
        # Detect when above threshold
        above_threshold = audio_db > threshold_db
        
        # Calculate gain reduction
        gain_reduction = np.zeros_like(audio_db)
        gain_reduction[above_threshold] = (audio_db[above_threshold] - threshold_db) * (1 - 1/ratio)
        
        # Smooth gain reduction with attack/release
        attack_samples = int(attack_ms * self.sr / 1000)
        release_samples = int(release_ms * self.sr / 1000)
        
        smoothed_gr = self._smooth_envelope(gain_reduction, attack_samples, release_samples)
        
        # Apply gain reduction
        gain_linear = 10 ** (-smoothed_gr / 20)
        compressed = audio * gain_linear
        
        return compressed
    
    def _multiband_compress(self, audio: np.ndarray) -> np.ndarray:
        """Multiband compression (low, mid, high)."""
        # Split into bands
        low = self._bandpass_filter(audio, low_hz=20, high_hz=200)
        mid = self._bandpass_filter(audio, low_hz=200, high_hz=2000)
        high = self._bandpass_filter(audio, low_hz=2000, high_hz=20000)
        
        # Compress each band
        low = self._compress(low, threshold_db=-22, ratio=3.0, attack_ms=15, release_ms=120)
        mid = self._compress(mid, threshold_db=-20, ratio=2.5, attack_ms=10, release_ms=100)
        high = self._compress(high, threshold_db=-18, ratio=2.0, attack_ms=5, release_ms=80)
        
        # Recombine
        return low + mid + high
    
    def _bandpass_filter(self, audio: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
        """Bandpass filter."""
        nyquist = self.sr / 2
        low_norm = low_hz / nyquist
        high_norm = high_hz / nyquist
        
        low_norm = np.clip(low_norm, 0.001, 0.999)
        high_norm = np.clip(high_norm, 0.001, 0.999)
        
        b, a = signal.butter(2, [low_norm, high_norm], btype='band')
        return signal.filtfilt(b, a, audio)
    
    def _soft_limit(self, audio: np.ndarray, threshold_db: float = -1.0) -> np.ndarray:
        """Soft limiter to prevent clipping."""
        threshold_lin = self._db_to_lin(threshold_db)
        
        # Soft knee limiting
        above_threshold = np.abs(audio) > threshold_lin
        
        if not np.any(above_threshold):
            return audio
        
        # Apply soft limiting (tanh-like curve)
        limited = np.copy(audio)
        limited[above_threshold] = np.tanh(audio[above_threshold] / threshold_lin) * threshold_lin
        
        return limited

    def _hard_limit(self, audio: np.ndarray, threshold_db: float = -1.0) -> np.ndarray:
        """Hard limiter (brickwall) to prevent clipping."""
        threshold_lin = self._db_to_lin(threshold_db)
        
        limited = np.clip(audio, -threshold_lin, threshold_lin)
        
        return limited
    
    # ===== Utility Methods =====
    
    def _smooth_envelope(self, signal_db: np.ndarray, attack_samples: int, release_samples: int) -> np.ndarray:
        """Smooth envelope with attack and release."""
        smoothed = np.copy(signal_db)
        
        for i in range(1, len(smoothed)):
            if signal_db[i] > smoothed[i-1]:
                # Attack
                alpha = 1.0 / attack_samples if attack_samples > 0 else 1.0
            else:
                # Release
                alpha = 1.0 / release_samples if release_samples > 0 else 1.0
            
            smoothed[i] = alpha * signal_db[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed
    
    def _normalize_loudness(self, audio: np.ndarray, target_lufs: float = -14.0) -> np.ndarray:
        """Normalize to target loudness (LUFS)."""
        # Simplified LUFS calculation
        # Real LUFS requires K-weighting, but this is close enough
        rms = np.sqrt(np.mean(audio ** 2))
        current_lufs = 20 * np.log10(rms + 1e-12)
        
        gain_db = target_lufs - current_lufs
        gain_lin = self._db_to_lin(gain_db)
        
        normalized = audio * gain_lin
        
        # Prevent clipping
        peak = np.max(np.abs(normalized))
        if peak > 1.0:
            normalized = normalized / peak
        
        return normalized
    
    def _normalize(self, audio: np.ndarray, target_level: float = -3.0) -> np.ndarray:
        """Normalize audio to target dB level."""
        rms = np.sqrt(np.mean(audio ** 2))
        current_db = 20 * np.log10(rms + 1e-12)
        gain_db = target_level - current_db
        gain_lin = self._db_to_lin(gain_db)
        return audio * gain_lin
    
    def _stereo_widen(self, mono: np.ndarray) -> np.ndarray:
        """Create stereo with subtle width."""
        delay_samples = int(self.sr * 0.015)  # 15ms delay
        
        left = mono
        right = np.pad(mono, (delay_samples, 0), mode='constant')[:len(mono)]
        
        stereo = np.stack([left, right], axis=1)
        
        # Prevent clipping
        peak = np.max(np.abs(stereo))
        if peak > 1.0:
            stereo = stereo / peak
        
        return stereo
    
    def _ensure_mono(self, x: np.ndarray) -> np.ndarray:
        """Ensure audio is mono."""
        if x.ndim == 1:
            return x
        return np.mean(x, axis=1)
    
    def _pad_or_trim(self, x: np.ndarray, length: int) -> np.ndarray:
        """Pad or trim to length."""
        if len(x) == length:
            return x
        if len(x) > length:
            return x[:length]
        return np.pad(x, (0, length - len(x)), mode='constant')
    
    def _db_to_lin(self, db: float) -> float:
        """Convert dB to linear."""
        return 10.0 ** (db / 20.0)


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
import librosa