# In app/arrangement_selector.py

import numpy as np
from typing import Dict, List

class AdvancedArrangementSelector:
    """Selects optimal arrangement based on vocal analysis."""
    
    def __init__(self):
        self.genre_instruments = {
            'pop': ['drums', 'bass', 'piano', 'strings'],
            'rock': ['drums', 'bass', 'guitar', 'organ'],
            'r&b': ['drums', 'bass', 'piano', 'strings'],
            'jazz': ['drums', 'bass', 'piano', 'brass'],
            'gospel': ['drums', 'bass', 'organ', 'strings'],
            'ballad': ['piano', 'strings', 'bass'],
            'afrobeat': ['drums', 'percussion', 'bass', 'guitar'],
            'reggae': ['drums', 'bass', 'guitar', 'organ'],
        }
    
    def select_best_arrangement(self, vocal_analysis: Dict, audio: np.ndarray, sr: int) -> Dict:
        """Select arrangement based on vocal analysis."""
        genre = vocal_analysis.get('detected_genre', 'pop')
        tempo = vocal_analysis.get('tempo', 120)
        
        # Determine complexity
        if tempo < 90:
            complexity = 'simple'
        elif tempo < 120:
            complexity = 'medium'
        else:
            complexity = 'complex'
        
        # Get instruments for genre
        instruments = self.genre_instruments.get(genre, self.genre_instruments['pop'])
        
        return {
            'complexity': complexity,
            'instruments': instruments,
            'intensity': 'high' if tempo > 120 else 'medium',
            'reasoning': f"Selected {complexity} arrangement with {', '.join(instruments)} for {genre} at {tempo:.0f} BPM"
        }
