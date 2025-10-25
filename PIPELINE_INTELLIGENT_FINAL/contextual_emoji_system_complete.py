# -*- coding: utf-8 -*-
"""
SystÃ¨me d'emojis contextuels COMPLET et PRÃ‰CIS pour les sous-titres Hormozi
Mapping prÃ©cis en anglais pour tous les concepts avec dÃ©tection intelligente
"""

import random
from typing import Dict, List, Optional, Tuple
import re

class ContextualEmojiSystemComplete:
    """SystÃ¨me d'emojis intelligents et contextuels COMPLET avec mapping prÃ©cis"""
    
    def __init__(self):
        # ðŸŽ¯ MAPPING SÃ‰MANTIQUE COMPLET ET PRÃ‰CIS (500+ EMOJIS)
        self.semantic_mapping = {
            # ðŸ§  COGNITIVE & LEARNING (NOUVEAU - COMPLET)
            'brain': {
                'positive': ['ðŸ§ ', 'ðŸ’­', 'ðŸ’¡', 'ðŸŽ¯', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ§ ', 'ðŸ’­', 'ðŸ’¡', 'ðŸŽ¯', 'ðŸ“', 'ðŸ“š']
            },
            'thinking': {
                'positive': ['ðŸ§ ', 'ðŸ’­', 'ðŸ’¡', 'ðŸŽ¯', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ§ ', 'ðŸ’­', 'ðŸ’¡', 'ðŸŽ¯', 'ðŸ“', 'ðŸ“š']
            },
            'attention': {
                'positive': ['ðŸ‘ï¸', 'ðŸŽ¯', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ‘ï¸', 'ðŸŽ¯', 'ðŸ’¡', 'ðŸ“', 'ðŸ“š']
            },
            'concentration': {
                'positive': ['ðŸ‘ï¸', 'ðŸŽ¯', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ‘ï¸', 'ðŸŽ¯', 'ðŸ’¡', 'ðŸ“', 'ðŸ“š']
            },
            'learning': {
                'positive': ['ðŸ“š', 'âœï¸', 'ðŸŽ“', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ“š', 'âœï¸', 'ðŸŽ“', 'ðŸ’¡', 'ðŸ“']
            },
            'studying': {
                'positive': ['ðŸ“š', 'âœï¸', 'ðŸŽ“', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ“š', 'âœï¸', 'ðŸŽ“', 'ðŸ’¡', 'ðŸ“']
            },
            'reading': {
                'positive': ['ðŸ“š', 'ðŸ“–', 'ðŸ‘“', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ“š', 'ðŸ“–', 'ðŸ‘“', 'ðŸ’¡', 'ðŸ“']
            },
            'math': {
                'positive': ['ðŸ”¢', 'ðŸ“', 'ðŸ“', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ”¢', 'ðŸ“', 'ðŸ“', 'ðŸ’¡', 'ðŸ“']
            },
            'workout': {
                'positive': ['ðŸ’ª', 'ðŸ‹ï¸', 'ðŸš´', 'ðŸƒ', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ’ª', 'ðŸ‹ï¸', 'ðŸš´', 'ðŸƒ', 'ðŸ”¥']
            },
            'exercise': {
                'positive': ['ðŸ’ª', 'ðŸ‹ï¸', 'ðŸš´', 'ðŸƒ', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ’ª', 'ðŸ‹ï¸', 'ðŸš´', 'ðŸƒ', 'ðŸ”¥']
            },
            'physical': {
                'positive': ['ðŸ’ª', 'ðŸ‹ï¸', 'ðŸš´', 'ðŸƒ', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ’ª', 'ðŸ‹ï¸', 'ðŸš´', 'ðŸƒ', 'ðŸ”¥']
            },
            'challenging': {
                'positive': ['ðŸ’ª', 'ðŸ‹ï¸', 'ðŸŽ¯', 'ðŸš€', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ’ª', 'ðŸ‹ï¸', 'ðŸŽ¯', 'ðŸ”¥']
            },
            'difficult': {
                'positive': ['ðŸ’ª', 'ðŸ‹ï¸', 'ðŸŽ¯', 'ðŸš€', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ’ª', 'ðŸ‹ï¸', 'ðŸŽ¯', 'ðŸ”¥']
            },
            
            # ðŸ§¬ NEUROSCIENCE & SCIENCE (NOUVEAU - COMPLET)
            'acetylcholine': {
                'positive': ['ðŸ§ ', 'ðŸ§¬', 'ðŸ”¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ§ ', 'ðŸ§¬', 'ðŸ”¬', 'ðŸ’¡', 'ðŸ“']
            },
            'norepinephrine': {
                'positive': ['ðŸ§ ', 'ðŸ§¬', 'ðŸ”¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ§ ', 'ðŸ§¬', 'ðŸ”¬', 'ðŸ’¡', 'ðŸ“']
            },
            'synapses': {
                'positive': ['ðŸ§ ', 'ðŸ§¬', 'ðŸ”¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ§ ', 'ðŸ§¬', 'ðŸ”¬', 'ðŸ’¡', 'ðŸ“']
            },
            'plasticity': {
                'positive': ['ðŸ§ ', 'ðŸ§¬', 'ðŸ”¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ§ ', 'ðŸ§¬', 'ðŸ”¬', 'ðŸ’¡', 'ðŸ“']
            },
            'neuroscience': {
                'positive': ['ðŸ§ ', 'ðŸ§¬', 'ðŸ”¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ§ ', 'ðŸ§¬', 'ðŸ”¬', 'ðŸ’¡', 'ðŸ“']
            },
            'research': {
                'positive': ['ðŸ”¬', 'ðŸ§¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ”¬', 'ðŸ§¬', 'ðŸ’¡', 'ðŸ“', 'ðŸ“š']
            },
            'studies': {
                'positive': ['ðŸ”¬', 'ðŸ§¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ”¬', 'ðŸ§¬', 'ðŸ’¡', 'ðŸ“', 'ðŸ“š']
            },
            'science': {
                'positive': ['ðŸ”¬', 'ðŸ§¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ”¬', 'ðŸ§¬', 'ðŸ’¡', 'ðŸ“', 'ðŸ“š']
            },
            
            # ðŸŽ“ UNIVERSITY & ACADEMIC (NOUVEAU - COMPLET)
            'stanford': {
                'positive': ['ðŸŽ“', 'ðŸ›ï¸', 'ðŸ“š', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸŽ“', 'ðŸ›ï¸', 'ðŸ“š', 'ðŸ’¡', 'ðŸ“']
            },
            'university': {
                'positive': ['ðŸŽ“', 'ðŸ›ï¸', 'ðŸ“š', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸŽ“', 'ðŸ›ï¸', 'ðŸ“š', 'ðŸ’¡', 'ðŸ“']
            },
            'college': {
                'positive': ['ðŸŽ“', 'ðŸ›ï¸', 'ðŸ“š', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸŽ“', 'ðŸ›ï¸', 'ðŸ“š', 'ðŸ’¡', 'ðŸ“']
            },
            'academic': {
                'positive': ['ðŸŽ“', 'ðŸ›ï¸', 'ðŸ“š', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸŽ“', 'ðŸ›ï¸', 'ðŸ“š', 'ðŸ’¡', 'ðŸ“']
            },
            'education': {
                'positive': ['ðŸŽ“', 'ðŸ›ï¸', 'ðŸ“š', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸŽ“', 'ðŸ›ï¸', 'ðŸ“š', 'ðŸ’¡', 'ðŸ“']
            },
            
            # ðŸ‘©â€ðŸŽ“ FEMALE LEARNING & FRUSTRATION (NOUVEAU - COMPLET)
            'she': {
                'positive': ['ðŸ‘©â€ðŸŽ“', 'ðŸ‘©â€ðŸ’¼', 'ðŸ‘©â€ðŸ”¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ‘©â€ðŸŽ“', 'ðŸ‘©â€ðŸ’¼', 'ðŸ‘©â€ðŸ”¬', 'ðŸ’¡', 'ðŸ“']
            },
            'her': {
                'positive': ['ðŸ‘©â€ðŸŽ“', 'ðŸ‘©â€ðŸ’¼', 'ðŸ‘©â€ðŸ”¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ‘©â€ðŸŽ“', 'ðŸ‘©â€ðŸ’¼', 'ðŸ‘©â€ðŸ”¬', 'ðŸ’¡', 'ðŸ“']
            },
            'woman': {
                'positive': ['ðŸ‘©â€ðŸŽ“', 'ðŸ‘©â€ðŸ’¼', 'ðŸ‘©â€ðŸ”¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ‘©â€ðŸŽ“', 'ðŸ‘©â€ðŸ’¼', 'ðŸ‘©â€ðŸ”¬', 'ðŸ’¡', 'ðŸ“']
            },
            'female': {
                'positive': ['ðŸ‘©â€ðŸŽ“', 'ðŸ‘©â€ðŸ’¼', 'ðŸ‘©â€ðŸ”¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ‘©â€ðŸŽ“', 'ðŸ‘©â€ðŸ’¼', 'ðŸ‘©â€ðŸ”¬', 'ðŸ’¡', 'ðŸ“']
            },
            'girl': {
                'positive': ['ðŸ‘©â€ðŸŽ“', 'ðŸ‘©â€ðŸ’¼', 'ðŸ‘©â€ðŸ”¬', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ‘©â€ðŸŽ“', 'ðŸ‘©â€ðŸ’¼', 'ðŸ‘©â€ðŸ”¬', 'ðŸ’¡', 'ðŸ“']
            },
            'frustrating': {
                'positive': ['ðŸ˜¤', 'ðŸ’ª', 'ðŸš€', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜¤', 'ðŸ˜ ', 'ðŸ˜¡', 'ðŸ¤¬', 'ðŸ’”', 'ðŸ˜°', 'ðŸ˜¨', 'ðŸ˜±'],
                'neutral': ['ðŸ˜¤', 'ðŸ˜ ', 'ðŸ˜¡', 'ðŸ¤¬']
            },
            'frustration': {
                'positive': ['ðŸ˜¤', 'ðŸ’ª', 'ðŸš€', 'ðŸ†', 'â­', 'ðŸŒŸ'],
                'negative': ['ðŸ˜¤', 'ðŸ˜ ', 'ðŸ˜¡', 'ðŸ¤¬', 'ðŸ’”', 'ðŸ˜°', 'ðŸ˜¨', 'ðŸ˜±'],
                'neutral': ['ðŸ˜¤', 'ðŸ˜ ', 'ðŸ˜¡', 'ðŸ¤¬']
            },
            
            # ðŸš¨ SERVICES D'URGENCE (EXISTANT - AMÃ‰LIORÃ‰)
            'emergency': {
                'positive': ['ðŸš¨', 'ðŸš‘', 'ðŸš’', 'ðŸ‘¨â€ðŸš’', 'ðŸ‘©â€ðŸš’', 'ðŸ‘®â€â™‚ï¸', 'ðŸ‘®â€â™€ï¸', 'ðŸš“', 'ðŸ’™', 'ðŸ†˜'],
                'negative': ['ðŸš¨', 'ðŸš‘', 'ðŸš’', 'ðŸ’”', 'ðŸ˜°', 'ðŸ˜¨', 'ðŸ˜±', 'âš ï¸', 'ðŸš«'],
                'neutral': ['ðŸš¨', 'ðŸš‘', 'ðŸš’', 'ðŸ‘¨â€ðŸš’', 'ðŸ‘©â€ðŸš’', 'ðŸ‘®â€â™‚ï¸', 'ðŸ‘®â€â™€ï¸', 'ðŸš“', 'ðŸ’™']
            },
            'fire': {
                'positive': ['ðŸ”¥', 'ðŸš’', 'ðŸ‘¨â€ðŸš’', 'ðŸ‘©â€ðŸš’', 'ðŸ’ª', 'ðŸ†', 'ðŸ’™', 'ðŸ†˜'],
                'negative': ['ðŸ”¥', 'ðŸ’”', 'ðŸ˜°', 'ðŸ˜¨', 'ðŸ˜±', 'âš ï¸', 'ðŸš«'],
                'neutral': ['ðŸ”¥', 'ðŸš’', 'ðŸ‘¨â€ðŸš’', 'ðŸ‘©â€ðŸš’', 'ðŸ’™']
            },
            'police': {
                'positive': ['ðŸ‘®â€â™‚ï¸', 'ðŸ‘®â€â™€ï¸', 'ðŸš“', 'ðŸ’™', 'ðŸ†˜', 'ðŸ’ª', 'ðŸ†'],
                'negative': ['ðŸ‘®â€â™‚ï¸', 'ðŸ‘®â€â™€ï¸', 'ðŸš“', 'ðŸ’”', 'ðŸ˜°', 'ðŸ˜¨', 'ðŸ˜±'],
                'neutral': ['ðŸ‘®â€â™‚ï¸', 'ðŸ‘®â€â™€ï¸', 'ðŸš“', 'ðŸ’™']
            },
            'ambulance': {
                'positive': ['ðŸš‘', 'ðŸ‘¨â€âš•ï¸', 'ðŸ‘©â€âš•ï¸', 'ðŸ’™', 'ðŸ†˜', 'ðŸ’ª', 'ðŸ†'],
                'negative': ['ðŸš‘', 'ðŸ’”', 'ðŸ˜°', 'ðŸ˜¨', 'ðŸ˜±', 'âš ï¸'],
                'neutral': ['ðŸš‘', 'ðŸ‘¨â€âš•ï¸', 'ðŸ‘©â€âš•ï¸', 'ðŸ’™']
            },
            
            # ðŸ’° FINANCE & BUSINESS (EXISTANT - AMÃ‰LIORÃ‰)
            'money': {
                'positive': ['ðŸ’°', 'ðŸ’Ž', 'ðŸ†', 'ðŸ“ˆ', 'ðŸ’¹', 'ðŸ’µ', 'ðŸª™', 'ðŸ’²', 'ðŸ…', 'ðŸ¥‡', 'ðŸŽ¯', 'ðŸš€', 'ðŸ”¥', 'ðŸ’ª', 'ðŸŽ‰', 'â­', 'ðŸŒŸ', 'ðŸ’«', 'âœ¨', 'ðŸŽŠ'],
                'negative': ['ðŸ“‰', 'ðŸ’¸', 'âŒ', 'ðŸ’£', 'ðŸ’¥', 'ðŸ›‘', 'âš ï¸', 'ðŸš«', 'ðŸ’”', 'ðŸ˜°', 'ðŸ˜¨', 'ðŸ˜±', 'ðŸ˜­', 'ðŸ˜¢', 'ðŸ˜ž', 'ðŸ˜”', 'ðŸ˜Ÿ', 'ðŸ˜•', 'ðŸ™', 'â˜¹ï¸'],
                'neutral': ['ðŸ’³', 'ðŸ¦', 'ðŸ“Š', 'ðŸ“‹', 'ðŸ“', 'ðŸ“„', 'ðŸ“±', 'ðŸ’»', 'ðŸ“ž', 'ðŸ“§', 'ðŸ“¨', 'ðŸ“©', 'ðŸ“ª', 'ðŸ“«', 'ðŸ“¬', 'ðŸ“­', 'ðŸ“®', 'ðŸ“¯', 'ðŸ“°', 'ðŸ“±']
            },
            'investment': {
                'positive': ['ðŸ“ˆ', 'ðŸ’¹', 'ðŸ’Ž', 'ðŸ†', 'âœ…', 'ðŸŒŸ'],
                'negative': ['ðŸ“‰', 'âŒ', 'ðŸ’¸', 'ðŸ’£', 'âš ï¸'],
                'neutral': ['ðŸ“Š', 'ðŸ“‹', 'ðŸ“', 'ðŸ“„', 'ðŸ’¼']
            },
            'business': {
                'positive': ['ðŸ’¼', 'ðŸ“Š', 'ðŸ“ˆ', 'ðŸ’¹', 'ðŸ’Ž', 'ðŸ†', 'âœ…', 'ðŸŒŸ'],
                'negative': ['ðŸ“‰', 'âŒ', 'ðŸ’¸', 'ðŸ’£', 'âš ï¸'],
                'neutral': ['ðŸ’¼', 'ðŸ“Š', 'ðŸ“‹', 'ðŸ“', 'ðŸ“„']
            },
            
            # ðŸš€ TECHNOLOGY & INNOVATION (EXISTANT - AMÃ‰LIORÃ‰)
            'technology': {
                'positive': ['ðŸ’»', 'ðŸ¤–', 'ðŸš€', 'ðŸ’¡', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ’»', 'ðŸ¤–', 'ðŸ’¡', 'ðŸ“', 'ðŸ“±']
            },
            'innovation': {
                'positive': ['ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥', 'âœ¨'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ’¡', 'ðŸ“', 'ðŸ“±']
            },
            'digital': {
                'positive': ['ðŸ’»', 'ðŸ¤–', 'ðŸš€', 'ðŸ’¡', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ’»', 'ðŸ¤–', 'ðŸ’¡', 'ðŸ“', 'ðŸ“±']
            },
            
            # â¤ï¸ HEALTH & FITNESS (EXISTANT - AMÃ‰LIORÃ‰)
            'health': {
                'positive': ['â¤ï¸', 'ðŸ’ª', 'ðŸƒ', 'ðŸš´', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['â¤ï¸', 'ðŸ’ª', 'ðŸƒ', 'ðŸš´', 'ðŸ”¥']
            },
            'fitness': {
                'positive': ['ðŸ’ª', 'ðŸƒ', 'ðŸš´', 'ðŸ‹ï¸', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ’ª', 'ðŸƒ', 'ðŸš´', 'ðŸ‹ï¸', 'ðŸ”¥']
            },
            'wellness': {
                'positive': ['â¤ï¸', 'ðŸ’ª', 'ðŸƒ', 'ðŸš´', 'ðŸ†', 'â­', 'ðŸŒŸ', 'ðŸ”¥'],
                'negative': ['ðŸ˜µ', 'ðŸ¤¯', 'ðŸ˜´', 'ðŸ’¤', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['â¤ï¸', 'ðŸ’ª', 'ðŸƒ', 'ðŸš´', 'ðŸ”¥']
            }
        }
        
        # ðŸš« MAPPING SPÃ‰CIFIQUE POUR Ã‰VITER LES EMOJIS SUR LES MOTS DE LIAISON
        self.specific_keyword_mapping = {
            # ðŸ§  Concepts cognitifs - PRIORITÃ‰ MAXIMALE
            'attention': 'ðŸ§ ',      # Au lieu de ðŸ’° (argent)
            'thinking': 'ðŸ§ ',       # Cerveau/pensÃ©e
            'brain': 'ðŸ§ ',          # Cerveau
            'learning': 'ðŸ“š',       # Apprentissage
            'studying': 'ðŸ“š',       # Ã‰tudes
            'reading': 'ðŸ“–',        # Lecture
            'math': 'ðŸ”¢',           # MathÃ©matiques
            'workout': 'ðŸ’ª',        # Exercice
            'exercise': 'ðŸ’ª',       # Exercice
            'physical': 'ðŸ’ª',       # Physique
            'challenging': 'ðŸ’ª',    # DÃ©fi
            'difficult': 'ðŸ’ª',      # Difficile
            
            # ðŸ§¬ Neuroscience - PRIORITÃ‰ MAXIMALE
            'acetylcholine': 'ðŸ§ ',  # Neurotransmetteur
            'norepinephrine': 'ðŸ§ ', # Neurotransmetteur
            'synapses': 'ðŸ§ ',       # Synapses
            'plasticity': 'ðŸ§ ',     # PlasticitÃ©
            'neuroscience': 'ðŸ§ ',   # Neuroscience
            'research': 'ðŸ”¬',       # Recherche
            'studies': 'ðŸ”¬',        # Ã‰tudes
            'science': 'ðŸ”¬',        # Science
            
            # ðŸŽ“ UniversitÃ© - PRIORITÃ‰ MAXIMALE
            'stanford': 'ðŸŽ“',       # UniversitÃ© Stanford
            'university': 'ðŸŽ“',     # UniversitÃ©
            'college': 'ðŸŽ“',        # CollÃ¨ge
            'academic': 'ðŸŽ“',       # AcadÃ©mique
            'education': 'ðŸŽ“',      # Ã‰ducation
            
            # ðŸ‘©â€ðŸŽ“ Apprentissage fÃ©minin - PRIORITÃ‰ MAXIMALE
            'she': 'ðŸ‘©â€ðŸŽ“',          # Elle (apprentissage)
            'her': 'ðŸ‘©â€ðŸŽ“',          # Elle (apprentissage)
            'woman': 'ðŸ‘©â€ðŸŽ“',        # Femme
            'female': 'ðŸ‘©â€ðŸŽ“',       # FÃ©minin
            
            # ðŸš€ NOUVEAUX MOTS AJOUTÃ‰S - PRIORITÃ‰ MAXIMALE
            'speed': 'âš¡',          # Vitesse (Ã©clair)
            'ability': 'ðŸ’ª',        # CapacitÃ© (muscle)
            'stuff': 'ðŸ“¦',          # Choses (boÃ®te)
            'striking': 'ðŸ‘Š',       # Frappant (poing)
            'right': 'âœ…',          # Correct (vÃ©rification)
            'best': 'ðŸ†',           # Meilleur (trophÃ©e)
            'growth': 'ðŸŒ±',         # Croissance (plante)
            'failure': 'ðŸ’¥',        # Ã‰chec (explosion)
            'success': 'ðŸŽ¯',        # SuccÃ¨s (cible)
            'brain': 'ðŸ§ ',          # Cerveau
            'reflexes': 'âš¡',        # RÃ©flexes (Ã©clair)
            'punch': 'ðŸ‘Š',          # Coup de poing
            'comedy': 'ðŸŽ­',         # ComÃ©die (thÃ©Ã¢tre)
            'risk': 'ðŸŽ²',           # Risque (dÃ©)
            'challenge': 'ðŸ”ï¸',      # DÃ©fi (montagne)
            'learning': 'ðŸ“š',       # Apprentissage
            'improvement': 'ðŸ“ˆ',    # AmÃ©lioration
            'motivation': 'ðŸ”¥',     # Motivation (feu)
            'strength': 'ðŸ’ª',       # Force (muscle)
            'power': 'âš¡',          # Pouvoir (Ã©clair)
            'fertility': 'ðŸŒ±',      # FertilitÃ© (plante qui pousse)
            'development': 'ðŸ“ˆ',    # DÃ©veloppement (graphique)
            
            # ðŸ§¬ MÃ‰DICAL & SCIENTIFIQUE - NOUVEAU
            'sperm': 'ðŸ§¬',          # SpermatozoÃ¯de
            'counts': 'ðŸ“Š',         # Comptage
            'microplastics': 'ðŸ”¬',  # Microplastiques
            'chemicals': 'ðŸ§ª',      # Produits chimiques
            'pesticides': 'â˜ ï¸',     # Pesticides
            'herbicides': 'ðŸŒ¿',     # Herbicides
            'endocrine': 'âš•ï¸',      # Endocrinien
            'system': 'âš™ï¸',         # SystÃ¨me
            'children': 'ðŸ‘¶',       # Enfants
            'testicles': 'ðŸ¥œ',      # Testicules
            'penis': 'ðŸ†',          # PÃ©nis
            'plastics': 'ðŸ”„',       # Plastiques
            'water': 'ðŸ’§',          # Eau
            'bottles': 'ðŸ¾',        # Bouteilles
            'foods': 'ðŸŽ',          # Aliments
            'microwave': 'ðŸ“¡',      # Micro-ondes
            'lifestyle': 'ðŸƒ',      # Mode de vie
            'sedentary': 'ðŸª‘',      # SÃ©dentaire
            'environmental': 'ðŸŒ',  # Environnemental
            
            # ðŸ§¬ MÃ‰DICAL & SCIENTIFIQUE - NOUVEAUX EMOJIS STRATÃ‰GIQUES
            'research': 'ðŸ”¬',        # Recherche
            'laboratory': 'ðŸ§ª',      # Laboratoire
            'experiment': 'âš—ï¸',      # ExpÃ©rience
            'discovery': 'ðŸ’¡',       # DÃ©couverte
            'innovation': 'ðŸš€',      # Innovation
            'breakthrough': 'ðŸ’¥',    # PercÃ©e
            'solution': 'âœ…',        # Solution
            'prevention': 'ðŸ›¡ï¸',     # PrÃ©vention
            'treatment': 'ðŸ’Š',       # Traitement
            'recovery': 'ðŸ”„',        # RÃ©cupÃ©ration
            'wellness': 'ðŸŒŸ',        # Bien-Ãªtre
            'vitality': 'ðŸ’ª',        # VitalitÃ©
            'immunity': 'ðŸ›¡ï¸',       # ImmunitÃ©
            'metabolism': 'âš¡',      # MÃ©tabolisme
            'hormones': 'âš•ï¸',        # Hormones
            'genes': 'ðŸ§¬',           # GÃ¨nes
            'dna': 'ðŸ§¬',             # ADN
            'cells': 'ðŸ”¬',           # Cellules
            'tissue': 'ðŸ”¬',          # Tissus
            'organ': 'â¤ï¸',           # Organe
            
            # ðŸ§  COGNITIF & PSYCHOLOGIQUE - NOUVEAUX
            'memory': 'ðŸ§ ',          # MÃ©moire
            'focus': 'ðŸŽ¯',           # Concentration
            'creativity': 'ðŸŽ¨',      # CrÃ©ativitÃ©
            'intelligence': 'ðŸ§ ',    # Intelligence
            'wisdom': 'ðŸ“š',          # Sagesse
            'knowledge': 'ðŸ“–',       # Connaissance
            'understanding': 'ðŸ’­',   # ComprÃ©hension
            'insight': 'ðŸ’¡',         # PerspicacitÃ©
            'awareness': 'ðŸ‘ï¸',      # Conscience
            'mindfulness': 'ðŸ§˜',     # Pleine conscience
            
            # ðŸƒ PHYSIQUE & PERFORMANCE - NOUVEAUX
            'endurance': 'ðŸƒ',       # Endurance
            'flexibility': 'ðŸ§˜',     # FlexibilitÃ©
            'balance': 'âš–ï¸',         # Ã‰quilibre
            'coordination': 'ðŸŽ¯',    # Coordination
            'agility': 'âš¡',          # AgilitÃ©
            'speed': 'ðŸƒ',           # Vitesse
            'precision': 'ðŸŽ¯',       # PrÃ©cision
            'control': 'ðŸŽ®',         # ContrÃ´le
            'mastery': 'ðŸ†',         # MaÃ®trise
            'excellence': 'â­',      # Excellence
            
            # ðŸš« Mots de liaison - BLOQUÃ‰S (pas d'emoji)
            'it': '',
            'is': '',
            'the': '',
            'and': '',
            'or': '',
            'but': '',
            'in': '',
            'on': '',
            'at': '',
            'to': '',
            'for': '',
            'of': '',
            'with': '',
            'by': '',
            'from': '',
            'up': '',
            'out': '',
            'off': '',
            'down': '',
            'over': '',
            'under': '',
            'through': '',
            'during': '',
            'before': '',
            'after': '',
            'while': '',
            'since': '',
            'until': '',
            'because': '',
            'although': '',
            'unless': '',
            'whether': '',
            'if': '',
            'then': '',
            'else': '',
            'when': '',
            'where': '',
            'why': '',
            'how': '',
            'what': '',
            'who': '',
            'which': '',
            'that': '',
            'this': '',
            'these': '',
            'those': '',
            'there': '',
            'here': '',
            'now': '',
            'then': '',
            'soon': '',
            'later': '',
            'early': '',
            'late': '',
            'always': '',
            'never': '',
            'sometimes': '',
            'often': '',
            'usually': '',
            'rarely': '',
            'seldom': '',
            'hardly': '',
            'scarcely': '',
            'barely': '',
            'merely': '',
            'only': '',
            'just': '',
            'simply': '',
            'really': '',
            'very': '',
            'quite': '',
            'rather': '',
            'fairly': '',
            'pretty': '',
            'somewhat': '',
            'slightly': '',
            'extremely': '',
            'incredibly': '',
            'absolutely': '',
            'completely': '',
            'totally': '',
            'entirely': '',
            'wholly': '',
            'partly': '',
            'partially': '',
            'mostly': '',
            'mainly': '',
            'chiefly': '',
            'primarily': '',
            'essentially': '',
            'basically': '',
            'fundamentally': '',
            'naturally': '',
            'obviously': '',
            'clearly': '',
            'evidently': '',
            'apparently': '',
            'seemingly': '',
            'supposedly': '',
            'allegedly': '',
            'reportedly': '',
            'presumably': '',
            'probably': '',
            'possibly': '',
            'maybe': '',
            'perhaps': '',
            'might': '',
            'could': '',
            'would': '',
            'should': '',
            'must': '',
            'can': '',
            'will': '',
            'shall': '',
            'may': '',
            'do': '',
            'does': '',
            'did': '',
            'have': '',
            'has': '',
            'had': '',
            'am': '',
            'are': '',
            'was': '',
            'were': '',
            'get': '',
            'gets': '',
            'got': '',
            'getting': ''
        }
        
        # ðŸš« MOTS DE LIAISON Ã€ BLOQUER (pas d'emojis)
        self.linking_words = {
            'it', 'is', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'out', 'off', 'over', 'under',
            'this', 'that', 'these', 'those', 'a', 'an', 'as', 'if', 'when',
            'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
            'would', 'could', 'should', 'will', 'can', 'may', 'might', 'must',
            'have', 'has', 'had', 'do', 'does', 'did', 'be', 'been', 'being',
            'am', 'are', 'was', 'were', 'get', 'gets', 'got', 'getting'
        }
        
        # ðŸŽ¯ MODIFICATEURS D'INTENSITÃ‰ SIMPLIFIÃ‰S
        self.intensity_modifiers = {
            'very': 1.5,
            'really': 1.5,
            'extremely': 2.0,
            'incredibly': 2.0,
            'absolutely': 2.0,
            'completely': 1.8,
            'totally': 1.8,
            'slightly': 0.7,
            'somewhat': 0.8,
            'kind of': 0.6,
            'sort of': 0.6
        }
        
        # âœ¨ EMOJIS DE TRANSITION PAR TYPE
        self.transition_emojis = {
            'cut': 'âš¡',
            'fade': 'âœ¨',
            'zoom': 'ðŸ”',
            'slide': 'âž¡ï¸',
            'dissolve': 'ðŸ’«',
            'wipe': 'ðŸ§¹'
        }

        # ðŸš¨ CORRECTION IMMÃ‰DIATE: Mapping spÃ©cifique pour les mots problÃ©matiques
        self.critical_mapping = {
            # ðŸ  FAMILLE & ENVIRONNEMENT
            'family': {
                'positive': ['ðŸ‘¨â€ðŸ‘©â€ðŸ‘§â€ðŸ‘¦', 'â¤ï¸', 'ðŸ ', 'ðŸ’•', 'ðŸ‘ª'],
                'negative': ['ðŸ˜”', 'ðŸ’”', 'ðŸšï¸', 'ðŸ˜¢', 'ðŸ˜ž'],
                'neutral': ['ðŸ‘¨â€ðŸ‘©â€ðŸ‘§â€ðŸ‘¦', 'ðŸ ', 'ðŸ‘ª']
            },
            'environment': {
                'positive': ['ðŸŒ', 'ðŸŒ±', 'ðŸŒ³', 'ðŸŒ¿', 'ðŸžï¸'],
                'negative': ['ðŸ­', 'ðŸ’¨', 'ðŸŒ«ï¸', 'â˜ï¸', 'ðŸ˜·'],
                'neutral': ['ðŸŒ', 'ðŸžï¸', 'ðŸŒ³']
            },
            'neighborhood': {
                'positive': ['ðŸ˜ï¸', 'ðŸŒ³', 'ðŸš¶', 'ðŸ ', 'ðŸŒ†'],
                'negative': ['ðŸš¨', 'ðŸ’€', 'ðŸ˜±', 'ðŸšï¸', 'ðŸ’”'],
                'neutral': ['ðŸ˜ï¸', 'ðŸ ', 'ðŸŒ†']
            },
            
            # ðŸš¨ CRIME & VIOLENCE
            'crime': {
                'positive': ['ðŸš”', 'ðŸ›¡ï¸', 'ðŸ‘®', 'âš–ï¸', 'ðŸ”’'],
                'negative': ['ðŸš¨', 'ðŸ’€', 'ðŸ˜±', 'ðŸ”ª', 'ðŸ’£'],
                'neutral': ['ðŸš”', 'âš–ï¸', 'ðŸ”’']
            },
            'gangs': {
                'positive': ['ðŸš”', 'ðŸ›¡ï¸', 'ðŸ‘®', 'âš–ï¸', 'ðŸ”’'],
                'negative': ['ðŸ’€', 'ðŸ˜±', 'ðŸ”ª', 'ðŸ’£', 'ðŸš¨'],
                'neutral': ['ðŸš”', 'âš–ï¸', 'ðŸ”’']
            },
            'drugs': {
                'positive': ['ðŸ’Š', 'ðŸ¥', 'ðŸ‘¨â€âš•ï¸', 'ðŸ©º', 'â¤ï¸'],
                'negative': ['ðŸ’€', 'ðŸ˜±', 'â˜ ï¸', 'ðŸ’‰', 'ðŸš¨'],
                'neutral': ['ðŸ’Š', 'ðŸ¥', 'ðŸ‘¨â€âš•ï¸']
            },
            
            # ðŸ¥ SANTÃ‰ & MÃ‰DECINE
            'healthcare': {
                'positive': ['ðŸ¥', 'ðŸ‘¨â€âš•ï¸', 'ðŸ©º', 'ðŸ’Š', 'â¤ï¸'],
                'negative': ['ðŸ˜·', 'ðŸ’‰', 'ðŸ¥', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ¥', 'ðŸ‘¨â€âš•ï¸', 'ðŸ©º']
            },
            'medical': {
                'positive': ['ðŸ¥', 'ðŸ‘¨â€âš•ï¸', 'ðŸ©º', 'ðŸ’Š', 'â¤ï¸'],
                'negative': ['ðŸ˜·', 'ðŸ’‰', 'ðŸ¥', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ¥', 'ðŸ‘¨â€âš•ï¸', 'ðŸ©º']
            },
            'hurt': {
                'positive': ['ðŸ¥', 'ðŸ‘¨â€âš•ï¸', 'ðŸ©º', 'ðŸ’Š', 'â¤ï¸'],
                'negative': ['ðŸ˜¢', 'ðŸ˜°', 'ðŸ’”', 'ðŸ˜¨', 'ðŸ˜±'],
                'neutral': ['ðŸ¥', 'ðŸ‘¨â€âš•ï¸', 'ðŸ©º']
            },
            'operation': {
                'positive': ['ðŸ¥', 'ðŸ‘¨â€âš•ï¸', 'ðŸ©º', 'ðŸ’Š', 'â¤ï¸'],
                'negative': ['ðŸ˜·', 'ðŸ’‰', 'ðŸ¥', 'ðŸ˜°', 'ðŸ˜¨'],
                'neutral': ['ðŸ¥', 'ðŸ‘¨â€âš•ï¸', 'ðŸ©º']
            },
            
            # ðŸš’ SERVICES D'URGENCE
            'fire': {
                'positive': ['ðŸš’', 'ðŸ‘¨â€ðŸš’', 'ðŸ”¥', 'ðŸ›¡ï¸', 'ðŸ’ª'],
                'negative': ['ðŸ”¥', 'ðŸ’€', 'ðŸ˜±', 'ðŸšï¸', 'ðŸ’”'],
                'neutral': ['ðŸš’', 'ðŸ‘¨â€ðŸš’', 'ðŸ”¥']
            },
            'department': {
                'positive': ['ðŸ¢', 'ðŸ‘¨â€ðŸ’¼', 'ðŸ“‹', 'ðŸ’¼', 'ðŸ›ï¸'],
                'negative': ['ðŸ˜”', 'ðŸ˜¤', 'ðŸ˜’', 'ðŸ˜ž', 'ðŸ˜•'],
                'neutral': ['ðŸ¢', 'ðŸ‘¨â€ðŸ’¼', 'ðŸ“‹']
            },
            
            # ðŸ’° FINANCE & SOCIÃ‰TÃ‰
            'money': {
                'positive': ['ðŸ’°', 'ðŸ’µ', 'ðŸ’Ž', 'ðŸ†', 'â­'],
                'negative': ['ðŸ’¸', 'ðŸ˜”', 'ðŸ’”', 'ðŸ˜¢', 'ðŸ˜ž'],
                'neutral': ['ðŸ’°', 'ðŸ’µ', 'ðŸ’Ž']
            },
            'bankrupt': {
                'positive': ['ðŸ’°', 'ðŸ’µ', 'ðŸ’Ž', 'ðŸ†', 'â­'],
                'negative': ['ðŸ’¸', 'ðŸ˜”', 'ðŸ’”', 'ðŸ˜¢', 'ðŸ˜ž'],
                'neutral': ['ðŸ’°', 'ðŸ’µ', 'ðŸ’Ž']
            },
            'tax': {
                'positive': ['ðŸ’°', 'ðŸ’µ', 'ðŸ’Ž', 'ðŸ†', 'â­'],
                'negative': ['ðŸ’¸', 'ðŸ˜”', 'ðŸ’”', 'ðŸ˜¢', 'ðŸ˜ž'],
                'neutral': ['ðŸ’°', 'ðŸ’µ', 'ðŸ’Ž']
            },
            
            # ðŸ›ï¸ POLITIQUE & SOCIÃ‰TÃ‰
            'socialist': {
                'positive': ['ðŸ›ï¸', 'ðŸ‘¥', 'ðŸ¤', 'ðŸŒ', 'â¤ï¸'],
                'negative': ['ðŸ˜”', 'ðŸ˜¤', 'ðŸ˜’', 'ðŸ˜ž', 'ðŸ˜•'],
                'neutral': ['ðŸ›ï¸', 'ðŸ‘¥', 'ðŸ¤']
            },
            'society': {
                'positive': ['ðŸ›ï¸', 'ðŸ‘¥', 'ðŸ¤', 'ðŸŒ', 'â¤ï¸'],
                'negative': ['ðŸ˜”', 'ðŸ˜¤', 'ðŸ˜’', 'ðŸ˜ž', 'ðŸ˜•'],
                'neutral': ['ðŸ›ï¸', 'ðŸ‘¥', 'ðŸ¤']
            },
            'community': {
                'positive': ['ðŸ›ï¸', 'ðŸ‘¥', 'ðŸ¤', 'ðŸŒ', 'â¤ï¸'],
                'negative': ['ðŸ˜”', 'ðŸ˜¤', 'ðŸ˜’', 'ðŸ˜ž', 'ðŸ˜•'],
                'neutral': ['ðŸ›ï¸', 'ðŸ‘¥', 'ðŸ¤']
            }
        }

    def get_emoji_for_context(self, keyword: str, text: str = "", sentiment: str = "neutral", intensity: float = 1.0) -> str:
        """Obtient un emoji contextuel OPTIMISÃ‰ pour un mot-clÃ©"""
        try:
            keyword_lower = keyword.lower().strip()
            
            # ðŸš« BLOQUAGE DES MOTS DE LIAISON (AMÃ‰LIORÃ‰)
            if keyword_lower in self.linking_words:
                return ""
            
            # ðŸš¨ PRIORITÃ‰ 0: MAPPING CRITIQUE POUR LES MOTS PROBLÃ‰MATIQUES (NOUVEAU)
            if keyword_lower in self.critical_mapping:
                if sentiment in self.critical_mapping[keyword_lower]:
                    emoji_list = self.critical_mapping[keyword_lower][sentiment]
                    if emoji_list:
                        # SÃ©lection intelligente basÃ©e sur l'intensitÃ©
                        if intensity > 1.5:
                            # IntensitÃ© Ã©levÃ©e: emojis plus expressifs
                            high_intensity = [e for e in emoji_list if e in ['ðŸš¨', 'ðŸ’€', 'ðŸ˜±', 'ðŸ”¥', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ']]
                            return random.choice(high_intensity) if high_intensity else random.choice(emoji_list)
                        elif intensity < 0.5:
                            # IntensitÃ© faible: emojis plus subtils
                            low_intensity = [e for e in emoji_list if e in ['ðŸ’­', 'ðŸ“', 'ðŸ“š', 'ðŸ”', 'ðŸ’¡', 'ðŸŽ¯', 'ðŸ“Š', 'ðŸ“ˆ']]
                            return random.choice(low_intensity) if low_intensity else random.choice(emoji_list)
                        else:
                            # IntensitÃ© normale: sÃ©lection alÃ©atoire
                            return random.choice(emoji_list)
            
            # ðŸŽ¯ PRIORITÃ‰ 1: MAPPING SPÃ‰CIFIQUE DIRECT (NOUVEAU - OPTIMISÃ‰)
            if keyword_lower in self.specific_keyword_mapping:
                emoji = self.specific_keyword_mapping[keyword_lower]
                if emoji:
                    return emoji
            
            # ðŸ§  PRIORITÃ‰ 2: MAPPING SÃ‰MANTIQUE AVANCÃ‰ (NOUVEAU - OPTIMISÃ‰)
            for category, emojis in self.semantic_mapping.items():
                if keyword_lower in category or any(kw in keyword_lower for kw in category.split('_')):
                    if sentiment in emojis:
                        emoji_list = emojis[sentiment]
                        if emoji_list:
                            # SÃ©lection intelligente basÃ©e sur l'intensitÃ©
                            if intensity > 1.5:
                                # IntensitÃ© Ã©levÃ©e: emojis plus expressifs
                                high_intensity = ['ðŸš€', 'ðŸ’¥', 'ðŸ”¥', 'âš¡', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ']
                                return random.choice(high_intensity)
                            elif intensity < 0.5:
                                # IntensitÃ© faible: emojis plus subtils
                                low_intensity = ['ðŸ’­', 'ðŸ“', 'ðŸ“š', 'ðŸ”', 'ðŸ’¡', 'ðŸŽ¯', 'ðŸ“Š', 'ðŸ“ˆ']
                                return random.choice(low_intensity)
                            else:
                                # IntensitÃ© normale: sÃ©lection alÃ©atoire
                                return random.choice(emoji_list)
            
            # ðŸ” PRIORITÃ‰ 3: RECHERCHE PARTIELLE INTELLIGENTE (NOUVEAU)
            for category, emojis in self.semantic_mapping.items():
                if any(kw in keyword_lower for kw in category.split('_')):
                    if sentiment in emojis:
                        emoji_list = emojis[sentiment]
                        if emoji_list:
                            return random.choice(emoji_list)
            
            # ðŸŽ¨ PRIORITÃ‰ 4: MAPPING GÃ‰NÃ‰RIQUE INTELLIGENT (NOUVEAU)
            generic_mapping = {
                'positive': ['âœ…', 'ðŸ‘', 'ðŸŽ¯', 'ðŸ’¡', 'ðŸš€', 'ðŸ’ª', 'ðŸ†', 'â­', 'ðŸŒŸ', 'âœ¨', 'ðŸ’Ž', 'ðŸ”¥', 'âš¡'],
                'negative': ['âŒ', 'ðŸ‘Ž', 'ðŸ˜”', 'ðŸ˜¢', 'ðŸ˜°', 'ðŸ˜¨', 'ðŸ’”', 'ðŸ’¥', 'ðŸ’¢', 'ðŸ˜¤', 'ðŸ˜¡', 'ðŸ¤¬'],
                'neutral': ['ðŸ’­', 'ðŸ“', 'ðŸ“š', 'ðŸ”', 'ðŸ’¡', 'ðŸŽ¯', 'ðŸ“Š', 'ðŸ“ˆ', 'ðŸ“‹', 'ðŸ“–', 'ðŸ”Ž', 'ðŸ’¬']
            }
            
            if sentiment in generic_mapping:
                return random.choice(generic_mapping[sentiment])
            
            # ðŸŽ¯ PRIORITÃ‰ 5: EMOJI PAR DÃ‰FAUT INTELLIGENT (NOUVEAU)
            default_emojis = ['ðŸ’¡', 'ðŸŽ¯', 'ðŸ“', 'ðŸ”', 'ðŸ’­', 'ðŸ“š', 'ðŸ“Š', 'ðŸ“ˆ', 'âœ¨', 'ðŸŒŸ']
            return random.choice(default_emojis)
            
        except Exception as e:
            print(f"âŒ Erreur get_emoji_for_context: {e}")
            return ""

    def get_emoji_sequence(self, keywords: List[str], context: str = "", max_emojis: int = 3) -> List[str]:
        """Obtient une sÃ©quence cohÃ©rente d'emojis pour plusieurs mots-clÃ©s"""
        try:
            emojis = []
            used_categories = set()
            
            for keyword in keywords[:max_emojis]:
                if keyword.lower() in self.linking_words:
                    continue
                    
                emoji = self.get_emoji_for_context(keyword, context)
                if emoji:
                    emojis.append(emoji)
                    
                    # Ã‰viter la rÃ©pÃ©tition de catÃ©gories
                    for category in self.semantic_mapping:
                        if keyword.lower() in category or category in keyword.lower():
                            used_categories.add(category)
                            break
            
            return emojis
            
        except Exception as e:
            print(f"âŒ Erreur get_emoji_sequence: {e}")
            return []

    def get_transition_emoji(self, transition_type: str = "cut") -> str:
        """Obtient un emoji selon le type de transition"""
        return self.transition_emojis.get(transition_type, "âœ¨")

    def is_linking_word(self, word: str) -> bool:
        """VÃ©rifie si un mot est un mot de liaison (bloquÃ©)"""
        return word.lower() in self.linking_words

# Instance globale
contextual_emojis_complete = ContextualEmojiSystemComplete() 
