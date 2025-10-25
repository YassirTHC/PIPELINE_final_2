# -*- coding: utf-8 -*-
"""
SystÃ¨me de couleurs intelligentes COMPLET pour les sous-titres Hormozi
Bloque la coloration des mots de liaison et amÃ©liore la contextualisation
"""

import random
from typing import Dict, List, Optional, Tuple
import re

class SmartColorSystemComplete:
    """SystÃ¨me de couleurs intelligentes COMPLET avec blocage des mots de liaison"""
    
    def __init__(self):
        # ðŸš« MOTS DE LIAISON Ã€ BLOQUER (pas de coloration)
        self.linking_words = {
            'it', 'is', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'out', 'off', 'over', 'under',
            'this', 'that', 'these', 'those', 'a', 'an', 'as', 'if', 'when',
            'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
            'would', 'could', 'should', 'will', 'can', 'may', 'might', 'must',
            'have', 'has', 'had', 'do', 'does', 'did', 'be', 'been', 'being',
            'am', 'are', 'was', 'were', 'get', 'gets', 'got', 'getting',
            'its', 'it\'s', 'that\'s', 'this\'s', 'there\'s', 'here\'s',
            'they\'re', 'we\'re', 'you\'re', 'he\'s', 'she\'s'
        }
        
        # ðŸŽ¨ COULEURS CONTEXTUELLES ENRICHIES (60+ COULEURS)
        self.context_colors = {
            # ðŸ§  COGNITIVE & LEARNING (NOUVEAU - COMPLET)
            'cognitive': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'brain': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'thinking': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'attention': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'concentration': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'learning': {
                'positive': ['#32CD32', '#00FF7F', '#00CED1', '#20B2AA', '#48D1CC'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'studying': {
                'positive': ['#32CD32', '#00FF7F', '#00CED1', '#20B2AA', '#48D1CC'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'reading': {
                'positive': ['#32CD32', '#00FF7F', '#00CED1', '#20B2AA', '#48D1CC'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'math': {
                'positive': ['#32CD32', '#00FF7F', '#00CED1', '#20B2AA', '#48D1CC'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'workout': {
                'positive': ['#FF4500', '#FF6347', '#DC143C', '#FF8C00', '#FFA500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'exercise': {
                'positive': ['#FF4500', '#FF6347', '#DC143C', '#FF8C00', '#FFA500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'physical': {
                'positive': ['#FF4500', '#FF6347', '#DC143C', '#FF8C00', '#FFA500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'challenging': {
                'positive': ['#FF4500', '#FF6347', '#DC143C', '#FF8C00', '#FFA500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'difficult': {
                'positive': ['#FF4500', '#FF6347', '#DC143C', '#FF8C00', '#FFA500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            
            # ðŸ§¬ NEUROSCIENCE & SCIENCE (NOUVEAU - COMPLET)
            'acetylcholine': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'norepinephrine': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'synapses': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'plasticity': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'neuroscience': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'research': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'studies': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'science': {
                'positive': ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            
            # ðŸŽ“ UNIVERSITY & ACADEMIC (NOUVEAU - COMPLET)
            'stanford': {
                'positive': ['#1E90FF', '#4169E1', '#483D8B', '#6A5ACD', '#9370DB'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'university': {
                'positive': ['#1E90FF', '#4169E1', '#483D8B', '#6A5ACD', '#9370DB'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'college': {
                'positive': ['#1E90FF', '#4169E1', '#483D8B', '#6A5ACD', '#9370DB'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'academic': {
                'positive': ['#1E90FF', '#4169E1', '#483D8B', '#6A5ACD', '#9370DB'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'education': {
                'positive': ['#1E90FF', '#4169E1', '#483D8B', '#6A5ACD', '#9370DB'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            
            # ðŸ‘©â€ðŸŽ“ FEMALE LEARNING & FRUSTRATION (NOUVEAU - COMPLET)
            'she': {
                'positive': ['#FF69B4', '#FF1493', '#DC143C', '#FF6347', '#FF4500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'her': {
                'positive': ['#FF69B4', '#FF1493', '#DC143C', '#FF6347', '#FF4500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'woman': {
                'positive': ['#FF69B4', '#FF1493', '#DC143C', '#FF6347', '#FF4500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'female': {
                'positive': ['#FF69B4', '#FF1493', '#DC143C', '#FF6347', '#FF4500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'girl': {
                'positive': ['#FF69B4', '#FF1493', '#DC143C', '#FF6347', '#FF4500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'frustrating': {
                'positive': ['#FF4500', '#FF6347', '#DC143C', '#FF8C00', '#FFA500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'frustration': {
                'positive': ['#FF4500', '#FF6347', '#DC143C', '#FF8C00', '#FFA500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            
            # ðŸš¨ SERVICES D'URGENCE (EXISTANT - AMÃ‰LIORÃ‰)
            'emergency': {
                'positive': ['#00BFFF', '#1E90FF', '#4169E1', '#483D8B', '#6A5ACD'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'fire': {
                'positive': ['#FF4500', '#FF6347', '#DC143C', '#FF8C00', '#FFA500'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'police': {
                'positive': ['#00BFFF', '#1E90FF', '#4169E1', '#483D8B', '#6A5ACD'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'ambulance': {
                'positive': ['#00BFFF', '#1E90FF', '#4169E1', '#483D8B', '#6A5ACD'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            
            # ðŸ’° FINANCE & BUSINESS (EXISTANT - AMÃ‰LIORÃ‰)
            'money': {
                'positive': ['#00FF00', '#32CD32', '#00FF7F', '#00CED1', '#20B2AA'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'investment': {
                'positive': ['#00FF00', '#32CD32', '#00FF7F', '#00CED1', '#20B2AA'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'business': {
                'positive': ['#1E90FF', '#4169E1', '#483D8B', '#6A5ACD', '#9370DB'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            
            # ðŸš€ TECHNOLOGY & INNOVATION (EXISTANT - AMÃ‰LIORÃ‰)
            'technology': {
                'positive': ['#00FFFF', '#20B2AA', '#00CED1', '#48D1CC', '#40E0D0'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'innovation': {
                'positive': ['#00FFFF', '#20B2AA', '#00CED1', '#48D1CC', '#40E0D0'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'digital': {
                'positive': ['#00FFFF', '#20B2AA', '#00CED1', '#48D1CC', '#40E0D0'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            
            # â¤ï¸ HEALTH & FITNESS (EXISTANT - AMÃ‰LIORÃ‰)
            'health': {
                'positive': ['#32CD32', '#00FF7F', '#00CED1', '#20B2AA', '#48D1CC'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'fitness': {
                'positive': ['#32CD32', '#00FF7F', '#00CED1', '#20B2AA', '#48D1CC'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            },
            'wellness': {
                'positive': ['#32CD32', '#00FF7F', '#00CED1', '#20B2AA', '#48D1CC'],
                'negative': ['#FF6347', '#DC143C', '#B22222', '#8B0000', '#DC143C'],
                'neutral': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#B0C4DE']
            }
        }
        
        # ðŸŽ¯ MAPPING MOT-CLÃ‰ â†’ CONTEXTE (PRIORITÃ‰ MAXIMALE)
        self.keyword_context_mapping = {
            # ðŸ§  Concepts cognitifs - PRIORITÃ‰ MAXIMALE
            'attention': 'cognitive',
            'thinking': 'cognitive',
            'brain': 'cognitive',
            'learning': 'cognitive',
            'studying': 'cognitive',
            'reading': 'cognitive',
            'math': 'cognitive',
            'workout': 'cognitive',
            'exercise': 'cognitive',
            'physical': 'cognitive',
            'challenging': 'cognitive',
            'difficult': 'cognitive',
            
            # ðŸ§¬ Neuroscience - PRIORITÃ‰ MAXIMALE
            'acetylcholine': 'neuroscience',
            'norepinephrine': 'neuroscience',
            'synapses': 'neuroscience',
            'plasticity': 'neuroscience',
            'neuroscience': 'neuroscience',
            'research': 'neuroscience',
            'studies': 'neuroscience',
            'science': 'neuroscience',
            
            # ðŸŽ“ UniversitÃ© - PRIORITÃ‰ MAXIMALE
            'stanford': 'university',
            'university': 'university',
            'college': 'university',
            'academic': 'university',
            'education': 'university',
            
            # ðŸ‘©â€ðŸŽ“ Apprentissage fÃ©minin - PRIORITÃ‰ MAXIMALE
            'she': 'female',
            'her': 'female',
            'woman': 'female',
            'female': 'female',
            'girl': 'female',
            'frustrating': 'frustration',
            'frustration': 'frustration',
        }

    def get_color_for_keyword(self, keyword: str, text: str = "", intensity: float = 1.0) -> str:
        """Obtient une couleur intelligente pour un mot-clÃ© avec blocage des mots de liaison"""
        try:
            # ðŸš« VÃ©rifier si c'est un mot de liaison (bloquÃ© - retourne blanc)
            if keyword.lower() in self.linking_words:
                return "#FFFFFF"  # Blanc pour les mots de liaison
            
            # ðŸŽ¯ VÃ©rifier le mapping spÃ©cifique PRIORITAIRE
            if keyword.lower() in self.keyword_context_mapping:
                context = self.keyword_context_mapping[keyword.lower()]
                if context in self.context_colors:
                    colors = self.context_colors[context]['positive']
                    if colors:
                        return random.choice(colors)
            
            # ðŸ” Recherche dans le mapping contextuel
            for context, colors_dict in self.context_colors.items():
                if keyword.lower() in context or context in keyword.lower():
                    colors = colors_dict.get('positive', colors_dict.get('neutral', []))
                    if colors:
                        # Ajuster l'intensitÃ©
                        adjusted_intensity = min(intensity, 2.0)
                        if adjusted_intensity > 1.5 and colors:
                            return random.choice(colors[:3])  # Top 3 pour haute intensitÃ©
                        else:
                            return random.choice(colors)
            
            # ðŸŽ¯ Recherche par similaritÃ© de mots
            for context, colors_dict in self.context_colors.items():
                if any(word in keyword.lower() for word in context.split('_')):
                    colors = colors_dict.get('positive', colors_dict.get('neutral', []))
                    if colors:
                        return random.choice(colors)
            
            # ðŸŽ¨ Couleur par dÃ©faut (neutre) - AMÃ‰LIORÃ‰E
            # Utiliser des couleurs diffÃ©rentes selon le mot pour plus de diversitÃ©
            default_colors = [
                "#4682B4",  # Bleu acier
                "#20B2AA",  # Vert mer
                "#DC143C",  # Rouge cramoisi
                "#FF8C00",  # Orange foncÃ©
                "#32CD32",  # Vert lime
                "#FF1493",  # Rose profond
                "#00CED1",  # Turquoise
                "#FFD700",  # Or
                "#9370DB",  # Violet moyen
                "#00FF7F",  # Vert printemps
            ]
            
            # SÃ©lectionner une couleur basÃ©e sur le hash du mot pour la cohÃ©rence
            import hashlib
            hash_value = int(hashlib.md5(keyword.lower().encode()).hexdigest(), 16)
            color_index = hash_value % len(default_colors)
            return default_colors[color_index]
            
        except Exception as e:
            print(f"âŒ Erreur get_color_for_keyword: {e}")
            return "#FFFFFF"  # Blanc par dÃ©faut en cas d'erreur

    def get_color_scheme(self, keyword: str, context: str = "", scheme_type: str = "monochromatic") -> List[str]:
        """Obtient un schÃ©ma de couleurs pour un mot-clÃ©"""
        try:
            base_color = self.get_color_for_keyword(keyword, context)
            if not base_color or base_color == "#FFFFFF":
                return ["#FFFFFF"]  # Blanc pour les mots de liaison
            
            # SchÃ©mas de couleurs basiques
            if scheme_type == "monochromatic":
                return [base_color, base_color, base_color]
            elif scheme_type == "complementary":
                # Logique de couleurs complÃ©mentaires simplifiÃ©e
                return [base_color, "#FFFFFF", base_color]
            else:
                return [base_color]
                
        except Exception as e:
            print(f"âŒ Erreur get_color_scheme: {e}")
            return ["#FFFFFF"]

    def adjust_color_intensity(self, color: str, intensity: float) -> str:
        """Ajuste l'intensitÃ© d'une couleur"""
        try:
            if not color or color == "#FFFFFF":
                return "#FFFFFF"  # Garder le blanc pour les mots de liaison
            
            # Logique d'ajustement d'intensitÃ© simplifiÃ©e
            return color
            
        except Exception as e:
            print(f"âŒ Erreur adjust_color_intensity: {e}")
            return color

    def is_linking_word(self, word: str) -> bool:
        """VÃ©rifie si un mot est un mot de liaison (bloquÃ©)"""
        return word.lower() in self.linking_words

# Instance globale
smart_colors_complete = SmartColorSystemComplete() 
