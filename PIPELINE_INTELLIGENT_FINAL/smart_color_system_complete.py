ï»¿# -*- coding: utf-8 -*-
"""
SystÃƒÂ¨me de couleurs intelligentes COMPLET pour les sous-titres Hormozi
Bloque la coloration des mots de liaison et amÃƒÂ©liore la contextualisation
"""

import random
from typing import Dict, List, Optional, Tuple
import re

class SmartColorSystemComplete:
    """SystÃƒÂ¨me de couleurs intelligentes COMPLET avec blocage des mots de liaison"""
    
    def __init__(self):
        # Ã°Å¸Å¡Â« MOTS DE LIAISON Ãƒâ‚¬ BLOQUER (pas de coloration)
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
        
        # Ã°Å¸Å½Â¨ COULEURS CONTEXTUELLES ENRICHIES (60+ COULEURS)
        self.context_colors = {
            # Ã°Å¸Â§Â  COGNITIVE & LEARNING (NOUVEAU - COMPLET)
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
            
            # Ã°Å¸Â§Â¬ NEUROSCIENCE & SCIENCE (NOUVEAU - COMPLET)
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
            
            # Ã°Å¸Å½â€œ UNIVERSITY & ACADEMIC (NOUVEAU - COMPLET)
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
            
            # Ã°Å¸â€˜Â©Ã¢â‚¬ÂÃ°Å¸Å½â€œ FEMALE LEARNING & FRUSTRATION (NOUVEAU - COMPLET)
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
            
            # Ã°Å¸Å¡Â¨ SERVICES D'URGENCE (EXISTANT - AMÃƒâ€°LIORÃƒâ€°)
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
            
            # Ã°Å¸â€™Â° FINANCE & BUSINESS (EXISTANT - AMÃƒâ€°LIORÃƒâ€°)
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
            
            # Ã°Å¸Å¡â‚¬ TECHNOLOGY & INNOVATION (EXISTANT - AMÃƒâ€°LIORÃƒâ€°)
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
            
            # Ã¢ÂÂ¤Ã¯Â¸Â HEALTH & FITNESS (EXISTANT - AMÃƒâ€°LIORÃƒâ€°)
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
        
        # Ã°Å¸Å½Â¯ MAPPING MOT-CLÃƒâ€° Ã¢â€ â€™ CONTEXTE (PRIORITÃƒâ€° MAXIMALE)
        self.keyword_context_mapping = {
            # Ã°Å¸Â§Â  Concepts cognitifs - PRIORITÃƒâ€° MAXIMALE
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
            
            # Ã°Å¸Â§Â¬ Neuroscience - PRIORITÃƒâ€° MAXIMALE
            'acetylcholine': 'neuroscience',
            'norepinephrine': 'neuroscience',
            'synapses': 'neuroscience',
            'plasticity': 'neuroscience',
            'neuroscience': 'neuroscience',
            'research': 'neuroscience',
            'studies': 'neuroscience',
            'science': 'neuroscience',
            
            # Ã°Å¸Å½â€œ UniversitÃƒÂ© - PRIORITÃƒâ€° MAXIMALE
            'stanford': 'university',
            'university': 'university',
            'college': 'university',
            'academic': 'university',
            'education': 'university',
            
            # Ã°Å¸â€˜Â©Ã¢â‚¬ÂÃ°Å¸Å½â€œ Apprentissage fÃƒÂ©minin - PRIORITÃƒâ€° MAXIMALE
            'she': 'female',
            'her': 'female',
            'woman': 'female',
            'female': 'female',
            'girl': 'female',
            'frustrating': 'frustration',
            'frustration': 'frustration',
        }

    def get_color_for_keyword(self, keyword: str, text: str = "", intensity: float = 1.0) -> str:
        """Obtient une couleur intelligente pour un mot-clÃƒÂ© avec blocage des mots de liaison"""
        try:
            # Ã°Å¸Å¡Â« VÃƒÂ©rifier si c'est un mot de liaison (bloquÃƒÂ© - retourne blanc)
            if keyword.lower() in self.linking_words:
                return "#FFFFFF"  # Blanc pour les mots de liaison
            
            # Ã°Å¸Å½Â¯ VÃƒÂ©rifier le mapping spÃƒÂ©cifique PRIORITAIRE
            if keyword.lower() in self.keyword_context_mapping:
                context = self.keyword_context_mapping[keyword.lower()]
                if context in self.context_colors:
                    colors = self.context_colors[context]['positive']
                    if colors:
                        return random.choice(colors)
            
            # Ã°Å¸â€Â Recherche dans le mapping contextuel
            for context, colors_dict in self.context_colors.items():
                if keyword.lower() in context or context in keyword.lower():
                    colors = colors_dict.get('positive', colors_dict.get('neutral', []))
                    if colors:
                        # Ajuster l'intensitÃƒÂ©
                        adjusted_intensity = min(intensity, 2.0)
                        if adjusted_intensity > 1.5 and colors:
                            return random.choice(colors[:3])  # Top 3 pour haute intensitÃƒÂ©
                        else:
                            return random.choice(colors)
            
            # Ã°Å¸Å½Â¯ Recherche par similaritÃƒÂ© de mots
            for context, colors_dict in self.context_colors.items():
                if any(word in keyword.lower() for word in context.split('_')):
                    colors = colors_dict.get('positive', colors_dict.get('neutral', []))
                    if colors:
                        return random.choice(colors)
            
            # Ã°Å¸Å½Â¨ Couleur par dÃƒÂ©faut (neutre) - AMÃƒâ€°LIORÃƒâ€°E
            # Utiliser des couleurs diffÃƒÂ©rentes selon le mot pour plus de diversitÃƒÂ©
            default_colors = [
                "#4682B4",  # Bleu acier
                "#20B2AA",  # Vert mer
                "#DC143C",  # Rouge cramoisi
                "#FF8C00",  # Orange foncÃƒÂ©
                "#32CD32",  # Vert lime
                "#FF1493",  # Rose profond
                "#00CED1",  # Turquoise
                "#FFD700",  # Or
                "#9370DB",  # Violet moyen
                "#00FF7F",  # Vert printemps
            ]
            
            # SÃƒÂ©lectionner une couleur basÃƒÂ©e sur le hash du mot pour la cohÃƒÂ©rence
            import hashlib
            hash_value = int(hashlib.md5(keyword.lower().encode()).hexdigest(), 16)
            color_index = hash_value % len(default_colors)
            return default_colors[color_index]
            
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur get_color_for_keyword: {e}")
            return "#FFFFFF"  # Blanc par dÃƒÂ©faut en cas d'erreur

    def get_color_scheme(self, keyword: str, context: str = "", scheme_type: str = "monochromatic") -> List[str]:
        """Obtient un schÃƒÂ©ma de couleurs pour un mot-clÃƒÂ©"""
        try:
            base_color = self.get_color_for_keyword(keyword, context)
            if not base_color or base_color == "#FFFFFF":
                return ["#FFFFFF"]  # Blanc pour les mots de liaison
            
            # SchÃƒÂ©mas de couleurs basiques
            if scheme_type == "monochromatic":
                return [base_color, base_color, base_color]
            elif scheme_type == "complementary":
                # Logique de couleurs complÃƒÂ©mentaires simplifiÃƒÂ©e
                return [base_color, "#FFFFFF", base_color]
            else:
                return [base_color]
                
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur get_color_scheme: {e}")
            return ["#FFFFFF"]

    def adjust_color_intensity(self, color: str, intensity: float) -> str:
        """Ajuste l'intensitÃƒÂ© d'une couleur"""
        try:
            if not color or color == "#FFFFFF":
                return "#FFFFFF"  # Garder le blanc pour les mots de liaison
            
            # Logique d'ajustement d'intensitÃƒÂ© simplifiÃƒÂ©e
            return color
            
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur adjust_color_intensity: {e}")
            return color

    def is_linking_word(self, word: str) -> bool:
        """VÃƒÂ©rifie si un mot est un mot de liaison (bloquÃƒÂ©)"""
        return word.lower() in self.linking_words

# Instance globale
smart_colors_complete = SmartColorSystemComplete() 

