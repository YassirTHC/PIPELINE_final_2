"""
Système d'emojis contextuels COMPLET et PRÉCIS pour les sous-titres Hormozi
Mapping précis en anglais pour tous les concepts avec détection intelligente
"""

import random
from typing import Dict, List, Optional, Tuple
import re

class ContextualEmojiSystemComplete:
    """Système d'emojis intelligents et contextuels COMPLET avec mapping précis"""
    
    def __init__(self):
        # 🎯 MAPPING SÉMANTIQUE COMPLET ET PRÉCIS (500+ EMOJIS)
        self.semantic_mapping = {
            # 🧠 COGNITIVE & LEARNING (NOUVEAU - COMPLET)
            'brain': {
                'positive': ['🧠', '💭', '💡', '🎯', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🧠', '💭', '💡', '🎯', '📝', '📚']
            },
            'thinking': {
                'positive': ['🧠', '💭', '💡', '🎯', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🧠', '💭', '💡', '🎯', '📝', '📚']
            },
            'attention': {
                'positive': ['👁️', '🎯', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['👁️', '🎯', '💡', '📝', '📚']
            },
            'concentration': {
                'positive': ['👁️', '🎯', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['👁️', '🎯', '💡', '📝', '📚']
            },
            'learning': {
                'positive': ['📚', '✏️', '🎓', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['📚', '✏️', '🎓', '💡', '📝']
            },
            'studying': {
                'positive': ['📚', '✏️', '🎓', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['📚', '✏️', '🎓', '💡', '📝']
            },
            'reading': {
                'positive': ['📚', '📖', '👓', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['📚', '📖', '👓', '💡', '📝']
            },
            'math': {
                'positive': ['🔢', '📐', '📏', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🔢', '📐', '📏', '💡', '📝']
            },
            'workout': {
                'positive': ['💪', '🏋️', '🚴', '🏃', '🏆', '⭐', '🌟', '🔥'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['💪', '🏋️', '🚴', '🏃', '🔥']
            },
            'exercise': {
                'positive': ['💪', '🏋️', '🚴', '🏃', '🏆', '⭐', '🌟', '🔥'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['💪', '🏋️', '🚴', '🏃', '🔥']
            },
            'physical': {
                'positive': ['💪', '🏋️', '🚴', '🏃', '🏆', '⭐', '🌟', '🔥'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['💪', '🏋️', '🚴', '🏃', '🔥']
            },
            'challenging': {
                'positive': ['💪', '🏋️', '🎯', '🚀', '🏆', '⭐', '🌟', '🔥'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['💪', '🏋️', '🎯', '🔥']
            },
            'difficult': {
                'positive': ['💪', '🏋️', '🎯', '🚀', '🏆', '⭐', '🌟', '🔥'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['💪', '🏋️', '🎯', '🔥']
            },
            
            # 🧬 NEUROSCIENCE & SCIENCE (NOUVEAU - COMPLET)
            'acetylcholine': {
                'positive': ['🧠', '🧬', '🔬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🧠', '🧬', '🔬', '💡', '📝']
            },
            'norepinephrine': {
                'positive': ['🧠', '🧬', '🔬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🧠', '🧬', '🔬', '💡', '📝']
            },
            'synapses': {
                'positive': ['🧠', '🧬', '🔬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🧠', '🧬', '🔬', '💡', '📝']
            },
            'plasticity': {
                'positive': ['🧠', '🧬', '🔬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🧠', '🧬', '🔬', '💡', '📝']
            },
            'neuroscience': {
                'positive': ['🧠', '🧬', '🔬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🧠', '🧬', '🔬', '💡', '📝']
            },
            'research': {
                'positive': ['🔬', '🧬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🔬', '🧬', '💡', '📝', '📚']
            },
            'studies': {
                'positive': ['🔬', '🧬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🔬', '🧬', '💡', '📝', '📚']
            },
            'science': {
                'positive': ['🔬', '🧬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🔬', '🧬', '💡', '📝', '📚']
            },
            
            # 🎓 UNIVERSITY & ACADEMIC (NOUVEAU - COMPLET)
            'stanford': {
                'positive': ['🎓', '🏛️', '📚', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🎓', '🏛️', '📚', '💡', '📝']
            },
            'university': {
                'positive': ['🎓', '🏛️', '📚', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🎓', '🏛️', '📚', '💡', '📝']
            },
            'college': {
                'positive': ['🎓', '🏛️', '📚', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🎓', '🏛️', '📚', '💡', '📝']
            },
            'academic': {
                'positive': ['🎓', '🏛️', '📚', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🎓', '🏛️', '📚', '💡', '📝']
            },
            'education': {
                'positive': ['🎓', '🏛️', '📚', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['🎓', '🏛️', '📚', '💡', '📝']
            },
            
            # 👩‍🎓 FEMALE LEARNING & FRUSTRATION (NOUVEAU - COMPLET)
            'she': {
                'positive': ['👩‍🎓', '👩‍💼', '👩‍🔬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['👩‍🎓', '👩‍💼', '👩‍🔬', '💡', '📝']
            },
            'her': {
                'positive': ['👩‍🎓', '👩‍💼', '👩‍🔬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['👩‍🎓', '👩‍💼', '👩‍🔬', '💡', '📝']
            },
            'woman': {
                'positive': ['👩‍🎓', '👩‍💼', '👩‍🔬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['👩‍🎓', '👩‍💼', '👩‍🔬', '💡', '📝']
            },
            'female': {
                'positive': ['👩‍🎓', '👩‍💼', '👩‍🔬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['👩‍🎓', '👩‍💼', '👩‍🔬', '💡', '📝']
            },
            'girl': {
                'positive': ['👩‍🎓', '👩‍💼', '👩‍🔬', '💡', '🚀', '💪', '🏆', '⭐', '🌟'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['👩‍🎓', '👩‍💼', '👩‍🔬', '💡', '📝']
            },
            'frustrating': {
                'positive': ['😤', '💪', '🚀', '🏆', '⭐', '🌟'],
                'negative': ['😤', '😠', '😡', '🤬', '💔', '😰', '😨', '😱'],
                'neutral': ['😤', '😠', '😡', '🤬']
            },
            'frustration': {
                'positive': ['😤', '💪', '🚀', '🏆', '⭐', '🌟'],
                'negative': ['😤', '😠', '😡', '🤬', '💔', '😰', '😨', '😱'],
                'neutral': ['😤', '😠', '😡', '🤬']
            },
            
            # 🚨 SERVICES D'URGENCE (EXISTANT - AMÉLIORÉ)
            'emergency': {
                'positive': ['🚨', '🚑', '🚒', '👨‍🚒', '👩‍🚒', '👮‍♂️', '👮‍♀️', '🚓', '💙', '🆘'],
                'negative': ['🚨', '🚑', '🚒', '💔', '😰', '😨', '😱', '⚠️', '🚫'],
                'neutral': ['🚨', '🚑', '🚒', '👨‍🚒', '👩‍🚒', '👮‍♂️', '👮‍♀️', '🚓', '💙']
            },
            'fire': {
                'positive': ['🔥', '🚒', '👨‍🚒', '👩‍🚒', '💪', '🏆', '💙', '🆘'],
                'negative': ['🔥', '💔', '😰', '😨', '😱', '⚠️', '🚫'],
                'neutral': ['🔥', '🚒', '👨‍🚒', '👩‍🚒', '💙']
            },
            'police': {
                'positive': ['👮‍♂️', '👮‍♀️', '🚓', '💙', '🆘', '💪', '🏆'],
                'negative': ['👮‍♂️', '👮‍♀️', '🚓', '💔', '😰', '😨', '😱'],
                'neutral': ['👮‍♂️', '👮‍♀️', '🚓', '💙']
            },
            'ambulance': {
                'positive': ['🚑', '👨‍⚕️', '👩‍⚕️', '💙', '🆘', '💪', '🏆'],
                'negative': ['🚑', '💔', '😰', '😨', '😱', '⚠️'],
                'neutral': ['🚑', '👨‍⚕️', '👩‍⚕️', '💙']
            },
            
            # 💰 FINANCE & BUSINESS (EXISTANT - AMÉLIORÉ)
            'money': {
                'positive': ['💰', '💎', '🏆', '📈', '💹', '💵', '🪙', '💲', '🏅', '🥇', '🎯', '🚀', '🔥', '💪', '🎉', '⭐', '🌟', '💫', '✨', '🎊'],
                'negative': ['📉', '💸', '❌', '💣', '💥', '🛑', '⚠️', '🚫', '💔', '😰', '😨', '😱', '😭', '😢', '😞', '😔', '😟', '😕', '🙁', '☹️'],
                'neutral': ['💳', '🏦', '📊', '📋', '📝', '📄', '📱', '💻', '📞', '📧', '📨', '📩', '📪', '📫', '📬', '📭', '📮', '📯', '📰', '📱']
            },
            'investment': {
                'positive': ['📈', '💹', '💎', '🏆', '✅', '🌟'],
                'negative': ['📉', '❌', '💸', '💣', '⚠️'],
                'neutral': ['📊', '📋', '📝', '📄', '💼']
            },
            'business': {
                'positive': ['💼', '📊', '📈', '💹', '💎', '🏆', '✅', '🌟'],
                'negative': ['📉', '❌', '💸', '💣', '⚠️'],
                'neutral': ['💼', '📊', '📋', '📝', '📄']
            },
            
            # 🚀 TECHNOLOGY & INNOVATION (EXISTANT - AMÉLIORÉ)
            'technology': {
                'positive': ['💻', '🤖', '🚀', '💡', '💪', '🏆', '⭐', '🌟', '🔥'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['💻', '🤖', '💡', '📝', '📱']
            },
            'innovation': {
                'positive': ['💡', '🚀', '💪', '🏆', '⭐', '🌟', '🔥', '✨'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['💡', '📝', '📱']
            },
            'digital': {
                'positive': ['💻', '🤖', '🚀', '💡', '💪', '🏆', '⭐', '🌟', '🔥'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['💻', '🤖', '💡', '📝', '📱']
            },
            
            # ❤️ HEALTH & FITNESS (EXISTANT - AMÉLIORÉ)
            'health': {
                'positive': ['❤️', '💪', '🏃', '🚴', '🏆', '⭐', '🌟', '🔥'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['❤️', '💪', '🏃', '🚴', '🔥']
            },
            'fitness': {
                'positive': ['💪', '🏃', '🚴', '🏋️', '🏆', '⭐', '🌟', '🔥'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['💪', '🏃', '🚴', '🏋️', '🔥']
            },
            'wellness': {
                'positive': ['❤️', '💪', '🏃', '🚴', '🏆', '⭐', '🌟', '🔥'],
                'negative': ['😵', '🤯', '😴', '💤', '😰', '😨'],
                'neutral': ['❤️', '💪', '🏃', '🚴', '🔥']
            }
        }
        
        # 🚫 MAPPING SPÉCIFIQUE POUR ÉVITER LES EMOJIS SUR LES MOTS DE LIAISON
        self.specific_keyword_mapping = {
            # 🧠 Concepts cognitifs - PRIORITÉ MAXIMALE
            'attention': '🧠',      # Au lieu de 💰 (argent)
            'thinking': '🧠',       # Cerveau/pensée
            'brain': '🧠',          # Cerveau
            'learning': '📚',       # Apprentissage
            'studying': '📚',       # Études
            'reading': '📖',        # Lecture
            'math': '🔢',           # Mathématiques
            'workout': '💪',        # Exercice
            'exercise': '💪',       # Exercice
            'physical': '💪',       # Physique
            'challenging': '💪',    # Défi
            'difficult': '💪',      # Difficile
            
            # 🧬 Neuroscience - PRIORITÉ MAXIMALE
            'acetylcholine': '🧠',  # Neurotransmetteur
            'norepinephrine': '🧠', # Neurotransmetteur
            'synapses': '🧠',       # Synapses
            'plasticity': '🧠',     # Plasticité
            'neuroscience': '🧠',   # Neuroscience
            'research': '🔬',       # Recherche
            'studies': '🔬',        # Études
            'science': '🔬',        # Science
            
            # 🎓 Université - PRIORITÉ MAXIMALE
            'stanford': '🎓',       # Université Stanford
            'university': '🎓',     # Université
            'college': '🎓',        # Collège
            'academic': '🎓',       # Académique
            'education': '🎓',      # Éducation
            
            # 👩‍🎓 Apprentissage féminin - PRIORITÉ MAXIMALE
            'she': '👩‍🎓',          # Elle (apprentissage)
            'her': '👩‍🎓',          # Elle (apprentissage)
            'woman': '👩‍🎓',        # Femme
            'female': '👩‍🎓',       # Féminin
            
            # 🚀 NOUVEAUX MOTS AJOUTÉS - PRIORITÉ MAXIMALE
            'speed': '⚡',          # Vitesse (éclair)
            'ability': '💪',        # Capacité (muscle)
            'stuff': '📦',          # Choses (boîte)
            'striking': '👊',       # Frappant (poing)
            'right': '✅',          # Correct (vérification)
            'best': '🏆',           # Meilleur (trophée)
            'growth': '🌱',         # Croissance (plante)
            'failure': '💥',        # Échec (explosion)
            'success': '🎯',        # Succès (cible)
            'brain': '🧠',          # Cerveau
            'reflexes': '⚡',        # Réflexes (éclair)
            'punch': '👊',          # Coup de poing
            'comedy': '🎭',         # Comédie (théâtre)
            'risk': '🎲',           # Risque (dé)
            'challenge': '🏔️',      # Défi (montagne)
            'learning': '📚',       # Apprentissage
            'improvement': '📈',    # Amélioration
            'motivation': '🔥',     # Motivation (feu)
            'strength': '💪',       # Force (muscle)
            'power': '⚡',          # Pouvoir (éclair)
            'fertility': '🌱',      # Fertilité (plante qui pousse)
            'development': '📈',    # Développement (graphique)
            
            # 🧬 MÉDICAL & SCIENTIFIQUE - NOUVEAU
            'sperm': '🧬',          # Spermatozoïde
            'counts': '📊',         # Comptage
            'microplastics': '🔬',  # Microplastiques
            'chemicals': '🧪',      # Produits chimiques
            'pesticides': '☠️',     # Pesticides
            'herbicides': '🌿',     # Herbicides
            'endocrine': '⚕️',      # Endocrinien
            'system': '⚙️',         # Système
            'children': '👶',       # Enfants
            'testicles': '🥜',      # Testicules
            'penis': '🍆',          # Pénis
            'plastics': '🔄',       # Plastiques
            'water': '💧',          # Eau
            'bottles': '🍾',        # Bouteilles
            'foods': '🍎',          # Aliments
            'microwave': '📡',      # Micro-ondes
            'lifestyle': '🏃',      # Mode de vie
            'sedentary': '🪑',      # Sédentaire
            'environmental': '🌍',  # Environnemental
            
            # 🧬 MÉDICAL & SCIENTIFIQUE - NOUVEAUX EMOJIS STRATÉGIQUES
            'research': '🔬',        # Recherche
            'laboratory': '🧪',      # Laboratoire
            'experiment': '⚗️',      # Expérience
            'discovery': '💡',       # Découverte
            'innovation': '🚀',      # Innovation
            'breakthrough': '💥',    # Percée
            'solution': '✅',        # Solution
            'prevention': '🛡️',     # Prévention
            'treatment': '💊',       # Traitement
            'recovery': '🔄',        # Récupération
            'wellness': '🌟',        # Bien-être
            'vitality': '💪',        # Vitalité
            'immunity': '🛡️',       # Immunité
            'metabolism': '⚡',      # Métabolisme
            'hormones': '⚕️',        # Hormones
            'genes': '🧬',           # Gènes
            'dna': '🧬',             # ADN
            'cells': '🔬',           # Cellules
            'tissue': '🔬',          # Tissus
            'organ': '❤️',           # Organe
            
            # 🧠 COGNITIF & PSYCHOLOGIQUE - NOUVEAUX
            'memory': '🧠',          # Mémoire
            'focus': '🎯',           # Concentration
            'creativity': '🎨',      # Créativité
            'intelligence': '🧠',    # Intelligence
            'wisdom': '📚',          # Sagesse
            'knowledge': '📖',       # Connaissance
            'understanding': '💭',   # Compréhension
            'insight': '💡',         # Perspicacité
            'awareness': '👁️',      # Conscience
            'mindfulness': '🧘',     # Pleine conscience
            
            # 🏃 PHYSIQUE & PERFORMANCE - NOUVEAUX
            'endurance': '🏃',       # Endurance
            'flexibility': '🧘',     # Flexibilité
            'balance': '⚖️',         # Équilibre
            'coordination': '🎯',    # Coordination
            'agility': '⚡',          # Agilité
            'speed': '🏃',           # Vitesse
            'precision': '🎯',       # Précision
            'control': '🎮',         # Contrôle
            'mastery': '🏆',         # Maîtrise
            'excellence': '⭐',      # Excellence
            
            # 🚫 Mots de liaison - BLOQUÉS (pas d'emoji)
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
        
        # 🚫 MOTS DE LIAISON À BLOQUER (pas d'emojis)
        self.linking_words = {
            'it', 'is', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'out', 'off', 'over', 'under',
            'this', 'that', 'these', 'those', 'a', 'an', 'as', 'if', 'when',
            'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
            'would', 'could', 'should', 'will', 'can', 'may', 'might', 'must',
            'have', 'has', 'had', 'do', 'does', 'did', 'be', 'been', 'being',
            'am', 'are', 'was', 'were', 'get', 'gets', 'got', 'getting'
        }
        
        # 🎯 MODIFICATEURS D'INTENSITÉ SIMPLIFIÉS
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
        
        # ✨ EMOJIS DE TRANSITION PAR TYPE
        self.transition_emojis = {
            'cut': '⚡',
            'fade': '✨',
            'zoom': '🔍',
            'slide': '➡️',
            'dissolve': '💫',
            'wipe': '🧹'
        }

        # 🚨 CORRECTION IMMÉDIATE: Mapping spécifique pour les mots problématiques
        self.critical_mapping = {
            # 🏠 FAMILLE & ENVIRONNEMENT
            'family': {
                'positive': ['👨‍👩‍👧‍👦', '❤️', '🏠', '💕', '👪'],
                'negative': ['😔', '💔', '🏚️', '😢', '😞'],
                'neutral': ['👨‍👩‍👧‍👦', '🏠', '👪']
            },
            'environment': {
                'positive': ['🌍', '🌱', '🌳', '🌿', '🏞️'],
                'negative': ['🏭', '💨', '🌫️', '☁️', '😷'],
                'neutral': ['🌍', '🏞️', '🌳']
            },
            'neighborhood': {
                'positive': ['🏘️', '🌳', '🚶', '🏠', '🌆'],
                'negative': ['🚨', '💀', '😱', '🏚️', '💔'],
                'neutral': ['🏘️', '🏠', '🌆']
            },
            
            # 🚨 CRIME & VIOLENCE
            'crime': {
                'positive': ['🚔', '🛡️', '👮', '⚖️', '🔒'],
                'negative': ['🚨', '💀', '😱', '🔪', '💣'],
                'neutral': ['🚔', '⚖️', '🔒']
            },
            'gangs': {
                'positive': ['🚔', '🛡️', '👮', '⚖️', '🔒'],
                'negative': ['💀', '😱', '🔪', '💣', '🚨'],
                'neutral': ['🚔', '⚖️', '🔒']
            },
            'drugs': {
                'positive': ['💊', '🏥', '👨‍⚕️', '🩺', '❤️'],
                'negative': ['💀', '😱', '☠️', '💉', '🚨'],
                'neutral': ['💊', '🏥', '👨‍⚕️']
            },
            
            # 🏥 SANTÉ & MÉDECINE
            'healthcare': {
                'positive': ['🏥', '👨‍⚕️', '🩺', '💊', '❤️'],
                'negative': ['😷', '💉', '🏥', '😰', '😨'],
                'neutral': ['🏥', '👨‍⚕️', '🩺']
            },
            'medical': {
                'positive': ['🏥', '👨‍⚕️', '🩺', '💊', '❤️'],
                'negative': ['😷', '💉', '🏥', '😰', '😨'],
                'neutral': ['🏥', '👨‍⚕️', '🩺']
            },
            'hurt': {
                'positive': ['🏥', '👨‍⚕️', '🩺', '💊', '❤️'],
                'negative': ['😢', '😰', '💔', '😨', '😱'],
                'neutral': ['🏥', '👨‍⚕️', '🩺']
            },
            'operation': {
                'positive': ['🏥', '👨‍⚕️', '🩺', '💊', '❤️'],
                'negative': ['😷', '💉', '🏥', '😰', '😨'],
                'neutral': ['🏥', '👨‍⚕️', '🩺']
            },
            
            # 🚒 SERVICES D'URGENCE
            'fire': {
                'positive': ['🚒', '👨‍🚒', '🔥', '🛡️', '💪'],
                'negative': ['🔥', '💀', '😱', '🏚️', '💔'],
                'neutral': ['🚒', '👨‍🚒', '🔥']
            },
            'department': {
                'positive': ['🏢', '👨‍💼', '📋', '💼', '🏛️'],
                'negative': ['😔', '😤', '😒', '😞', '😕'],
                'neutral': ['🏢', '👨‍💼', '📋']
            },
            
            # 💰 FINANCE & SOCIÉTÉ
            'money': {
                'positive': ['💰', '💵', '💎', '🏆', '⭐'],
                'negative': ['💸', '😔', '💔', '😢', '😞'],
                'neutral': ['💰', '💵', '💎']
            },
            'bankrupt': {
                'positive': ['💰', '💵', '💎', '🏆', '⭐'],
                'negative': ['💸', '😔', '💔', '😢', '😞'],
                'neutral': ['💰', '💵', '💎']
            },
            'tax': {
                'positive': ['💰', '💵', '💎', '🏆', '⭐'],
                'negative': ['💸', '😔', '💔', '😢', '😞'],
                'neutral': ['💰', '💵', '💎']
            },
            
            # 🏛️ POLITIQUE & SOCIÉTÉ
            'socialist': {
                'positive': ['🏛️', '👥', '🤝', '🌍', '❤️'],
                'negative': ['😔', '😤', '😒', '😞', '😕'],
                'neutral': ['🏛️', '👥', '🤝']
            },
            'society': {
                'positive': ['🏛️', '👥', '🤝', '🌍', '❤️'],
                'negative': ['😔', '😤', '😒', '😞', '😕'],
                'neutral': ['🏛️', '👥', '🤝']
            },
            'community': {
                'positive': ['🏛️', '👥', '🤝', '🌍', '❤️'],
                'negative': ['😔', '😤', '😒', '😞', '😕'],
                'neutral': ['🏛️', '👥', '🤝']
            }
        }

    def get_emoji_for_context(self, keyword: str, text: str = "", sentiment: str = "neutral", intensity: float = 1.0) -> str:
        """Obtient un emoji contextuel OPTIMISÉ pour un mot-clé"""
        try:
            keyword_lower = keyword.lower().strip()
            
            # 🚫 BLOQUAGE DES MOTS DE LIAISON (AMÉLIORÉ)
            if keyword_lower in self.linking_words:
                return ""
            
            # 🚨 PRIORITÉ 0: MAPPING CRITIQUE POUR LES MOTS PROBLÉMATIQUES (NOUVEAU)
            if keyword_lower in self.critical_mapping:
                if sentiment in self.critical_mapping[keyword_lower]:
                    emoji_list = self.critical_mapping[keyword_lower][sentiment]
                    if emoji_list:
                        # Sélection intelligente basée sur l'intensité
                        if intensity > 1.5:
                            # Intensité élevée: emojis plus expressifs
                            high_intensity = [e for e in emoji_list if e in ['🚨', '💀', '😱', '🔥', '💪', '🏆', '⭐', '🌟']]
                            return random.choice(high_intensity) if high_intensity else random.choice(emoji_list)
                        elif intensity < 0.5:
                            # Intensité faible: emojis plus subtils
                            low_intensity = [e for e in emoji_list if e in ['💭', '📝', '📚', '🔍', '💡', '🎯', '📊', '📈']]
                            return random.choice(low_intensity) if low_intensity else random.choice(emoji_list)
                        else:
                            # Intensité normale: sélection aléatoire
                            return random.choice(emoji_list)
            
            # 🎯 PRIORITÉ 1: MAPPING SPÉCIFIQUE DIRECT (NOUVEAU - OPTIMISÉ)
            if keyword_lower in self.specific_keyword_mapping:
                emoji = self.specific_keyword_mapping[keyword_lower]
                if emoji:
                    return emoji
            
            # 🧠 PRIORITÉ 2: MAPPING SÉMANTIQUE AVANCÉ (NOUVEAU - OPTIMISÉ)
            for category, emojis in self.semantic_mapping.items():
                if keyword_lower in category or any(kw in keyword_lower for kw in category.split('_')):
                    if sentiment in emojis:
                        emoji_list = emojis[sentiment]
                        if emoji_list:
                            # Sélection intelligente basée sur l'intensité
                            if intensity > 1.5:
                                # Intensité élevée: emojis plus expressifs
                                high_intensity_pool = ['🚀', '💥', '🔥', '⚡', '💪', '🏆', '⭐', '🌟']
                                high_intensity_available = [e for e in emoji_list if e in high_intensity_pool]
                                return random.choice(high_intensity_available if high_intensity_available else emoji_list)
                            elif intensity < 0.5:
                                # Intensité faible: emojis plus subtils
                                low_intensity_pool = ['💭', '📝', '📚', '🔍', '💡', '🎯', '📊', '📈']
                                low_intensity_available = [e for e in emoji_list if e in low_intensity_pool]
                                return random.choice(low_intensity_available if low_intensity_available else emoji_list)
                            else:
                                # Intensité normale: sélection aléatoire
                                return random.choice(emoji_list)
            
            # 🔍 PRIORITÉ 3: RECHERCHE PARTIELLE INTELLIGENTE (NOUVEAU)
            for category, emojis in self.semantic_mapping.items():
                if any(kw in keyword_lower for kw in category.split('_')):
                    if sentiment in emojis:
                        emoji_list = emojis[sentiment]
                        if emoji_list:
                            return random.choice(emoji_list)
            
            # 🎨 PRIORITÉ 4: MAPPING GÉNÉRIQUE INTELLIGENT (NOUVEAU)
            generic_mapping = {
                'positive': ['✅', '👍', '🎯', '💡', '🚀', '💪', '🏆', '⭐', '🌟', '✨', '💎', '🔥', '⚡'],
                'negative': ['❌', '👎', '😔', '😢', '😰', '😨', '💔', '💥', '💢', '😤', '😡', '🤬'],
                'neutral': ['💭', '📝', '📚', '🔍', '💡', '🎯', '📊', '📈', '📋', '📖', '🔎', '💬']
            }
            
            if sentiment in generic_mapping:
                return random.choice(generic_mapping[sentiment])
            
            # 🎯 PRIORITÉ 5: EMOJI PAR DÉFAUT INTELLIGENT (NOUVEAU)
            default_emojis = ['💡', '🎯', '📝', '🔍', '💭', '📚', '📊', '📈', '✨', '🌟']
            return random.choice(default_emojis)
            
        except Exception as e:
            print(f"❌ Erreur get_emoji_for_context: {e}")
            return ""

    def get_emoji_sequence(self, keywords: List[str], context: str = "", max_emojis: int = 3) -> List[str]:
        """Obtient une séquence cohérente d'emojis pour plusieurs mots-clés"""
        try:
            emojis = []
            used_categories = set()
            
            for keyword in keywords[:max_emojis]:
                if keyword.lower() in self.linking_words:
                    continue
                    
                emoji = self.get_emoji_for_context(keyword, context)
                if emoji:
                    emojis.append(emoji)
                    
                    # Éviter la répétition de catégories
                    for category in self.semantic_mapping:
                        if keyword.lower() in category or category in keyword.lower():
                            used_categories.add(category)
                            break
            
            return emojis
            
        except Exception as e:
            print(f"❌ Erreur get_emoji_sequence: {e}")
            return []

    def get_transition_emoji(self, transition_type: str = "cut") -> str:
        """Obtient un emoji selon le type de transition"""
        return self.transition_emojis.get(transition_type, "✨")

    def is_linking_word(self, word: str) -> bool:
        """Vérifie si un mot est un mot de liaison (bloqué)"""
        return word.lower() in self.linking_words

# Instance globale
contextual_emojis_complete = ContextualEmojiSystemComplete() 