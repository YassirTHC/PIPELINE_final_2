#!/usr/bin/env python3
"""
Analyseur Contextuel Synchrone
Interface synchrone pour l'analyse contextuelle des vidéos
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class ContextAnalysis:
    """Résultat de l'analyse contextuelle"""
    main_theme: str
    key_topics: List[str]
    sentiment: float
    complexity: float
    keywords: List[str]
    context_score: float
    sub_themes: List[str]
    overall_tone: str
    target_audience: str

class SyncContextAnalyzer:
    """Analyseur contextuel synchrone pour l'intégration avec le pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Analyseur contextuel synchrone initialisé")
    
    def analyze_context(self, transcript_text: str) -> ContextAnalysis:
        """Analyse contextuelle synchrone du transcript"""
        try:
            self.logger.info(f"Analyse contextuelle synchrone: {len(transcript_text)} caractères")
            
            # Analyse basique du contexte
            words = transcript_text.lower().split()
            
            # Détection du thème principal
            theme_keywords = {
                'technology': ['ai', 'artificial', 'intelligence', 'digital', 'tech', 'innovation'],
                'business': ['business', 'entrepreneur', 'success', 'growth', 'strategy'],
                'education': ['learn', 'education', 'knowledge', 'study', 'teaching'],
                'health': ['health', 'fitness', 'wellness', 'medical', 'exercise'],
                'science': ['science', 'research', 'discovery', 'experiment', 'analysis']
            }
            
            main_theme = 'general'
            max_score = 0
            
            for theme, keywords in theme_keywords.items():
                score = sum(1 for word in words if word in keywords)
                if score > max_score:
                    max_score = score
                    main_theme = theme
            
            # Extraction des mots-clés
            keywords = [word for word in words if len(word) > 3 and word.isalpha()][:10]
            
            # Analyse du sentiment (basique)
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'positive']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'problem']
            
            sentiment = 0.0
            for word in words:
                if word in positive_words:
                    sentiment += 0.1
                elif word in negative_words:
                    sentiment -= 0.1
            
            sentiment = max(-1.0, min(1.0, sentiment))
            
            # Calcul de la complexité
            complexity = min(1.0, len(set(words)) / len(words) if words else 0.0)
            
            # Sujets clés
            key_topics = list(set(keywords[:5]))
            
            # Thèmes secondaires
            sub_themes = [theme for theme in theme_keywords.keys() if theme != main_theme][:3]
            
            # Ton général
            if sentiment > 0.3:
                overall_tone = 'positive'
            elif sentiment < -0.3:
                overall_tone = 'negative'
            else:
                overall_tone = 'neutral'
            
            # Audience cible
            if 'business' in main_theme or 'entrepreneur' in words:
                target_audience = 'business_professionals'
            elif 'education' in main_theme or 'learn' in words:
                target_audience = 'students_learners'
            elif 'technology' in main_theme:
                target_audience = 'tech_enthusiasts'
            else:
                target_audience = 'general_public'
            
            # Score de contexte
            context_score = (len(keywords) / 10.0 + abs(sentiment) + complexity) / 3.0
            
            result = ContextAnalysis(
                main_theme=main_theme,
                key_topics=key_topics,
                sentiment=sentiment,
                complexity=complexity,
                keywords=keywords,
                context_score=context_score,
                sub_themes=sub_themes,
                overall_tone=overall_tone,
                target_audience=target_audience
            )
            
            self.logger.info(f"Analyse contextuelle terminée: {main_theme} (score: {context_score:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse contextuelle: {e}")
            # Retourner une analyse par défaut
            return ContextAnalysis(
                main_theme='general',
                key_topics=['general'],
                sentiment=0.0,
                complexity=0.5,
                keywords=['content', 'video', 'general'],
                context_score=0.5,
                sub_themes=['general'],
                overall_tone='neutral',
                target_audience='general_public'
            )
    
    def analyze_segment(self, text: str, start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyse d'un segment individuel"""
        try:
            self.logger.info(f"Analyse du segment: {text[:50]}...")
            
            # Utiliser l'analyse contextuelle
            context = self.analyze_context(text)
            
            return {
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'main_theme': context.main_theme,
                'key_topics': context.key_topics,
                'sentiment': context.sentiment,
                'complexity': context.complexity,
                'keywords': context.keywords,
                'context_score': context.context_score
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse du segment: {e}")
            return {
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'main_theme': 'general',
                'key_topics': ['general'],
                'sentiment': 0.0,
                'complexity': 0.5,
                'keywords': ['content'],
                'context_score': 0.5
            }
    
    def get_global_analysis(self) -> Dict[str, Any]:
        """Récupère l'analyse globale (interface de compatibilité)"""
        return {
            'main_theme': 'general',
            'key_topics': ['general'],
            'sentiment': 0.0,
            'complexity': 0.5,
            'keywords': ['content', 'video'],
            'context_score': 0.5,
            'sub_themes': ['general'],
            'overall_tone': 'neutral',
            'target_audience': 'general_public'
        }

# Instance globale pour compatibilité
sync_context_analyzer = SyncContextAnalyzer() 