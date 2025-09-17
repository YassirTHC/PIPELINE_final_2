"""
Système de Scoring Avancé et Adaptatif pour B-rolls
Scoring multi-critères avec adaptation automatique par domaine
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ScoringWeights:
    """Poids adaptatifs pour le scoring"""
    semantic_similarity: float = 0.45
    visual_quality: float = 0.25
    diversity: float = 0.15
    timeline_compatibility: float = 0.15

@dataclass
class QualityThresholds:
    """Seuils de qualité adaptatifs par domaine"""
    min_quality: float = 0.7
    min_semantic: float = 0.6
    min_diversity: float = 0.5
    min_timeline: float = 0.6

class AdaptiveScoringSystem:
    """Système de scoring adaptatif par domaine"""
    
    def __init__(self):
        # Configuration des domaines avec leurs spécificités
        self.domain_configs = {
            "neuroscience": {
                "weights": ScoringWeights(0.55, 0.20, 0.15, 0.10),  # Privilégier la pertinence sémantique
                "thresholds": QualityThresholds(0.4, 0.3, 0.4, 0.5),  # Seuils plus permissifs
                "description": "Domaine technique spécialisé - privilégier la pertinence conceptuelle"
            },
            "science": {
                "weights": ScoringWeights(0.50, 0.25, 0.15, 0.10),
                "thresholds": QualityThresholds(0.5, 0.4, 0.4, 0.5),
                "description": "Domaine scientifique - équilibre pertinence/qualité"
            },
            "technology": {
                "weights": ScoringWeights(0.45, 0.30, 0.15, 0.10),
                "thresholds": QualityThresholds(0.6, 0.5, 0.5, 0.6),
                "description": "Domaine technologique - qualité visuelle importante"
            },
            "business": {
                "weights": ScoringWeights(0.40, 0.30, 0.20, 0.10),
                "thresholds": QualityThresholds(0.7, 0.6, 0.6, 0.7),
                "description": "Domaine business - qualité professionnelle requise"
            },
            "lifestyle": {
                "weights": ScoringWeights(0.35, 0.35, 0.20, 0.10),
                "thresholds": QualityThresholds(0.8, 0.7, 0.7, 0.8),
                "description": "Domaine lifestyle - qualité visuelle et esthétique élevées"
            },
            "education": {
                "weights": ScoringWeights(0.45, 0.25, 0.20, 0.10),
                "thresholds": QualityThresholds(0.6, 0.5, 0.5, 0.6),
                "description": "Domaine éducatif - équilibre pertinence/qualité"
            },
            "general": {
                "weights": ScoringWeights(0.45, 0.25, 0.15, 0.15),
                "thresholds": QualityThresholds(0.7, 0.6, 0.5, 0.6),
                "description": "Domaine général - configuration standard"
            }
        }
        
        # Bonus de qualité par source
        self.source_quality_bonus = {
            'pexels': 1.0,       # Qualité professionnelle
            'unsplash': 1.0,     # Qualité professionnelle
            'pixabay': 0.9,      # Bonne qualité
            'giphy': 0.8,        # Contenu viral
            'archive_org': 0.7,  # Contenu historique
            'wikimedia': 0.8,    # Contenu éducatif
            'nasa': 0.9,         # Contenu scientifique
            'wellcome': 0.8,     # Contenu médical
            'local_cache': 0.6   # Cache local
        }
        
        # Pénalités de diversité
        self.diversity_penalties = {
            'same_source': 0.3,      # Même source
            'same_category': 0.2,    # Même catégorie
            'similar_visual': 0.4,   # Visuellement similaire
            'recent_use': 0.5        # Utilisation récente
        }
    
    def get_adaptive_weights(self, domain: str, context_complexity: str = "medium") -> ScoringWeights:
        """Récupère les poids adaptatifs pour un domaine donné"""
        try:
            # Normaliser le domaine
            domain = domain.lower().strip()
            
            # Récupérer la configuration du domaine
            if domain in self.domain_configs:
                base_weights = self.domain_configs[domain]["weights"]
                logger.info(f"Poids adaptatifs pour le domaine '{domain}': {base_weights}")
            else:
                base_weights = self.domain_configs["general"]["weights"]
                logger.info(f"Domaine '{domain}' non reconnu, utilisation des poids généraux")
            
            # Ajustement par complexité du contexte
            if context_complexity == "high":
                # Pour les contextes complexes, privilégier la pertinence sémantique
                adjusted_weights = ScoringWeights(
                    semantic_similarity=min(base_weights.semantic_similarity * 1.2, 0.7),
                    visual_quality=base_weights.visual_quality * 0.8,
                    diversity=base_weights.diversity,
                    timeline_compatibility=base_weights.timeline_compatibility * 0.8
                )
                logger.info(f"Ajustement pour contexte complexe: {adjusted_weights}")
                return adjusted_weights
            
            elif context_complexity == "low":
                # Pour les contextes simples, privilégier la qualité visuelle
                adjusted_weights = ScoringWeights(
                    semantic_similarity=base_weights.semantic_similarity * 0.8,
                    visual_quality=min(base_weights.visual_quality * 1.2, 0.5),
                    diversity=base_weights.diversity,
                    timeline_compatibility=base_weights.timeline_compatibility
                )
                logger.info(f"Ajustement pour contexte simple: {adjusted_weights}")
                return adjusted_weights
            
            return base_weights
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des poids adaptatifs: {e}")
            return self.domain_configs["general"]["weights"]
    
    def get_adaptive_thresholds(self, domain: str, context_complexity: str = "medium") -> QualityThresholds:
        """Récupère les seuils adaptatifs pour un domaine donné"""
        try:
            # Normaliser le domaine
            domain = domain.lower().strip()
            
            # Récupérer la configuration du domaine
            if domain in self.domain_configs:
                base_thresholds = self.domain_configs[domain]["thresholds"]
                logger.info(f"Seuils adaptatifs pour le domaine '{domain}': {base_thresholds}")
            else:
                base_thresholds = self.domain_configs["general"]["thresholds"]
                logger.info(f"Domaine '{domain}' non reconnu, utilisation des seuils généraux")
            
            # Ajustement par complexité du contexte
            if context_complexity == "high":
                # Pour les contextes complexes, réduire les seuils
                adjusted_thresholds = QualityThresholds(
                    min_quality=base_thresholds.min_quality * 0.8,
                    min_semantic=base_thresholds.min_semantic * 0.7,
                    min_diversity=base_thresholds.min_diversity * 0.8,
                    min_timeline=base_thresholds.min_timeline * 0.8
                )
                logger.info(f"Seuils ajustés pour contexte complexe: {adjusted_thresholds}")
                return adjusted_thresholds
            
            elif context_complexity == "low":
                # Pour les contextes simples, augmenter les seuils
                adjusted_thresholds = QualityThresholds(
                    min_quality=min(base_thresholds.min_quality * 1.1, 0.9),
                    min_semantic=min(base_thresholds.min_semantic * 1.1, 0.8),
                    min_diversity=min(base_thresholds.min_diversity * 1.1, 0.8),
                    min_timeline=min(base_thresholds.min_timeline * 1.1, 0.8)
                )
                logger.info(f"Seuils ajustés pour contexte simple: {adjusted_thresholds}")
                return adjusted_thresholds
            
            return base_thresholds
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des seuils adaptatifs: {e}")
            return self.domain_configs["general"]["thresholds"]
    
    def calculate_semantic_score(self, candidate: Dict, context: Dict) -> float:
        """Calcule le score de similarité sémantique"""
        try:
            score = 0.0
            
            # Correspondance des mots-clés
            candidate_keywords = candidate.get("keywords", [])
            context_keywords = context.get("keywords", [])
            
            if candidate_keywords and context_keywords:
                # Calcul de la correspondance exacte
                exact_matches = sum(1 for kw in context_keywords if kw.lower() in [ck.lower() for ck in candidate_keywords])
                if context_keywords:
                    exact_match_ratio = exact_matches / len(context_keywords)
                    score += exact_match_ratio * 0.6
                
                # Calcul de la correspondance partielle (mots similaires)
                partial_matches = 0
                for context_kw in context_keywords:
                    for candidate_kw in candidate_keywords:
                        if (context_kw.lower() in candidate_kw.lower() or 
                            candidate_kw.lower() in context_kw.lower()):
                            partial_matches += 1
                            break
                
                if context_keywords:
                    partial_match_ratio = partial_matches / len(context_keywords)
                    score += partial_match_ratio * 0.4
            
            # Bonus pour la source
            source = candidate.get("source", "unknown")
            if source in self.source_quality_bonus:
                score *= self.source_quality_bonus[source]
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score sémantique: {e}")
            return 0.5
    
    def calculate_visual_quality_score(self, candidate: Dict) -> float:
        """Calcule le score de qualité visuelle"""
        try:
            score = 0.0
            
            # Score de résolution
            resolution = candidate.get("resolution", [0, 0])
            if resolution and len(resolution) >= 2:
                width, height = resolution[0], resolution[1]
                if width >= 1920 and height >= 1080:
                    score += 0.4  # Full HD
                elif width >= 1280 and height >= 720:
                    score += 0.3  # HD
                elif width >= 854 and height >= 480:
                    score += 0.2  # SD
                else:
                    score += 0.1  # Basse résolution
            
            # Score de stabilité (FPS)
            fps = candidate.get("fps", 0)
            if fps >= 30:
                score += 0.3
            elif fps >= 24:
                score += 0.2
            elif fps >= 15:
                score += 0.1
            
            # Score de durée
            duration = candidate.get("duration", 0)
            if 2.0 <= duration <= 6.0:
                score += 0.2  # Durée optimale
            elif 1.0 <= duration <= 8.0:
                score += 0.1  # Durée acceptable
            
            # Bonus pour la source
            source = candidate.get("source", "unknown")
            if source in self.source_quality_bonus:
                score *= self.source_quality_bonus[source]
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score de qualité visuelle: {e}")
            return 0.5
    
    def calculate_diversity_score(self, candidate: Dict, context: Dict) -> float:
        """Calcule le score de diversité"""
        try:
            score = 1.0  # Score de base
            
            # Pénalité pour la même source
            current_source = candidate.get("source", "unknown")
            used_sources = context.get("used_sources", [])
            if current_source in used_sources:
                score *= self.diversity_penalties["same_source"]
            
            # Pénalité pour la même catégorie
            current_category = candidate.get("category", "unknown")
            used_categories = context.get("used_categories", [])
            if current_category in used_categories:
                score *= self.diversity_penalties["same_category"]
            
            # Pénalité pour utilisation récente
            recent_use = context.get("recent_use", {})
            candidate_id = candidate.get("id", "unknown")
            if candidate_id in recent_use:
                score *= self.diversity_penalties["recent_use"]
            
            return max(score, 0.1)  # Minimum 10%
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score de diversité: {e}")
            return 0.5
    
    def calculate_timeline_compatibility(self, candidate: Dict, context: Dict) -> float:
        """Calcule la compatibilité temporelle"""
        try:
            score = 1.0
            
            # Durée du segment
            segment_duration = context.get("segment_duration", 0)
            candidate_duration = candidate.get("duration", 0)
            
            if segment_duration > 0 and candidate_duration > 0:
                # Ratio optimal : B-roll = 60% du segment
                optimal_ratio = 0.6
                actual_ratio = candidate_duration / segment_duration
                
                # Score basé sur la proximité du ratio optimal
                ratio_diff = abs(actual_ratio - optimal_ratio)
                if ratio_diff <= 0.1:
                    score = 1.0  # Ratio parfait
                elif ratio_diff <= 0.2:
                    score = 0.8  # Ratio bon
                elif ratio_diff <= 0.3:
                    score = 0.6  # Ratio acceptable
                else:
                    score = 0.4  # Ratio faible
            
            return score
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la compatibilité temporelle: {e}")
            return 0.5
    
    def calculate_final_score(self, candidate: Dict, context: Dict, domain: str = "general") -> Dict[str, Any]:
        """Calcule le score final avec tous les critères"""
        try:
            # Récupérer les poids et seuils adaptatifs
            context_complexity = context.get("complexity", "medium")
            weights = self.get_adaptive_weights(domain, context_complexity)
            thresholds = self.get_adaptive_thresholds(domain, context_complexity)
            
            # Calculer les scores individuels
            semantic_score = self.calculate_semantic_score(candidate, context)
            visual_score = self.calculate_visual_quality_score(candidate)
            diversity_score = self.calculate_diversity_score(candidate, context)
            timeline_score = self.calculate_timeline_compatibility(candidate, context)
            
            # Calculer le score final pondéré
            final_score = (
                semantic_score * weights.semantic_similarity +
                visual_score * weights.visual_quality +
                diversity_score * weights.diversity +
                timeline_score * weights.timeline_compatibility
            )
            
            # Vérifier si le candidat passe les seuils
            passes_thresholds = (
                semantic_score >= thresholds.min_semantic and
                visual_score >= thresholds.min_quality and
                diversity_score >= thresholds.min_diversity and
                timeline_score >= thresholds.min_timeline
            )
            
            # Générer la raison de sélection
            selection_reason = self._generate_selection_reason(
                semantic_score, visual_score, diversity_score, timeline_score,
                weights, domain
            )
            
            result = {
                "final_score": min(final_score, 1.0),
                "semantic_score": semantic_score,
                "visual_score": visual_score,
                "diversity_score": diversity_score,
                "timeline_score": timeline_score,
                "context_relevance": semantic_score,  # Alias pour compatibilité
                "passes_thresholds": passes_thresholds,
                "selection_reason": selection_reason,
                "weights_used": {
                    "semantic": weights.semantic_similarity,
                    "visual": weights.visual_quality,
                    "diversity": weights.diversity,
                    "timeline": weights.timeline_compatibility
                },
                "thresholds_used": {
                    "min_quality": thresholds.min_quality,
                    "min_semantic": thresholds.min_semantic,
                    "min_diversity": thresholds.min_diversity,
                    "min_timeline": thresholds.min_timeline
                }
            }
            
            logger.info(f"Score final calculé: {final_score:.3f} (domaine: {domain})")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score final: {e}")
            return {
                "final_score": 0.0,
                "passes_thresholds": False,
                "selection_reason": f"Erreur de calcul: {e}"
            }
    
    def _generate_selection_reason(self, semantic: float, visual: float, 
                                 diversity: float, timeline: float,
                                 weights: ScoringWeights, domain: str) -> str:
        """Génère la raison de sélection basée sur les scores"""
        try:
            reasons = []
            
            # Raison principale basée sur le score le plus élevé
            scores = [
                ("sémantique", semantic, weights.semantic_similarity),
                ("visuelle", visual, weights.visual_quality),
                ("diversité", diversity, weights.diversity),
                ("temporelle", timeline, weights.timeline_compatibility)
            ]
            
            # Trier par score pondéré
            scores.sort(key=lambda x: x[1] * x[2], reverse=True)
            
            # Ajouter les raisons principales
            for i, (criterion, score, weight) in enumerate(scores[:2]):
                if score > 0.7:
                    reasons.append(f"Excellent score {criterion} ({score:.1%})")
                elif score > 0.5:
                    reasons.append(f"Bon score {criterion} ({score:.1%})")
                elif score > 0.3:
                    reasons.append(f"Score {criterion} acceptable ({score:.1%})")
            
            # Ajouter des informations sur le domaine
            if domain in self.domain_configs:
                domain_desc = self.domain_configs[domain]["description"]
                reasons.append(f"Domaine: {domain_desc}")
            
            return " | ".join(reasons) if reasons else "Score global acceptable"
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la raison: {e}")
            return "Raison non disponible"

# Instance globale pour utilisation dans le pipeline
adaptive_scoring = AdaptiveScoringSystem()

def calculate_adaptive_scoring(candidate: Dict, context: Dict, domain: str = "general") -> Dict[str, Any]:
    """Fonction utilitaire pour le scoring adaptatif"""
    return adaptive_scoring.calculate_final_score(candidate, context, domain)

def get_adaptive_weights(domain: str, context_complexity: str = "medium") -> ScoringWeights:
    """Fonction utilitaire pour récupérer les poids adaptatifs"""
    return adaptive_scoring.get_adaptive_weights(domain, context_complexity)

def get_adaptive_thresholds(domain: str, context_complexity: str = "medium") -> QualityThresholds:
    """Fonction utilitaire pour récupérer les seuils adaptatifs"""
    return adaptive_scoring.get_adaptive_thresholds(domain, context_complexity) 