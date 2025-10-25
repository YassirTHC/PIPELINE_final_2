# -*- coding: utf-8 -*-
# ðŸŽ¯ POST-PROCESSING DES MOTS-CLÃ‰S B-ROLL - FILTRAGE + CATÃ‰GORISATION + DÃ‰-DUP
# Pipeline de nettoyage et optimisation des mots-clÃ©s pour la recherche B-roll

import re
import logging
from typing import Dict, List, Tuple, Any, Set
from collections import OrderedDict, Counter
from dataclasses import dataclass

# Import de l'optimiseur de diversitÃ©
try:
    from keyword_diversity_optimizer import optimize_broll_keywords_diversity
    DIVERSITY_OPTIMIZER_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("âœ… Optimiseur de diversitÃ© disponible")
except ImportError:
    DIVERSITY_OPTIMIZER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("âš ï¸ Optimiseur de diversitÃ© non disponible - utilisation du mode basique")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KeywordQuality:
    """MÃ©tadonnÃ©es de qualitÃ© pour un mot-clÃ©"""
    keyword: str
    length: int
    is_visual: bool
    category: str
    confidence: float
    search_ready: bool

class KeywordProcessor:
    """Processeur de mots-clÃ©s avec filtrage et catÃ©gorisation intelligente"""
    
    def __init__(self):
        # CatÃ©gories de mots-clÃ©s visuels
        self.visual_categories = {
            'actions': ['running', 'walking', 'talking', 'smiling', 'working', 'studying', 'cooking', 'driving'],
            'objects': ['computer', 'phone', 'book', 'car', 'house', 'tree', 'flower', 'food', 'clothes'],
            'places': ['office', 'home', 'park', 'school', 'hospital', 'restaurant', 'street', 'beach'],
            'people': ['doctor', 'teacher', 'student', 'worker', 'family', 'children', 'elderly', 'professional'],
            'emotions': ['happy', 'sad', 'excited', 'calm', 'focused', 'relaxed', 'energetic', 'peaceful'],
            'abstract': ['success', 'growth', 'change', 'improvement', 'development', 'learning', 'healing']
        }
        
        # Mots-clÃ©s non-visuels Ã  filtrer
        self.non_visual_keywords = {
            'abstract_concepts': ['success', 'failure', 'happiness', 'sadness', 'love', 'hate', 'hope', 'fear'],
            'time_words': ['always', 'never', 'sometimes', 'often', 'rarely', 'today', 'yesterday', 'tomorrow'],
            'intensity_words': ['very', 'extremely', 'slightly', 'completely', 'totally', 'partially'],
            'logical_words': ['because', 'therefore', 'however', 'although', 'unless', 'if', 'then', 'else']
        }
        
        # Patterns de nettoyage
        self.cleaning_patterns = [
            (r'[^a-zA-Z0-9\s\-]', ''),  # Supprimer caractÃ¨res spÃ©ciaux
            (r'\s+', ' '),              # Normaliser espaces
            (r'^\s+|\s+$', ''),         # Supprimer espaces dÃ©but/fin
        ]
        
        # Seuils de qualitÃ©
        self.min_length = 3
        self.max_length = 20
        self.min_confidence = 0.6
    
    def clean_keywords(self, raw_keywords: List[str]) -> List[str]:
        """
        Nettoyage et normalisation des mots-clÃ©s
        """
        cleaned = []
        
        for keyword in raw_keywords:
            if not isinstance(keyword, str):
                continue
                
            # Application des patterns de nettoyage
            cleaned_keyword = keyword
            for pattern, replacement in self.cleaning_patterns:
                cleaned_keyword = re.sub(pattern, replacement, cleaned_keyword)
            
            # Validation de la longueur
            if len(cleaned_keyword) < self.min_length or len(cleaned_keyword) > self.max_length:
                continue
            
            # Supprimer les mots vides
            if cleaned_keyword.lower() in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']:
                continue
            
            cleaned.append(cleaned_keyword.lower())
        
        # DÃ©-duplication en prÃ©servant l'ordre
        unique_keywords = list(OrderedDict.fromkeys(cleaned))
        
        logger.info(f"ðŸ§¹ Mots-clÃ©s nettoyÃ©s: {len(raw_keywords)} â†’ {len(unique_keywords)}")
        return unique_keywords
    
    def categorize_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """
        CatÃ©gorisation automatique des mots-clÃ©s
        """
        categorized = {category: [] for category in self.visual_categories.keys()}
        categorized['uncategorized'] = []
        
        for keyword in keywords:
            categorized_flag = False
            
            # VÃ©rifier chaque catÃ©gorie
            for category, examples in self.visual_categories.items():
                # VÃ©rifier si le mot-clÃ© correspond Ã  la catÃ©gorie
                if self._matches_category(keyword, examples):
                    categorized[category].append(keyword)
                    categorized_flag = True
                    break
            
            # Si aucune catÃ©gorie trouvÃ©e
            if not categorized_flag:
                categorized['uncategorized'].append(keyword)
        
        # Log des rÃ©sultats
        for category, words in categorized.items():
            if words:
                logger.info(f"ðŸ·ï¸ {category}: {len(words)} mots-clÃ©s")
        
        return categorized
    
    def _matches_category(self, keyword: str, examples: List[str]) -> bool:
        """
        VÃ©rifie si un mot-clÃ© correspond Ã  une catÃ©gorie
        """
        keyword_lower = keyword.lower()
        
        # Correspondance exacte
        if keyword_lower in [ex.lower() for ex in examples]:
            return True
        
        # Correspondance partielle (suffixe/prÃ©fixe)
        for example in examples:
            example_lower = example.lower()
            if (keyword_lower.endswith(example_lower) or 
                keyword_lower.startswith(example_lower) or
                example_lower in keyword_lower):
                return True
        
        # Correspondance sÃ©mantique basique
        if any(word in keyword_lower for word in ['ing', 'ed', 'er', 'tion', 'sion', 'ness']):
            # Mots avec suffixes verbaux/nominaux
            return True
        
        return False
    
    def filter_visual_keywords(self, keywords: List[str]) -> List[str]:
        """
        Filtrage pour ne garder que les mots-clÃ©s visuellement reprÃ©sentables
        """
        visual_keywords = []
        
        for keyword in keywords:
            # VÃ©rifier si c'est un concept abstrait
            is_abstract = any(keyword in words for words in self.non_visual_keywords.values())
            
            # VÃ©rifier si c'est visuellement reprÃ©sentable
            is_visual = any(keyword in words for words in self.visual_categories.values())
            
            if is_visual and not is_abstract:
                visual_keywords.append(keyword)
            elif not is_abstract and len(keyword) > 4:  # Mots longs non-abstraits
                visual_keywords.append(keyword)
        
        logger.info(f"ðŸŽ¨ Mots-clÃ©s visuels filtrÃ©s: {len(keywords)} â†’ {len(visual_keywords)}")
        return visual_keywords
    
    def generate_search_queries(self, keywords: List[str], max_queries: int = 12) -> List[str]:
        """
        GÃ©nÃ©ration de requÃªtes de recherche optimisÃ©es pour les APIs B-roll
        """
        search_queries = []
        
        # RequÃªtes simples (1-2 mots)
        for keyword in keywords[:max_queries//2]:
            if len(keyword.split()) <= 2:
                search_queries.append(keyword)
        
        # RequÃªtes composÃ©es (2-3 mots)
        if len(search_queries) < max_queries:
            for i, keyword1 in enumerate(keywords):
                if len(search_queries) >= max_queries:
                    break
                    
                for keyword2 in keywords[i+1:]:
                    if len(search_queries) >= max_queries:
                        break
                    
                    combined = f"{keyword1} {keyword2}"
                    if len(combined) <= 25:  # Limite de longueur pour les APIs
                        search_queries.append(combined)
        
        # Limiter le nombre de requÃªtes
        final_queries = search_queries[:max_queries]
        
        logger.info(f"ðŸ” RequÃªtes de recherche gÃ©nÃ©rÃ©es: {len(final_queries)}")
        return final_queries
    
    def assess_keyword_quality(self, keywords: List[str]) -> List[KeywordQuality]:
        """
        Ã‰valuation de la qualitÃ© de chaque mot-clÃ©
        """
        quality_scores = []
        
        for keyword in keywords:
            # Longueur
            length = len(keyword)
            
            # VisibilitÃ©
            is_visual = any(keyword in words for words in self.visual_categories.values())
            
            # CatÃ©gorie
            category = self._get_keyword_category(keyword)
            
            # Confiance (basÃ©e sur la longueur et la visibilitÃ©)
            confidence = min(1.0, (length / 10) + (0.5 if is_visual else 0.0))
            
            # PrÃªt pour la recherche
            search_ready = length >= 3 and confidence >= self.min_confidence
            
            quality = KeywordQuality(
                keyword=keyword,
                length=length,
                is_visual=is_visual,
                category=category,
                confidence=confidence,
                search_ready=search_ready
            )
            
            quality_scores.append(quality)
        
        return quality_scores
    
    def _get_keyword_category(self, keyword: str) -> str:
        """
        DÃ©termine la catÃ©gorie d'un mot-clÃ©
        """
        for category, examples in self.visual_categories.items():
            if self._matches_category(keyword, examples):
                return category
        return 'uncategorized'
    
    def optimize_for_broll(self, keywords: List[str], target_count: int = 10) -> Dict[str, Any]:
        """
        Optimisation complÃ¨te des mots-clÃ©s pour la recherche B-roll avec diversitÃ©
        """
        logger.info(f"ðŸš€ Optimisation B-roll pour {len(keywords)} mots-clÃ©s vers {target_count} cibles")
        
        # 1. Nettoyage
        cleaned = self.clean_keywords(keywords)
        
        # 2. Filtrage visuel
        visual = self.filter_visual_keywords(cleaned)
        
        # 3. OPTIMISATION DE DIVERSITÃ‰ (NOUVEAU)
        if DIVERSITY_OPTIMIZER_AVAILABLE and len(visual) > target_count:
            logger.info("ðŸŽ¯ Application de l'optimiseur de diversitÃ©")
            try:
                diversity_result = optimize_broll_keywords_diversity(visual, target_count)
                
                if diversity_result.get('optimization_applied', False):
                    # Utiliser les mots-clÃ©s optimisÃ©s par diversitÃ©
                    optimal_keywords = diversity_result['keywords']
                    search_queries = diversity_result['search_queries']
                    categorized = diversity_result['categories']
                    diversity_metrics = diversity_result['metrics']
                    
                    logger.info(f"âœ… DiversitÃ© appliquÃ©e: {diversity_metrics.get('categories_covered', 0)} catÃ©gories couvertes")
                    logger.info(f"ðŸ“Š Score de diversitÃ©: {diversity_metrics.get('diversity_score', 0):.2f}")
                else:
                    # Fallback vers l'ancienne mÃ©thode
                    logger.warning("âš ï¸ Optimiseur de diversitÃ© Ã©chouÃ©, fallback vers mÃ©thode basique")
                    optimal_keywords = self._select_optimal_keywords(quality_scores, target_count)
                    search_queries = self.generate_search_queries(optimal_keywords)
                    categorized = self.categorize_keywords(visual)
                    diversity_metrics = {}
            except Exception as e:
                logger.error(f"âŒ Erreur optimiseur de diversitÃ©: {e}")
                # Fallback vers l'ancienne mÃ©thode
                quality_scores = self.assess_keyword_quality(visual)
                optimal_keywords = self._select_optimal_keywords(quality_scores, target_count)
                search_queries = self.generate_search_queries(optimal_keywords)
                categorized = self.categorize_keywords(visual)
                diversity_metrics = {}
        else:
            # MÃ©thode basique si l'optimiseur n'est pas disponible
            logger.info("ðŸ”„ Utilisation de la mÃ©thode d'optimisation basique")
            quality_scores = self.assess_keyword_quality(visual)
            optimal_keywords = self._select_optimal_keywords(quality_scores, target_count)
            search_queries = self.generate_search_queries(optimal_keywords)
            categorized = self.categorize_keywords(visual)
            diversity_metrics = {}
        
        # 4. Statistiques
        # S'assurer que quality_scores est dÃ©fini
        if 'quality_scores' not in locals():
            quality_scores = self.assess_keyword_quality(optimal_keywords)
        
        stats = {
            'total_input': len(keywords),
            'cleaned': len(cleaned),
            'visual': len(visual),
            'optimal': len(optimal_keywords),
            'search_queries': len(search_queries),
            'categories': {cat: len(words) for cat, words in categorized.items() if words},
            'quality_distribution': {
                'high': len([q for q in quality_scores if q.confidence >= 0.8]),
                'medium': len([q for q in quality_scores if 0.6 <= q.confidence < 0.8]),
                'low': len([q for q in quality_scores if q.confidence < 0.6])
            }
        }
        
        # Ajouter les mÃ©triques de diversitÃ© si disponibles
        if diversity_metrics:
            stats['diversity_metrics'] = diversity_metrics
        
        result = {
            'keywords': optimal_keywords,
            'search_queries': search_queries,
            'categorized': categorized,
            'quality_scores': quality_scores if 'quality_scores' in locals() else [],
            'statistics': stats,
            'diversity_optimized': DIVERSITY_OPTIMIZER_AVAILABLE
        }
        
        logger.info(f"âœ… Optimisation terminÃ©e: {stats['optimal']} mots-clÃ©s optimaux")
        return result
    
    def _select_optimal_keywords(self, quality_scores: List[KeywordQuality], target_count: int) -> List[str]:
        """
        SÃ©lection optimale des mots-clÃ©s basÃ©e sur la qualitÃ©
        """
        # Trier par confiance dÃ©croissante
        sorted_keywords = sorted(quality_scores, key=lambda x: x.confidence, reverse=True)
        
        # SÃ©lectionner les meilleurs
        selected = []
        category_counts = Counter()
        
        for quality in sorted_keywords:
            if len(selected) >= target_count:
                break
            
            # VÃ©rifier la diversitÃ© des catÃ©gories
            if category_counts[quality.category] < target_count // len(self.visual_categories):
                selected.append(quality.keyword)
                category_counts[quality.category] += 1
            elif quality.confidence >= 0.9:  # Exception pour les mots-clÃ©s de trÃ¨s haute qualitÃ©
                selected.append(quality.keyword)
        
        return selected

# === INSTANCE GLOBALE ===
keyword_processor = KeywordProcessor()

# === FONCTIONS UTILITAIRES ===
def clean_keywords(keywords: List[str]) -> List[str]:
    """Nettoyage des mots-clÃ©s"""
    return keyword_processor.clean_keywords(keywords)

def filter_visual_keywords(keywords: List[str]) -> List[str]:
    """Filtrage des mots-clÃ©s visuels"""
    return keyword_processor.filter_visual_keywords(keywords)

def categorize_keywords(keywords: List[str]) -> Dict[str, List[str]]:
    """CatÃ©gorisation des mots-clÃ©s"""
    return keyword_processor.categorize_keywords(keywords)

def generate_search_queries(keywords: List[str], max_queries: int = 12) -> List[str]:
    """GÃ©nÃ©ration de requÃªtes de recherche"""
    return keyword_processor.generate_search_queries(keywords, max_queries)

def optimize_for_broll(keywords: List[str], target_count: int = 10) -> Dict[str, Any]:
    """Optimisation complÃ¨te pour B-roll avec diversitÃ©"""
    return keyword_processor.optimize_for_broll(keywords, target_count)

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("ðŸ§ª Test du processeur de mots-clÃ©s...")
    
    # Test avec des mots-clÃ©s variÃ©s
    test_keywords = [
        "therapy", "trauma", "memory", "brain", "patient", "healing", "psychology",
        "success", "growth", "strategy", "marketing", "innovation", "technology",
        "mindfulness", "wellness", "fitness", "health", "balance", "happiness",
        "very", "extremely", "because", "therefore", "always", "never"
    ]
    
    print(f"ðŸ“ Mots-clÃ©s de test: {len(test_keywords)}")
    
    # Test d'optimisation complÃ¨te
    result = optimize_for_broll(test_keywords, 12)
    
    print(f"\nðŸŽ¯ RÃ©sultats:")
    print(f"   Mots-clÃ©s optimaux: {result['keywords']}")
    print(f"   RequÃªtes de recherche: {result['search_queries']}")
    print(f"   Statistiques: {result['statistics']}")
    
    print(f"\nðŸ·ï¸ CatÃ©gorisation:")
    for category, words in result['categorized'].items():
        if words:
            print(f"   {category}: {words}")
    
    print(f"\nðŸ“Š QualitÃ©:")
    for quality in result['quality_scores'][:5]:  # Afficher les 5 premiers
        print(f"   {quality.keyword}: confiance {quality.confidence:.2f}, visuel: {quality.is_visual}")
    
    print("\nï¿½ï¿½ Test terminÃ© !") 
