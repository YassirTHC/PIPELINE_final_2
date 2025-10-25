ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å½Â¯ POST-PROCESSING DES MOTS-CLÃƒâ€°S B-ROLL - FILTRAGE + CATÃƒâ€°GORISATION + DÃƒâ€°-DUP
# Pipeline de nettoyage et optimisation des mots-clÃƒÂ©s pour la recherche B-roll

import re
import logging
from typing import Dict, List, Tuple, Any, Set
from collections import OrderedDict, Counter
from dataclasses import dataclass

# Import de l'optimiseur de diversitÃƒÂ©
try:
    from keyword_diversity_optimizer import optimize_broll_keywords_diversity
    DIVERSITY_OPTIMIZER_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Ã¢Å“â€¦ Optimiseur de diversitÃƒÂ© disponible")
except ImportError:
    DIVERSITY_OPTIMIZER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Ã¢Å¡Â Ã¯Â¸Â Optimiseur de diversitÃƒÂ© non disponible - utilisation du mode basique")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KeywordQuality:
    """MÃƒÂ©tadonnÃƒÂ©es de qualitÃƒÂ© pour un mot-clÃƒÂ©"""
    keyword: str
    length: int
    is_visual: bool
    category: str
    confidence: float
    search_ready: bool

class KeywordProcessor:
    """Processeur de mots-clÃƒÂ©s avec filtrage et catÃƒÂ©gorisation intelligente"""
    
    def __init__(self):
        # CatÃƒÂ©gories de mots-clÃƒÂ©s visuels
        self.visual_categories = {
            'actions': ['running', 'walking', 'talking', 'smiling', 'working', 'studying', 'cooking', 'driving'],
            'objects': ['computer', 'phone', 'book', 'car', 'house', 'tree', 'flower', 'food', 'clothes'],
            'places': ['office', 'home', 'park', 'school', 'hospital', 'restaurant', 'street', 'beach'],
            'people': ['doctor', 'teacher', 'student', 'worker', 'family', 'children', 'elderly', 'professional'],
            'emotions': ['happy', 'sad', 'excited', 'calm', 'focused', 'relaxed', 'energetic', 'peaceful'],
            'abstract': ['success', 'growth', 'change', 'improvement', 'development', 'learning', 'healing']
        }
        
        # Mots-clÃƒÂ©s non-visuels ÃƒÂ  filtrer
        self.non_visual_keywords = {
            'abstract_concepts': ['success', 'failure', 'happiness', 'sadness', 'love', 'hate', 'hope', 'fear'],
            'time_words': ['always', 'never', 'sometimes', 'often', 'rarely', 'today', 'yesterday', 'tomorrow'],
            'intensity_words': ['very', 'extremely', 'slightly', 'completely', 'totally', 'partially'],
            'logical_words': ['because', 'therefore', 'however', 'although', 'unless', 'if', 'then', 'else']
        }
        
        # Patterns de nettoyage
        self.cleaning_patterns = [
            (r'[^a-zA-Z0-9\s\-]', ''),  # Supprimer caractÃƒÂ¨res spÃƒÂ©ciaux
            (r'\s+', ' '),              # Normaliser espaces
            (r'^\s+|\s+$', ''),         # Supprimer espaces dÃƒÂ©but/fin
        ]
        
        # Seuils de qualitÃƒÂ©
        self.min_length = 3
        self.max_length = 20
        self.min_confidence = 0.6
    
    def clean_keywords(self, raw_keywords: List[str]) -> List[str]:
        """
        Nettoyage et normalisation des mots-clÃƒÂ©s
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
        
        # DÃƒÂ©-duplication en prÃƒÂ©servant l'ordre
        unique_keywords = list(OrderedDict.fromkeys(cleaned))
        
        logger.info(f"Ã°Å¸Â§Â¹ Mots-clÃƒÂ©s nettoyÃƒÂ©s: {len(raw_keywords)} Ã¢â€ â€™ {len(unique_keywords)}")
        return unique_keywords
    
    def categorize_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """
        CatÃƒÂ©gorisation automatique des mots-clÃƒÂ©s
        """
        categorized = {category: [] for category in self.visual_categories.keys()}
        categorized['uncategorized'] = []
        
        for keyword in keywords:
            categorized_flag = False
            
            # VÃƒÂ©rifier chaque catÃƒÂ©gorie
            for category, examples in self.visual_categories.items():
                # VÃƒÂ©rifier si le mot-clÃƒÂ© correspond ÃƒÂ  la catÃƒÂ©gorie
                if self._matches_category(keyword, examples):
                    categorized[category].append(keyword)
                    categorized_flag = True
                    break
            
            # Si aucune catÃƒÂ©gorie trouvÃƒÂ©e
            if not categorized_flag:
                categorized['uncategorized'].append(keyword)
        
        # Log des rÃƒÂ©sultats
        for category, words in categorized.items():
            if words:
                logger.info(f"Ã°Å¸ÂÂ·Ã¯Â¸Â {category}: {len(words)} mots-clÃƒÂ©s")
        
        return categorized
    
    def _matches_category(self, keyword: str, examples: List[str]) -> bool:
        """
        VÃƒÂ©rifie si un mot-clÃƒÂ© correspond ÃƒÂ  une catÃƒÂ©gorie
        """
        keyword_lower = keyword.lower()
        
        # Correspondance exacte
        if keyword_lower in [ex.lower() for ex in examples]:
            return True
        
        # Correspondance partielle (suffixe/prÃƒÂ©fixe)
        for example in examples:
            example_lower = example.lower()
            if (keyword_lower.endswith(example_lower) or 
                keyword_lower.startswith(example_lower) or
                example_lower in keyword_lower):
                return True
        
        # Correspondance sÃƒÂ©mantique basique
        if any(word in keyword_lower for word in ['ing', 'ed', 'er', 'tion', 'sion', 'ness']):
            # Mots avec suffixes verbaux/nominaux
            return True
        
        return False
    
    def filter_visual_keywords(self, keywords: List[str]) -> List[str]:
        """
        Filtrage pour ne garder que les mots-clÃƒÂ©s visuellement reprÃƒÂ©sentables
        """
        visual_keywords = []
        
        for keyword in keywords:
            # VÃƒÂ©rifier si c'est un concept abstrait
            is_abstract = any(keyword in words for words in self.non_visual_keywords.values())
            
            # VÃƒÂ©rifier si c'est visuellement reprÃƒÂ©sentable
            is_visual = any(keyword in words for words in self.visual_categories.values())
            
            if is_visual and not is_abstract:
                visual_keywords.append(keyword)
            elif not is_abstract and len(keyword) > 4:  # Mots longs non-abstraits
                visual_keywords.append(keyword)
        
        logger.info(f"Ã°Å¸Å½Â¨ Mots-clÃƒÂ©s visuels filtrÃƒÂ©s: {len(keywords)} Ã¢â€ â€™ {len(visual_keywords)}")
        return visual_keywords
    
    def generate_search_queries(self, keywords: List[str], max_queries: int = 12) -> List[str]:
        """
        GÃƒÂ©nÃƒÂ©ration de requÃƒÂªtes de recherche optimisÃƒÂ©es pour les APIs B-roll
        """
        search_queries = []
        
        # RequÃƒÂªtes simples (1-2 mots)
        for keyword in keywords[:max_queries//2]:
            if len(keyword.split()) <= 2:
                search_queries.append(keyword)
        
        # RequÃƒÂªtes composÃƒÂ©es (2-3 mots)
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
        
        # Limiter le nombre de requÃƒÂªtes
        final_queries = search_queries[:max_queries]
        
        logger.info(f"Ã°Å¸â€Â RequÃƒÂªtes de recherche gÃƒÂ©nÃƒÂ©rÃƒÂ©es: {len(final_queries)}")
        return final_queries
    
    def assess_keyword_quality(self, keywords: List[str]) -> List[KeywordQuality]:
        """
        Ãƒâ€°valuation de la qualitÃƒÂ© de chaque mot-clÃƒÂ©
        """
        quality_scores = []
        
        for keyword in keywords:
            # Longueur
            length = len(keyword)
            
            # VisibilitÃƒÂ©
            is_visual = any(keyword in words for words in self.visual_categories.values())
            
            # CatÃƒÂ©gorie
            category = self._get_keyword_category(keyword)
            
            # Confiance (basÃƒÂ©e sur la longueur et la visibilitÃƒÂ©)
            confidence = min(1.0, (length / 10) + (0.5 if is_visual else 0.0))
            
            # PrÃƒÂªt pour la recherche
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
        DÃƒÂ©termine la catÃƒÂ©gorie d'un mot-clÃƒÂ©
        """
        for category, examples in self.visual_categories.items():
            if self._matches_category(keyword, examples):
                return category
        return 'uncategorized'
    
    def optimize_for_broll(self, keywords: List[str], target_count: int = 10) -> Dict[str, Any]:
        """
        Optimisation complÃƒÂ¨te des mots-clÃƒÂ©s pour la recherche B-roll avec diversitÃƒÂ©
        """
        logger.info(f"Ã°Å¸Å¡â‚¬ Optimisation B-roll pour {len(keywords)} mots-clÃƒÂ©s vers {target_count} cibles")
        
        # 1. Nettoyage
        cleaned = self.clean_keywords(keywords)
        
        # 2. Filtrage visuel
        visual = self.filter_visual_keywords(cleaned)
        
        # 3. OPTIMISATION DE DIVERSITÃƒâ€° (NOUVEAU)
        if DIVERSITY_OPTIMIZER_AVAILABLE and len(visual) > target_count:
            logger.info("Ã°Å¸Å½Â¯ Application de l'optimiseur de diversitÃƒÂ©")
            try:
                diversity_result = optimize_broll_keywords_diversity(visual, target_count)
                
                if diversity_result.get('optimization_applied', False):
                    # Utiliser les mots-clÃƒÂ©s optimisÃƒÂ©s par diversitÃƒÂ©
                    optimal_keywords = diversity_result['keywords']
                    search_queries = diversity_result['search_queries']
                    categorized = diversity_result['categories']
                    diversity_metrics = diversity_result['metrics']
                    
                    logger.info(f"Ã¢Å“â€¦ DiversitÃƒÂ© appliquÃƒÂ©e: {diversity_metrics.get('categories_covered', 0)} catÃƒÂ©gories couvertes")
                    logger.info(f"Ã°Å¸â€œÅ  Score de diversitÃƒÂ©: {diversity_metrics.get('diversity_score', 0):.2f}")
                else:
                    # Fallback vers l'ancienne mÃƒÂ©thode
                    logger.warning("Ã¢Å¡Â Ã¯Â¸Â Optimiseur de diversitÃƒÂ© ÃƒÂ©chouÃƒÂ©, fallback vers mÃƒÂ©thode basique")
                    optimal_keywords = self._select_optimal_keywords(quality_scores, target_count)
                    search_queries = self.generate_search_queries(optimal_keywords)
                    categorized = self.categorize_keywords(visual)
                    diversity_metrics = {}
            except Exception as e:
                logger.error(f"Ã¢ÂÅ’ Erreur optimiseur de diversitÃƒÂ©: {e}")
                # Fallback vers l'ancienne mÃƒÂ©thode
                quality_scores = self.assess_keyword_quality(visual)
                optimal_keywords = self._select_optimal_keywords(quality_scores, target_count)
                search_queries = self.generate_search_queries(optimal_keywords)
                categorized = self.categorize_keywords(visual)
                diversity_metrics = {}
        else:
            # MÃƒÂ©thode basique si l'optimiseur n'est pas disponible
            logger.info("Ã°Å¸â€â€ž Utilisation de la mÃƒÂ©thode d'optimisation basique")
            quality_scores = self.assess_keyword_quality(visual)
            optimal_keywords = self._select_optimal_keywords(quality_scores, target_count)
            search_queries = self.generate_search_queries(optimal_keywords)
            categorized = self.categorize_keywords(visual)
            diversity_metrics = {}
        
        # 4. Statistiques
        # S'assurer que quality_scores est dÃƒÂ©fini
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
        
        # Ajouter les mÃƒÂ©triques de diversitÃƒÂ© si disponibles
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
        
        logger.info(f"Ã¢Å“â€¦ Optimisation terminÃƒÂ©e: {stats['optimal']} mots-clÃƒÂ©s optimaux")
        return result
    
    def _select_optimal_keywords(self, quality_scores: List[KeywordQuality], target_count: int) -> List[str]:
        """
        SÃƒÂ©lection optimale des mots-clÃƒÂ©s basÃƒÂ©e sur la qualitÃƒÂ©
        """
        # Trier par confiance dÃƒÂ©croissante
        sorted_keywords = sorted(quality_scores, key=lambda x: x.confidence, reverse=True)
        
        # SÃƒÂ©lectionner les meilleurs
        selected = []
        category_counts = Counter()
        
        for quality in sorted_keywords:
            if len(selected) >= target_count:
                break
            
            # VÃƒÂ©rifier la diversitÃƒÂ© des catÃƒÂ©gories
            if category_counts[quality.category] < target_count // len(self.visual_categories):
                selected.append(quality.keyword)
                category_counts[quality.category] += 1
            elif quality.confidence >= 0.9:  # Exception pour les mots-clÃƒÂ©s de trÃƒÂ¨s haute qualitÃƒÂ©
                selected.append(quality.keyword)
        
        return selected

# === INSTANCE GLOBALE ===
keyword_processor = KeywordProcessor()

# === FONCTIONS UTILITAIRES ===
def clean_keywords(keywords: List[str]) -> List[str]:
    """Nettoyage des mots-clÃƒÂ©s"""
    return keyword_processor.clean_keywords(keywords)

def filter_visual_keywords(keywords: List[str]) -> List[str]:
    """Filtrage des mots-clÃƒÂ©s visuels"""
    return keyword_processor.filter_visual_keywords(keywords)

def categorize_keywords(keywords: List[str]) -> Dict[str, List[str]]:
    """CatÃƒÂ©gorisation des mots-clÃƒÂ©s"""
    return keyword_processor.categorize_keywords(keywords)

def generate_search_queries(keywords: List[str], max_queries: int = 12) -> List[str]:
    """GÃƒÂ©nÃƒÂ©ration de requÃƒÂªtes de recherche"""
    return keyword_processor.generate_search_queries(keywords, max_queries)

def optimize_for_broll(keywords: List[str], target_count: int = 10) -> Dict[str, Any]:
    """Optimisation complÃƒÂ¨te pour B-roll avec diversitÃƒÂ©"""
    return keyword_processor.optimize_for_broll(keywords, target_count)

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("Ã°Å¸Â§Âª Test du processeur de mots-clÃƒÂ©s...")
    
    # Test avec des mots-clÃƒÂ©s variÃƒÂ©s
    test_keywords = [
        "therapy", "trauma", "memory", "brain", "patient", "healing", "psychology",
        "success", "growth", "strategy", "marketing", "innovation", "technology",
        "mindfulness", "wellness", "fitness", "health", "balance", "happiness",
        "very", "extremely", "because", "therefore", "always", "never"
    ]
    
    print(f"Ã°Å¸â€œÂ Mots-clÃƒÂ©s de test: {len(test_keywords)}")
    
    # Test d'optimisation complÃƒÂ¨te
    result = optimize_for_broll(test_keywords, 12)
    
    print(f"\nÃ°Å¸Å½Â¯ RÃƒÂ©sultats:")
    print(f"   Mots-clÃƒÂ©s optimaux: {result['keywords']}")
    print(f"   RequÃƒÂªtes de recherche: {result['search_queries']}")
    print(f"   Statistiques: {result['statistics']}")
    
    print(f"\nÃ°Å¸ÂÂ·Ã¯Â¸Â CatÃƒÂ©gorisation:")
    for category, words in result['categorized'].items():
        if words:
            print(f"   {category}: {words}")
    
    print(f"\nÃ°Å¸â€œÅ  QualitÃƒÂ©:")
    for quality in result['quality_scores'][:5]:  # Afficher les 5 premiers
        print(f"   {quality.keyword}: confiance {quality.confidence:.2f}, visuel: {quality.is_visual}")
    
    print("\nÃ¯Â¿Â½Ã¯Â¿Â½ Test terminÃƒÂ© !") 

