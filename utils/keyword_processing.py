# 🎯 POST-PROCESSING DES MOTS-CLÉS B-ROLL - FILTRAGE + CATÉGORISATION + DÉ-DUP
# Pipeline de nettoyage et optimisation des mots-clés pour la recherche B-roll

import re
import logging
from typing import Dict, List, Tuple, Any, Set
from collections import OrderedDict, Counter
from dataclasses import dataclass

# Import de l'optimiseur de diversité
try:
    from keyword_diversity_optimizer import optimize_broll_keywords_diversity
    DIVERSITY_OPTIMIZER_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Optimiseur de diversité disponible")
except ImportError:
    DIVERSITY_OPTIMIZER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Optimiseur de diversité non disponible - utilisation du mode basique")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KeywordQuality:
    """Métadonnées de qualité pour un mot-clé"""
    keyword: str
    length: int
    is_visual: bool
    category: str
    confidence: float
    search_ready: bool

class KeywordProcessor:
    """Processeur de mots-clés avec filtrage et catégorisation intelligente"""
    
    def __init__(self):
        # Catégories de mots-clés visuels
        self.visual_categories = {
            'actions': ['running', 'walking', 'talking', 'smiling', 'working', 'studying', 'cooking', 'driving'],
            'objects': ['computer', 'phone', 'book', 'car', 'house', 'tree', 'flower', 'food', 'clothes'],
            'places': ['office', 'home', 'park', 'school', 'hospital', 'restaurant', 'street', 'beach'],
            'people': ['doctor', 'teacher', 'student', 'worker', 'family', 'children', 'elderly', 'professional'],
            'emotions': ['happy', 'sad', 'excited', 'calm', 'focused', 'relaxed', 'energetic', 'peaceful'],
            'abstract': ['success', 'growth', 'change', 'improvement', 'development', 'learning', 'healing']
        }
        
        # Mots-clés non-visuels à filtrer
        self.non_visual_keywords = {
            'abstract_concepts': ['success', 'failure', 'happiness', 'sadness', 'love', 'hate', 'hope', 'fear'],
            'time_words': ['always', 'never', 'sometimes', 'often', 'rarely', 'today', 'yesterday', 'tomorrow'],
            'intensity_words': ['very', 'extremely', 'slightly', 'completely', 'totally', 'partially'],
            'logical_words': ['because', 'therefore', 'however', 'although', 'unless', 'if', 'then', 'else']
        }
        
        # Patterns de nettoyage
        self.cleaning_patterns = [
            (r'[^a-zA-Z0-9\s\-]', ''),  # Supprimer caractères spéciaux
            (r'\s+', ' '),              # Normaliser espaces
            (r'^\s+|\s+$', ''),         # Supprimer espaces début/fin
        ]
        
        # Seuils de qualité
        self.min_length = 3
        self.max_length = 20
        self.min_confidence = 0.6
    
    def clean_keywords(self, raw_keywords: List[str]) -> List[str]:
        """
        Nettoyage et normalisation des mots-clés
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
        
        # Dé-duplication en préservant l'ordre
        unique_keywords = list(OrderedDict.fromkeys(cleaned))
        
        logger.info(f"🧹 Mots-clés nettoyés: {len(raw_keywords)} → {len(unique_keywords)}")
        return unique_keywords
    
    def categorize_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """
        Catégorisation automatique des mots-clés
        """
        categorized = {category: [] for category in self.visual_categories.keys()}
        categorized['uncategorized'] = []
        
        for keyword in keywords:
            categorized_flag = False
            
            # Vérifier chaque catégorie
            for category, examples in self.visual_categories.items():
                # Vérifier si le mot-clé correspond à la catégorie
                if self._matches_category(keyword, examples):
                    categorized[category].append(keyword)
                    categorized_flag = True
                    break
            
            # Si aucune catégorie trouvée
            if not categorized_flag:
                categorized['uncategorized'].append(keyword)
        
        # Log des résultats
        for category, words in categorized.items():
            if words:
                logger.info(f"🏷️ {category}: {len(words)} mots-clés")
        
        return categorized
    
    def _matches_category(self, keyword: str, examples: List[str]) -> bool:
        """
        Vérifie si un mot-clé correspond à une catégorie
        """
        keyword_lower = keyword.lower()
        
        # Correspondance exacte
        if keyword_lower in [ex.lower() for ex in examples]:
            return True
        
        # Correspondance partielle (suffixe/préfixe)
        for example in examples:
            example_lower = example.lower()
            if (keyword_lower.endswith(example_lower) or 
                keyword_lower.startswith(example_lower) or
                example_lower in keyword_lower):
                return True
        
        # Correspondance sémantique basique
        if any(word in keyword_lower for word in ['ing', 'ed', 'er', 'tion', 'sion', 'ness']):
            # Mots avec suffixes verbaux/nominaux
            return True
        
        return False
    
    def filter_visual_keywords(self, keywords: List[str]) -> List[str]:
        """
        Filtrage pour ne garder que les mots-clés visuellement représentables
        """
        visual_keywords = []
        
        for keyword in keywords:
            # Vérifier si c'est un concept abstrait
            is_abstract = any(keyword in words for words in self.non_visual_keywords.values())
            
            # Vérifier si c'est visuellement représentable
            is_visual = any(keyword in words for words in self.visual_categories.values())
            
            if is_visual and not is_abstract:
                visual_keywords.append(keyword)
            elif not is_abstract and len(keyword) > 4:  # Mots longs non-abstraits
                visual_keywords.append(keyword)
        
        logger.info(f"🎨 Mots-clés visuels filtrés: {len(keywords)} → {len(visual_keywords)}")
        return visual_keywords
    
    def generate_search_queries(self, keywords: List[str], max_queries: int = 12) -> List[str]:
        """
        Génération de requêtes de recherche optimisées pour les APIs B-roll
        """
        search_queries = []
        
        # Requêtes simples (1-2 mots)
        for keyword in keywords[:max_queries//2]:
            if len(keyword.split()) <= 2:
                search_queries.append(keyword)
        
        # Requêtes composées (2-3 mots)
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
        
        # Limiter le nombre de requêtes
        final_queries = search_queries[:max_queries]
        
        logger.info(f"🔍 Requêtes de recherche générées: {len(final_queries)}")
        return final_queries
    
    def assess_keyword_quality(self, keywords: List[str]) -> List[KeywordQuality]:
        """
        Évaluation de la qualité de chaque mot-clé
        """
        quality_scores = []
        
        for keyword in keywords:
            # Longueur
            length = len(keyword)
            
            # Visibilité
            is_visual = any(keyword in words for words in self.visual_categories.values())
            
            # Catégorie
            category = self._get_keyword_category(keyword)
            
            # Confiance (basée sur la longueur et la visibilité)
            confidence = min(1.0, (length / 10) + (0.5 if is_visual else 0.0))
            
            # Prêt pour la recherche
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
        Détermine la catégorie d'un mot-clé
        """
        for category, examples in self.visual_categories.items():
            if self._matches_category(keyword, examples):
                return category
        return 'uncategorized'
    
    def optimize_for_broll(self, keywords: List[str], target_count: int = 10) -> Dict[str, Any]:
        """
        Optimisation complète des mots-clés pour la recherche B-roll avec diversité
        """
        logger.info(f"🚀 Optimisation B-roll pour {len(keywords)} mots-clés vers {target_count} cibles")
        
        # 1. Nettoyage
        cleaned = self.clean_keywords(keywords)
        
        # 2. Filtrage visuel
        visual = self.filter_visual_keywords(cleaned)
        
        # 3. OPTIMISATION DE DIVERSITÉ (NOUVEAU)
        if DIVERSITY_OPTIMIZER_AVAILABLE and len(visual) > target_count:
            logger.info("🎯 Application de l'optimiseur de diversité")
            try:
                diversity_result = optimize_broll_keywords_diversity(visual, target_count)
                
                if diversity_result.get('optimization_applied', False):
                    # Utiliser les mots-clés optimisés par diversité
                    optimal_keywords = diversity_result['keywords']
                    search_queries = diversity_result['search_queries']
                    categorized = diversity_result['categories']
                    diversity_metrics = diversity_result['metrics']
                    
                    logger.info(f"✅ Diversité appliquée: {diversity_metrics.get('categories_covered', 0)} catégories couvertes")
                    logger.info(f"📊 Score de diversité: {diversity_metrics.get('diversity_score', 0):.2f}")
                else:
                    # Fallback vers l'ancienne méthode
                    logger.warning("⚠️ Optimiseur de diversité échoué, fallback vers méthode basique")
                    optimal_keywords = self._select_optimal_keywords(quality_scores, target_count)
                    search_queries = self.generate_search_queries(optimal_keywords)
                    categorized = self.categorize_keywords(visual)
                    diversity_metrics = {}
            except Exception as e:
                logger.error(f"❌ Erreur optimiseur de diversité: {e}")
                # Fallback vers l'ancienne méthode
                quality_scores = self.assess_keyword_quality(visual)
                optimal_keywords = self._select_optimal_keywords(quality_scores, target_count)
                search_queries = self.generate_search_queries(optimal_keywords)
                categorized = self.categorize_keywords(visual)
                diversity_metrics = {}
        else:
            # Méthode basique si l'optimiseur n'est pas disponible
            logger.info("🔄 Utilisation de la méthode d'optimisation basique")
            quality_scores = self.assess_keyword_quality(visual)
            optimal_keywords = self._select_optimal_keywords(quality_scores, target_count)
            search_queries = self.generate_search_queries(optimal_keywords)
            categorized = self.categorize_keywords(visual)
            diversity_metrics = {}
        
        # 4. Statistiques
        # S'assurer que quality_scores est défini
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
        
        # Ajouter les métriques de diversité si disponibles
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
        
        logger.info(f"✅ Optimisation terminée: {stats['optimal']} mots-clés optimaux")
        return result
    
    def _select_optimal_keywords(self, quality_scores: List[KeywordQuality], target_count: int) -> List[str]:
        """
        Sélection optimale des mots-clés basée sur la qualité
        """
        # Trier par confiance décroissante
        sorted_keywords = sorted(quality_scores, key=lambda x: x.confidence, reverse=True)
        
        # Sélectionner les meilleurs
        selected = []
        category_counts = Counter()
        
        for quality in sorted_keywords:
            if len(selected) >= target_count:
                break
            
            # Vérifier la diversité des catégories
            if category_counts[quality.category] < target_count // len(self.visual_categories):
                selected.append(quality.keyword)
                category_counts[quality.category] += 1
            elif quality.confidence >= 0.9:  # Exception pour les mots-clés de très haute qualité
                selected.append(quality.keyword)
        
        return selected

# === INSTANCE GLOBALE ===
keyword_processor = KeywordProcessor()

# === FONCTIONS UTILITAIRES ===
def clean_keywords(keywords: List[str]) -> List[str]:
    """Nettoyage des mots-clés"""
    return keyword_processor.clean_keywords(keywords)

def filter_visual_keywords(keywords: List[str]) -> List[str]:
    """Filtrage des mots-clés visuels"""
    return keyword_processor.filter_visual_keywords(keywords)

def categorize_keywords(keywords: List[str]) -> Dict[str, List[str]]:
    """Catégorisation des mots-clés"""
    return keyword_processor.categorize_keywords(keywords)

def generate_search_queries(keywords: List[str], max_queries: int = 12) -> List[str]:
    """Génération de requêtes de recherche"""
    return keyword_processor.generate_search_queries(keywords, max_queries)

def optimize_for_broll(keywords: List[str], target_count: int = 10) -> Dict[str, Any]:
    """Optimisation complète pour B-roll avec diversité"""
    return keyword_processor.optimize_for_broll(keywords, target_count)

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("🧪 Test du processeur de mots-clés...")
    
    # Test avec des mots-clés variés
    test_keywords = [
        "therapy", "trauma", "memory", "brain", "patient", "healing", "psychology",
        "success", "growth", "strategy", "marketing", "innovation", "technology",
        "mindfulness", "wellness", "fitness", "health", "balance", "happiness",
        "very", "extremely", "because", "therefore", "always", "never"
    ]
    
    print(f"📝 Mots-clés de test: {len(test_keywords)}")
    
    # Test d'optimisation complète
    result = optimize_for_broll(test_keywords, 12)
    
    print(f"\n🎯 Résultats:")
    print(f"   Mots-clés optimaux: {result['keywords']}")
    print(f"   Requêtes de recherche: {result['search_queries']}")
    print(f"   Statistiques: {result['statistics']}")
    
    print(f"\n🏷️ Catégorisation:")
    for category, words in result['categorized'].items():
        if words:
            print(f"   {category}: {words}")
    
    print(f"\n📊 Qualité:")
    for quality in result['quality_scores'][:5]:  # Afficher les 5 premiers
        print(f"   {quality.keyword}: confiance {quality.confidence:.2f}, visuel: {quality.is_visual}")
    
    print("\n�� Test terminé !") 