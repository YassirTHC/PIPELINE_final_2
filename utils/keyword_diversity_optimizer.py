# -*- coding: utf-8 -*-
# ðŸŽ¯ OPTIMISATEUR DE DIVERSITÃ‰ DES MOTS-CLÃ‰S B-ROLL
# Ã‰vite les rÃ©pÃ©titions et assure une couverture visuelle complÃ¨te

import logging
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class KeywordDiversityOptimizer:
    """Optimise la diversitÃ© et la pertinence des mots-clÃ©s B-roll"""
    
    def __init__(self):
        # CatÃ©gories de diversitÃ© pour couvrir tous les aspects visuels
        self.diversity_categories = {
            'people': {
                'keywords': ['doctor', 'patient', 'therapist', 'nurse', 'specialist', 'consultant'],
                'weight': 0.25,  # 25% des mots-clÃ©s
                'max_per_category': 2
            },
            'actions': {
                'keywords': ['consultation', 'examination', 'treatment', 'therapy', 'diagnosis', 'assessment'],
                'weight': 0.25,  # 25% des mots-clÃ©s
                'max_per_category': 2
            },
            'environments': {
                'keywords': ['office', 'clinic', 'hospital', 'consultation_room', 'medical_center', 'therapy_room'],
                'weight': 0.20,  # 20% des mots-clÃ©s
                'max_per_category': 2
            },
            'objects': {
                'keywords': ['medical_charts', 'stethoscope', 'brain_scan', 'equipment', 'instruments', 'documents'],
                'weight': 0.20,  # 20% des mots-clÃ©s
                'max_per_category': 2
            },
            'context': {
                'keywords': ['professional', 'medical', 'clinical', 'therapeutic', 'diagnostic', 'treatment'],
                'weight': 0.10,  # 10% des mots-clÃ©s
                'max_per_category': 1
            }
        }
        
        # Mots-clÃ©s trop gÃ©nÃ©riques Ã  Ã©viter
        self.generic_keywords = {
            'therapy', 'healing', 'treatment', 'office', 'room', 'building',
            'person', 'people', 'man', 'woman', 'thing', 'stuff', 'way',
            'time', 'place', 'work', 'make', 'do', 'get', 'go', 'come',
            'see', 'look', 'hear', 'feel', 'think', 'know', 'want', 'need'
        }
        
        # Patterns pour identifier la spÃ©cificitÃ©
        self.specificity_patterns = [
            r'[a-z]+_[a-z]+',  # doctor_office, therapy_session
            r'[a-z]+\s+[a-z]+',  # medical consultation, brain scan
            r'[a-z]+[A-Z][a-z]+',  # medicalChart, brainScan
        ]
    
    def optimize_keywords(self, raw_keywords: List[str], target_count: int = 10) -> Dict[str, Any]:
        """
        Optimise les mots-clÃ©s pour la diversitÃ© et la spÃ©cificitÃ©
        
        Args:
            raw_keywords: Liste brute de mots-clÃ©s
            target_count: Nombre cible de mots-clÃ©s optimisÃ©s
        
        Returns:
            Dict avec mots-clÃ©s optimisÃ©s et mÃ©triques
        """
        try:
            logger.info(f"ðŸŽ¯ Optimisation de {len(raw_keywords)} mots-clÃ©s vers {target_count} cibles")
            
            # 1. Nettoyer et filtrer
            cleaned_keywords = self._clean_and_filter_keywords(raw_keywords)
            
            # 2. Ã‰valuer la spÃ©cificitÃ©
            specificity_scores = self._evaluate_specificity(cleaned_keywords)
            
            # 3. CatÃ©goriser automatiquement
            categorized_keywords = self._categorize_keywords(cleaned_keywords)
            
            # 4. Optimiser pour la diversitÃ©
            optimized_keywords = self._apply_diversity_strategy(categorized_keywords, target_count)
            
            # 5. GÃ©nÃ©rer des requÃªtes de recherche optimisÃ©es
            search_queries = self._generate_optimized_search_queries(optimized_keywords)
            
            # 6. Calculer les mÃ©triques
            metrics = self._calculate_optimization_metrics(raw_keywords, optimized_keywords)
            
            result = {
                'keywords': optimized_keywords,
                'search_queries': search_queries,
                'categories': categorized_keywords,
                'metrics': metrics,
                'optimization_applied': True
            }
            
            logger.info(f"âœ… Optimisation terminÃ©e: {len(optimized_keywords)} mots-clÃ©s, {len(search_queries)} requÃªtes")
            return result
            
        except Exception as e:
            logger.error(f"âŒ Erreur optimisation diversitÃ©: {e}")
            # Fallback: retourner les mots-clÃ©s originaux
            return {
                'keywords': raw_keywords[:target_count],
                'search_queries': [],
                'categories': {},
                'metrics': {'error': str(e)},
                'optimization_applied': False
            }
    
    def _clean_and_filter_keywords(self, keywords: List[str]) -> List[str]:
        """Nettoie et filtre les mots-clÃ©s"""
        cleaned = []
        
        for kw in keywords:
            if not isinstance(kw, str):
                continue
                
            # Nettoyer
            clean_kw = kw.strip().lower()
            if len(clean_kw) < 3:
                continue
                
            # Filtrer les mots trop gÃ©nÃ©riques
            if clean_kw in self.generic_keywords:
                continue
                
            # Filtrer les mots trop courts ou trop longs
            if len(clean_kw) > 25:
                continue
                
            cleaned.append(clean_kw)
        
        # DÃ©dupliquer
        return list(dict.fromkeys(cleaned))
    
    def _evaluate_specificity(self, keywords: List[str]) -> Dict[str, float]:
        """Ã‰value la spÃ©cificitÃ© de chaque mot-clÃ©"""
        specificity_scores = {}
        
        for kw in keywords:
            score = 0.0
            
            # Bonus pour les patterns de spÃ©cificitÃ©
            for pattern in self.specificity_patterns:
                if re.search(pattern, kw):
                    score += 0.3
                    break
            
            # Bonus pour la longueur (mots plus longs = plus spÃ©cifiques)
            if len(kw) > 8:
                score += 0.2
            elif len(kw) > 5:
                score += 0.1
            
            # Bonus pour les mots composÃ©s
            if '_' in kw or ' ' in kw:
                score += 0.2
            
            # Bonus pour les termes techniques
            technical_terms = ['medical', 'clinical', 'therapeutic', 'diagnostic', 'professional']
            if any(term in kw for term in technical_terms):
                score += 0.1
            
            specificity_scores[kw] = min(1.0, score)
        
        return specificity_scores
    
    def _categorize_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """CatÃ©gorise automatiquement les mots-clÃ©s"""
        categorized = defaultdict(list)
        
        for kw in keywords:
            # Essayer de catÃ©goriser automatiquement
            category_found = False
            
            for category, info in self.diversity_categories.items():
                category_keywords = info['keywords']
                
                # VÃ©rifier si le mot-clÃ© correspond Ã  cette catÃ©gorie
                if any(cat_kw in kw or kw in cat_kw for cat_kw in category_keywords):
                    categorized[category].append(kw)
                    category_found = True
                    break
            
            # Si aucune catÃ©gorie trouvÃ©e, mettre dans 'uncategorized'
            if not category_found:
                categorized['uncategorized'].append(kw)
        
        return dict(categorized)
    
    def _apply_diversity_strategy(self, categorized_keywords: Dict[str, List[str]], target_count: int) -> List[str]:
        """Applique la stratÃ©gie de diversitÃ©"""
        optimized = []
        
        # Calculer le nombre de mots-clÃ©s par catÃ©gorie
        category_allocations = {}
        for category, info in self.diversity_categories.items():
            max_per_cat = min(info['max_per_category'], int(target_count * info['weight']))
            category_allocations[category] = max_per_cat
        
        # SÃ©lectionner les meilleurs mots-clÃ©s de chaque catÃ©gorie
        for category, max_count in category_allocations.items():
            if category in categorized_keywords:
                category_keywords = categorized_keywords[category]
                # Prendre les premiers (dÃ©jÃ  triÃ©s par pertinence)
                selected = category_keywords[:max_count]
                optimized.extend(selected)
        
        # Ajouter des mots-clÃ©s non catÃ©gorisÃ©s si on n'a pas atteint le target
        if len(optimized) < target_count and 'uncategorized' in categorized_keywords:
            remaining = target_count - len(optimized)
            uncategorized = categorized_keywords['uncategorized'][:remaining]
            optimized.extend(uncategorized)
        
        # Limiter au nombre cible
        return optimized[:target_count]
    
    def _generate_optimized_search_queries(self, keywords: List[str]) -> List[str]:
        """GÃ©nÃ¨re des requÃªtes de recherche optimisÃ©es"""
        search_queries = []
        
        # Combiner les mots-clÃ©s pour crÃ©er des requÃªtes de recherche
        for i, kw1 in enumerate(keywords):
            # RequÃªte simple avec le mot-clÃ© principal
            if len(kw1) > 3:
                search_queries.append(kw1)
            
            # RequÃªtes combinÃ©es (2-3 mots)
            for j, kw2 in enumerate(keywords[i+1:], i+1):
                if len(search_queries) >= 8:  # Limiter Ã  8 requÃªtes
                    break
                    
                combined = f"{kw1} {kw2}"
                if len(combined) <= 25:  # Limite pour les APIs
                    search_queries.append(combined)
        
        # Ajouter des requÃªtes contextuelles
        context_queries = [
            "medical consultation",
            "therapy session",
            "professional office",
            "clinical environment"
        ]
        
        for query in context_queries:
            if len(search_queries) < 10:  # Garder un total raisonnable
                search_queries.append(query)
        
        return search_queries[:10]  # Max 10 requÃªtes
    
    def _calculate_optimization_metrics(self, original: List[str], optimized: List[str]) -> Dict[str, Any]:
        """Calcule les mÃ©triques d'optimisation"""
        try:
            # DiversitÃ© des catÃ©gories
            categories_covered = len(set(self._categorize_keywords(optimized).keys()))
            
            # SpÃ©cificitÃ© moyenne
            specificity_scores = self._evaluate_specificity(optimized)
            avg_specificity = sum(specificity_scores.values()) / len(specificity_scores) if specificity_scores else 0.0
            
            # Taux de rÃ©duction
            reduction_rate = 1 - (len(optimized) / len(original)) if original else 0.0
            
            return {
                'original_count': len(original),
                'optimized_count': len(optimized),
                'reduction_rate': reduction_rate,
                'categories_covered': categories_covered,
                'avg_specificity': avg_specificity,
                'diversity_score': categories_covered / len(self.diversity_categories)
            }
        except Exception as e:
            return {'error': str(e)}

# === FONCTIONS UTILITAIRES ===
def create_diversity_optimizer() -> KeywordDiversityOptimizer:
    """Factory pour crÃ©er un optimiseur de diversitÃ©"""
    return KeywordDiversityOptimizer()

def optimize_broll_keywords_diversity(keywords: List[str], target_count: int = 10) -> Dict[str, Any]:
    """Fonction utilitaire pour optimiser rapidement les mots-clÃ©s"""
    optimizer = create_diversity_optimizer()
    return optimizer.optimize_keywords(keywords, target_count)

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("ðŸ§ª Test de l'optimiseur de diversitÃ©...")
    
    # Test avec des mots-clÃ©s de test
    test_keywords = [
        "therapy", "healing", "doctor", "office", "patient", "consultation",
        "brain", "scan", "medical", "charts", "stethoscope", "examination"
    ]
    
    optimizer = create_diversity_optimizer()
    result = optimizer.optimize_keywords(test_keywords, 8)
    
    print(f"âœ… Mots-clÃ©s optimisÃ©s: {result['keywords']}")
    print(f"ðŸ” RequÃªtes de recherche: {result['search_queries']}")
    print(f"ðŸ“Š MÃ©triques: {result['metrics']}")
    
    print("\nï¿½ï¿½ Test terminÃ© !") 
