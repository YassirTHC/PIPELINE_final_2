# 🎯 OPTIMISATEUR DE DIVERSITÉ DES MOTS-CLÉS B-ROLL
# Évite les répétitions et assure une couverture visuelle complète

import logging
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class KeywordDiversityOptimizer:
    """Optimise la diversité et la pertinence des mots-clés B-roll"""
    
    def __init__(self):
        # Catégories de diversité pour couvrir tous les aspects visuels
        self.diversity_categories = {
            'people': {
                'keywords': ['doctor', 'patient', 'therapist', 'nurse', 'specialist', 'consultant'],
                'weight': 0.25,  # 25% des mots-clés
                'max_per_category': 2
            },
            'actions': {
                'keywords': ['consultation', 'examination', 'treatment', 'therapy', 'diagnosis', 'assessment'],
                'weight': 0.25,  # 25% des mots-clés
                'max_per_category': 2
            },
            'environments': {
                'keywords': ['office', 'clinic', 'hospital', 'consultation_room', 'medical_center', 'therapy_room'],
                'weight': 0.20,  # 20% des mots-clés
                'max_per_category': 2
            },
            'objects': {
                'keywords': ['medical_charts', 'stethoscope', 'brain_scan', 'equipment', 'instruments', 'documents'],
                'weight': 0.20,  # 20% des mots-clés
                'max_per_category': 2
            },
            'context': {
                'keywords': ['professional', 'medical', 'clinical', 'therapeutic', 'diagnostic', 'treatment'],
                'weight': 0.10,  # 10% des mots-clés
                'max_per_category': 1
            }
        }
        
        # Mots-clés trop génériques à éviter
        self.generic_keywords = {
            'therapy', 'healing', 'treatment', 'office', 'room', 'building',
            'person', 'people', 'man', 'woman', 'thing', 'stuff', 'way',
            'time', 'place', 'work', 'make', 'do', 'get', 'go', 'come',
            'see', 'look', 'hear', 'feel', 'think', 'know', 'want', 'need'
        }
        
        # Patterns pour identifier la spécificité
        self.specificity_patterns = [
            r'[a-z]+_[a-z]+',  # doctor_office, therapy_session
            r'[a-z]+\s+[a-z]+',  # medical consultation, brain scan
            r'[a-z]+[A-Z][a-z]+',  # medicalChart, brainScan
        ]
    
    def optimize_keywords(self, raw_keywords: List[str], target_count: int = 10) -> Dict[str, Any]:
        """
        Optimise les mots-clés pour la diversité et la spécificité
        
        Args:
            raw_keywords: Liste brute de mots-clés
            target_count: Nombre cible de mots-clés optimisés
        
        Returns:
            Dict avec mots-clés optimisés et métriques
        """
        try:
            logger.info(f"🎯 Optimisation de {len(raw_keywords)} mots-clés vers {target_count} cibles")
            
            # 1. Nettoyer et filtrer
            cleaned_keywords = self._clean_and_filter_keywords(raw_keywords)
            
            # 2. Évaluer la spécificité
            specificity_scores = self._evaluate_specificity(cleaned_keywords)
            
            # 3. Catégoriser automatiquement
            categorized_keywords = self._categorize_keywords(cleaned_keywords)
            
            # 4. Optimiser pour la diversité
            optimized_keywords = self._apply_diversity_strategy(categorized_keywords, target_count)
            
            # 5. Générer des requêtes de recherche optimisées
            search_queries = self._generate_optimized_search_queries(optimized_keywords)
            
            # 6. Calculer les métriques
            metrics = self._calculate_optimization_metrics(raw_keywords, optimized_keywords)
            
            result = {
                'keywords': optimized_keywords,
                'search_queries': search_queries,
                'categories': categorized_keywords,
                'metrics': metrics,
                'optimization_applied': True
            }
            
            logger.info(f"✅ Optimisation terminée: {len(optimized_keywords)} mots-clés, {len(search_queries)} requêtes")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur optimisation diversité: {e}")
            # Fallback: retourner les mots-clés originaux
            return {
                'keywords': raw_keywords[:target_count],
                'search_queries': [],
                'categories': {},
                'metrics': {'error': str(e)},
                'optimization_applied': False
            }
    
    def _clean_and_filter_keywords(self, keywords: List[str]) -> List[str]:
        """Nettoie et filtre les mots-clés"""
        cleaned = []
        
        for kw in keywords:
            if not isinstance(kw, str):
                continue
                
            # Nettoyer
            clean_kw = kw.strip().lower()
            if len(clean_kw) < 3:
                continue
                
            # Filtrer les mots trop génériques
            if clean_kw in self.generic_keywords:
                continue
                
            # Filtrer les mots trop courts ou trop longs
            if len(clean_kw) > 25:
                continue
                
            cleaned.append(clean_kw)
        
        # Dédupliquer
        return list(dict.fromkeys(cleaned))
    
    def _evaluate_specificity(self, keywords: List[str]) -> Dict[str, float]:
        """Évalue la spécificité de chaque mot-clé"""
        specificity_scores = {}
        
        for kw in keywords:
            score = 0.0
            
            # Bonus pour les patterns de spécificité
            for pattern in self.specificity_patterns:
                if re.search(pattern, kw):
                    score += 0.3
                    break
            
            # Bonus pour la longueur (mots plus longs = plus spécifiques)
            if len(kw) > 8:
                score += 0.2
            elif len(kw) > 5:
                score += 0.1
            
            # Bonus pour les mots composés
            if '_' in kw or ' ' in kw:
                score += 0.2
            
            # Bonus pour les termes techniques
            technical_terms = ['medical', 'clinical', 'therapeutic', 'diagnostic', 'professional']
            if any(term in kw for term in technical_terms):
                score += 0.1
            
            specificity_scores[kw] = min(1.0, score)
        
        return specificity_scores
    
    def _categorize_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Catégorise automatiquement les mots-clés"""
        categorized = defaultdict(list)
        
        for kw in keywords:
            # Essayer de catégoriser automatiquement
            category_found = False
            
            for category, info in self.diversity_categories.items():
                category_keywords = info['keywords']
                
                # Vérifier si le mot-clé correspond à cette catégorie
                if any(cat_kw in kw or kw in cat_kw for cat_kw in category_keywords):
                    categorized[category].append(kw)
                    category_found = True
                    break
            
            # Si aucune catégorie trouvée, mettre dans 'uncategorized'
            if not category_found:
                categorized['uncategorized'].append(kw)
        
        return dict(categorized)
    
    def _apply_diversity_strategy(self, categorized_keywords: Dict[str, List[str]], target_count: int) -> List[str]:
        """Applique la stratégie de diversité"""
        optimized = []
        
        # Calculer le nombre de mots-clés par catégorie
        category_allocations = {}
        for category, info in self.diversity_categories.items():
            max_per_cat = min(info['max_per_category'], int(target_count * info['weight']))
            category_allocations[category] = max_per_cat
        
        # Sélectionner les meilleurs mots-clés de chaque catégorie
        for category, max_count in category_allocations.items():
            if category in categorized_keywords:
                category_keywords = categorized_keywords[category]
                # Prendre les premiers (déjà triés par pertinence)
                selected = category_keywords[:max_count]
                optimized.extend(selected)
        
        # Ajouter des mots-clés non catégorisés si on n'a pas atteint le target
        if len(optimized) < target_count and 'uncategorized' in categorized_keywords:
            remaining = target_count - len(optimized)
            uncategorized = categorized_keywords['uncategorized'][:remaining]
            optimized.extend(uncategorized)
        
        # Limiter au nombre cible
        return optimized[:target_count]
    
    def _generate_optimized_search_queries(self, keywords: List[str]) -> List[str]:
        """Génère des requêtes de recherche optimisées"""
        search_queries = []
        
        # Combiner les mots-clés pour créer des requêtes de recherche
        for i, kw1 in enumerate(keywords):
            # Requête simple avec le mot-clé principal
            if len(kw1) > 3:
                search_queries.append(kw1)
            
            # Requêtes combinées (2-3 mots)
            for j, kw2 in enumerate(keywords[i+1:], i+1):
                if len(search_queries) >= 8:  # Limiter à 8 requêtes
                    break
                    
                combined = f"{kw1} {kw2}"
                if len(combined) <= 25:  # Limite pour les APIs
                    search_queries.append(combined)
        
        # Ajouter des requêtes contextuelles
        context_queries = [
            "medical consultation",
            "therapy session",
            "professional office",
            "clinical environment"
        ]
        
        for query in context_queries:
            if len(search_queries) < 10:  # Garder un total raisonnable
                search_queries.append(query)
        
        return search_queries[:10]  # Max 10 requêtes
    
    def _calculate_optimization_metrics(self, original: List[str], optimized: List[str]) -> Dict[str, Any]:
        """Calcule les métriques d'optimisation"""
        try:
            # Diversité des catégories
            categories_covered = len(set(self._categorize_keywords(optimized).keys()))
            
            # Spécificité moyenne
            specificity_scores = self._evaluate_specificity(optimized)
            avg_specificity = sum(specificity_scores.values()) / len(specificity_scores) if specificity_scores else 0.0
            
            # Taux de réduction
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
    """Factory pour créer un optimiseur de diversité"""
    return KeywordDiversityOptimizer()

def optimize_broll_keywords_diversity(keywords: List[str], target_count: int = 10) -> Dict[str, Any]:
    """Fonction utilitaire pour optimiser rapidement les mots-clés"""
    optimizer = create_diversity_optimizer()
    return optimizer.optimize_keywords(keywords, target_count)

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("🧪 Test de l'optimiseur de diversité...")
    
    # Test avec des mots-clés de test
    test_keywords = [
        "therapy", "healing", "doctor", "office", "patient", "consultation",
        "brain", "scan", "medical", "charts", "stethoscope", "examination"
    ]
    
    optimizer = create_diversity_optimizer()
    result = optimizer.optimize_keywords(test_keywords, 8)
    
    print(f"✅ Mots-clés optimisés: {result['keywords']}")
    print(f"🔍 Requêtes de recherche: {result['search_queries']}")
    print(f"📊 Métriques: {result['metrics']}")
    
    print("\n�� Test terminé !") 