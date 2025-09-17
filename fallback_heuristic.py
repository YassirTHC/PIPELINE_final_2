#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 FALLBACK HEURISTIQUE - EXTRACTION MOTS-CLÉS INTELLIGENTE
Système de fallback pour générer des mots-clés B-roll quand le LLM échoue
"""

import re
import json
from collections import Counter
from pathlib import Path

class HeuristicKeywordExtractor:
    """Extracteur de mots-clés heuristique intelligent"""
    
    def __init__(self):
        # Mots-clés universels B-roll (toujours utiles)
        self.universal_keywords = [
            "people walking", "family dinner", "office desk", "city street",
            "hands typing", "coffee shop", "nature trail", "kids playing",
            "doctor clinic", "gym workout", "car driving", "sunset sky",
            "business meeting", "hospital room", "school classroom", "park bench",
            "kitchen cooking", "bedroom sleeping", "garden flowers", "beach waves"
        ]
        
        # Mots-clés contextuels par domaine
        self.domain_keywords = {
            "medical": ["doctor", "nurse", "hospital", "clinic", "patient", "treatment", "medicine"],
            "business": ["office", "meeting", "presentation", "computer", "phone", "desk", "work"],
            "family": ["family", "children", "parents", "home", "kitchen", "living room", "garden"],
            "technology": ["computer", "phone", "screen", "keyboard", "coding", "data", "network"],
            "education": ["student", "teacher", "classroom", "book", "learning", "study", "school"],
            "health": ["exercise", "gym", "running", "yoga", "meditation", "wellness", "fitness"]
        }
        
        # Mots-clés d'action dynamiques
        self.action_keywords = [
            "walking", "running", "talking", "thinking", "working", "studying",
            "cooking", "driving", "reading", "writing", "listening", "speaking",
            "moving", "standing", "sitting", "dancing", "singing", "playing"
        ]
    
    def extract_keywords_from_text(self, text, target_count=25, context_hint=None):
        """Extraction heuristique de mots-clés depuis un texte"""
        
        print(f"🧠 EXTRACTION HEURISTIQUE - {target_count} mots-clés cibles")
        print("=" * 60)
        
        # 1. Extraction des mots significatifs
        words = self._extract_significant_words(text)
        print(f"📝 Mots significatifs extraits: {len(words)}")
        
        # 2. Analyse de fréquence
        word_freq = Counter(words)
        top_words = [word for word, freq in word_freq.most_common(15)]
        print(f"🎯 Top mots par fréquence: {top_words[:10]}")
        
        # 3. Détection du domaine
        detected_domain = self._detect_domain(text, context_hint)
        print(f"🏥 Domaine détecté: {detected_domain}")
        
        # 4. Génération des mots-clés
        keywords = self._generate_keywords(top_words, detected_domain, target_count)
        
        # 5. Validation et formatage
        final_keywords = self._validate_and_format(keywords, target_count)
        
        print(f"✅ Mots-clés générés: {len(final_keywords)}")
        print(f"🎯 Exemples: {final_keywords[:5]}")
        
        return final_keywords
    
    def _extract_significant_words(self, text):
        """Extraction des mots significatifs du texte"""
        
        # Nettoyage du texte
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Extraction des mots
        words = text.split()
        
        # Filtrage des mots significatifs
        significant_words = []
        for word in words:
            if len(word) >= 4 and word not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 'would', 'could', 'should']:
                significant_words.append(word)
        
        return significant_words
    
    def _detect_domain(self, text, context_hint=None):
        """Détection automatique du domaine du texte"""
        
        text_lower = text.lower()
        
        # Scores par domaine
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            domain_scores[domain] = score
        
        # Domaine avec le plus haut score
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain
        
        # Fallback sur le contexte fourni
        if context_hint:
            return context_hint
        
        return "general"
    
    def _generate_keywords(self, top_words, domain, target_count):
        """Génération des mots-clés combinant plusieurs sources"""
        
        keywords = []
        
        # 1. Mots-clés du texte (priorité haute)
        keywords.extend(top_words[:10])
        
        # 2. Mots-clés du domaine détecté
        if domain in self.domain_keywords:
            domain_words = self.domain_keywords[domain][:5]
            keywords.extend(domain_words)
        
        # 3. Mots-clés d'action dynamiques
        action_words = self.action_keywords[:5]
        keywords.extend(action_words)
        
        # 4. Mots-clés universels pour compléter
        remaining = target_count - len(keywords)
        if remaining > 0:
            universal_subset = self.universal_keywords[:remaining]
            keywords.extend(universal_subset)
        
        # 5. Déduplication et limitation
        unique_keywords = list(dict.fromkeys(keywords))  # Garde l'ordre
        return unique_keywords[:target_count]
    
    def _validate_and_format(self, keywords, target_count):
        """Validation et formatage final des mots-clés"""
        
        # Vérification du nombre
        if len(keywords) < target_count * 0.8:  # 80% minimum
            print(f"⚠️ Nombre insuffisant de mots-clés: {len(keywords)}/{target_count}")
            # Compléter avec des mots-clés universels
            missing = target_count - len(keywords)
            additional = self.universal_keywords[:missing]
            keywords.extend(additional)
        
        # Limitation finale
        final_keywords = keywords[:target_count]
        
        # Formatage pour B-roll
        formatted_keywords = []
        for keyword in final_keywords:
            # Conversion en format "searchable"
            formatted = keyword.replace(' ', '_').lower()
            formatted_keywords.append(formatted)
        
        return formatted_keywords
    
    def generate_json_output(self, keywords):
        """Génération d'une sortie JSON formatée"""
        
        try:
            output = {
                "keywords": keywords,
                "count": len(keywords),
                "source": "heuristic_fallback",
                "confidence": 0.85
            }
            
            return json.dumps(output, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"❌ Erreur génération JSON: {e}")
            # Fallback simple
            return json.dumps({"keywords": keywords[:10]})

def test_heuristic_extractor():
    """Test du système heuristique"""
    
    print("🧪 TEST FALLBACK HEURISTIQUE")
    print("=" * 60)
    
    extractor = HeuristicKeywordExtractor()
    
    # Test avec le transcript du pipeline
    test_text = "EMDR movement sensation reprocessing lateralized movements people doing clinic got goofy looking things"
    
    print(f"📝 Texte de test: {test_text}")
    print()
    
    # Extraction des mots-clés
    keywords = extractor.extract_keywords_from_text(test_text, target_count=25)
    
    print()
    print("📊 RÉSULTATS FINAUX")
    print("-" * 40)
    print(f"🎯 Mots-clés générés: {len(keywords)}")
    print(f"📝 Liste complète: {keywords}")
    
    # Génération JSON
    json_output = extractor.generate_json_output(keywords)
    print()
    print("🔧 SORTIE JSON")
    print("-" * 40)
    print(json_output)
    
    return keywords, json_output

if __name__ == "__main__":
    keywords, json_output = test_heuristic_extractor()
    
    print()
    print("=" * 60)
    print("🏁 TEST HEURISTIQUE TERMINÉ")
    print(f"✅ {len(keywords)} mots-clés générés avec succès")
    
    input("\nAppuyez sur Entrée pour continuer...") 