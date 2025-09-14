# 🎯 DÉTECTION DE DOMAINE RENFORCÉE - TF-IDF + SEUILS ADAPTATIFS
# Remplace la méthode de comptage simple par une approche plus robuste

import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDomainDetection:
    """Détection de domaine renforcée avec TF-IDF et seuils adaptatifs"""
    
    def __init__(self):
        # Domaines avec leurs caractéristiques enrichies
        self.domain_patterns = {
            'medical_psychology': {
                'keywords': [
                    'therapy', 'trauma', 'memory', 'brain', 'patient', 'healing', 'psychology', 
                    'treatment', 'mental', 'health', 'anxiety', 'depression', 'emdr', 'bilateral',
                    'therapist', 'counseling', 'recovery', 'wellness', 'mindfulness', 'stress'
                ],
                'visual_themes': ['medical', 'therapy', 'brain', 'healing', 'professional'],
                'hashtag_templates': ['#mentalhealth', '#therapy', '#healing', '#psychology', '#wellness'],
                'confidence_threshold': 0.30  # Seuil ajusté (était 0.35)
            },
            'business_entrepreneurship': {
                'keywords': [
                    'startup', 'business', 'entrepreneur', 'success', 'money', 'profit', 'revenue',
                    'growth', 'strategy', 'marketing', 'sales', 'leadership', 'company', 'investment',
                    'scaling', 'venture', 'capital', 'innovation', 'product', 'market', 'customer'
                ],
                'visual_themes': ['business', 'office', 'meeting', 'success', 'growth'],
                'hashtag_templates': ['#entrepreneur', '#business', '#success', '#startup', '#growth'],
                'confidence_threshold': 0.25  # Seuil ajusté (était 0.30)
            },
            'technology_ai': {
                'keywords': [
                    'ai', 'artificial intelligence', 'technology', 'innovation', 'future', 'digital',
                    'automation', 'machine', 'learning', 'data', 'software', 'algorithm', 'neural',
                    'network', 'deep learning', 'computer', 'programming', 'code', 'development'
                ],
                'visual_themes': ['technology', 'digital', 'innovation', 'future', 'automation'],
                'hashtag_templates': ['#ai', '#technology', '#innovation', '#future', '#digital'],
                'confidence_threshold': 0.35  # Seuil ajusté (était 0.40)
            },
            'lifestyle_wellness': {
                'keywords': [
                    'health', 'fitness', 'wellness', 'lifestyle', 'mindfulness', 'balance', 'happiness',
                    'growth', 'selfcare', 'motivation', 'meditation', 'yoga', 'exercise', 'nutrition',
                    'sleep', 'energy', 'vitality', 'peace', 'calm', 'zen', 'mindful'
                ],
                'visual_themes': ['lifestyle', 'wellness', 'fitness', 'nature', 'balance'],
                'hashtag_templates': ['#lifestyle', '#wellness', '#fitness', '#mindfulness', '#balance'],
                'confidence_threshold': 0.25  # Seuil ajusté (était 0.35)
            },
            'education_learning': {
                'keywords': [
                    'learning', 'education', 'knowledge', 'study', 'growth', 'skills', 'development',
                    'training', 'course', 'improvement', 'teaching', 'student', 'school', 'university',
                    'class', 'lesson', 'tutorial', 'workshop', 'seminar', 'lecture', 'research'
                ],
                'visual_themes': ['education', 'learning', 'study', 'growth', 'development'],
                'hashtag_templates': ['#education', '#learning', '#growth', '#skills', '#development'],
                'confidence_threshold': 0.25  # Seuil ajusté (était 0.35)
            },
            'finance_investment': {
                'keywords': [
                    'money', 'finance', 'investment', 'wealth', 'financial', 'budget', 'saving',
                    'trading', 'portfolio', 'retirement', 'stock', 'market', 'economy', 'banking',
                    'credit', 'debt', 'income', 'expense', 'planning', 'strategy', 'risk'
                ],
                'visual_themes': ['finance', 'money', 'investment', 'wealth', 'financial'],
                'hashtag_templates': ['#finance', '#investment', '#money', '#wealth', '#financial'],
                'confidence_threshold': 0.30  # Seuil ajusté (était 0.40)
            }
        }
        
        # Initialisation du vectoriseur TF-IDF
        self.vectorizer = None
        self.domain_vectors = {}
        self._initialize_tfidf()
    
    def _initialize_tfidf(self):
        """Initialise le vectoriseur TF-IDF avec les domaines"""
        try:
            # Préparer les textes de référence pour chaque domaine
            domain_texts = []
            domain_names = []
            
            for domain, info in self.domain_patterns.items():
                # Créer un texte de référence pour chaque domaine
                reference_text = " ".join(info['keywords'])
                # Ajouter des variations et synonymes
                reference_text += f" {domain.replace('_', ' ')}"
                
                domain_texts.append(reference_text)
                domain_names.append(domain)
            
            # Entraîner le vectoriseur TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            # Vectoriser les textes de référence
            domain_vectors = self.vectorizer.fit_transform(domain_texts)
            
            # Stocker les vecteurs pour chaque domaine
            for i, domain in enumerate(domain_names):
                self.domain_vectors[domain] = domain_vectors[i]
            
            logger.info(f"✅ TF-IDF initialisé avec {len(domain_names)} domaines")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation TF-IDF: {e}")
            self.vectorizer = None
    
    def detect_domain_enhanced(self, transcript: str) -> Tuple[str, float]:
        """
        Détection de domaine renforcée avec TF-IDF et seuils adaptatifs
        """
        if not self.vectorizer or not self.domain_vectors:
            logger.warning("⚠️ TF-IDF non disponible, fallback vers méthode simple")
            return self._detect_domain_simple(transcript)
        
        try:
            # Vectoriser le transcript
            transcript_vector = self.vectorizer.transform([transcript])
            
            # Calculer les similarités avec tous les domaines
            similarities = {}
            for domain, domain_vector in self.domain_vectors.items():
                similarity = cosine_similarity(transcript_vector, domain_vector)[0][0]
                similarities[domain] = float(similarity)
            
            # Trouver le domaine avec la plus haute similarité
            best_domain = max(similarities, key=similarities.get)
            best_score = similarities[best_domain]
            
            # Seuil adaptatif basé sur le domaine
            threshold = self.domain_patterns[best_domain]['confidence_threshold']
            
            if best_score >= threshold:
                logger.info(f"🎯 Domaine détecté (TF-IDF): {best_domain} (confiance: {best_score:.3f})")
                return best_domain, best_score
            else:
                logger.info(f"🎯 Score insuffisant: {best_domain} ({best_score:.3f}) < {threshold}")
                return 'generic', best_score
                
        except Exception as e:
            logger.error(f"❌ Erreur détection TF-IDF: {e}")
            return self._detect_domain_simple(transcript)
    
    def _detect_domain_simple(self, transcript: str) -> Tuple[str, float]:
        """Méthode de fallback basée sur le comptage de mots"""
        transcript_lower = transcript.lower()
        domain_scores = {}
        
        for domain, info in self.domain_patterns.items():
            score = 0
            keywords = info['keywords']
            
            for keyword in keywords:
                if keyword in transcript_lower:
                    score += 1
            
            # Normaliser le score
            if keywords:
                domain_scores[domain] = score / len(keywords)
            else:
                domain_scores[domain] = 0
        
        # Trouver le domaine avec le score le plus élevé
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            best_score = domain_scores[best_domain]
            
            # Seuil de confiance minimum pour la méthode simple
            if best_score >= 0.10:  # Seuil ajusté (était 0.15)
                logger.info(f"🎯 Domaine détecté (simple): {best_domain} (confiance: {best_score:.3f})")
                return best_domain, best_score
            else:
                logger.info(f"🎯 Score insuffisant (simple): {best_domain} ({best_score:.3f}) < 0.10")
                return 'generic', 0.0
        else:
            return 'generic', 0.0
    
    def get_domain_info(self, domain: str) -> Dict[str, Any]:
        """Récupère les informations d'un domaine"""
        return self.domain_patterns.get(domain, {})
    
    def get_all_domains(self) -> List[str]:
        """Liste tous les domaines supportés"""
        return list(self.domain_patterns.keys())
    
    def add_domain(self, name: str, keywords: List[str], visual_themes: List[str], 
                   hashtag_templates: List[str], confidence_threshold: float = 0.35):
        """Ajoute un nouveau domaine dynamiquement"""
        self.domain_patterns[name] = {
            'keywords': keywords,
            'visual_themes': visual_themes,
            'hashtag_templates': hashtag_templates,
            'confidence_threshold': confidence_threshold
        }
        
        # Réinitialiser TF-IDF avec le nouveau domaine
        self._initialize_tfidf()
        logger.info(f"✅ Nouveau domaine ajouté: {name}")
    
    def analyze_domain_distribution(self, transcript: str) -> Dict[str, float]:
        """Analyse la distribution des domaines dans un transcript"""
        if not self.vectorizer:
            return {}
        
        try:
            transcript_vector = self.vectorizer.transform([transcript])
            similarities = {}
            
            for domain, domain_vector in self.domain_vectors.items():
                similarity = cosine_similarity(transcript_vector, domain_vector)[0][0]
                similarities[domain] = float(similarity)
            
            # Normaliser les scores
            total_similarity = sum(similarities.values())
            if total_similarity > 0:
                normalized = {d: s/total_similarity for d, s in similarities.items()}
            else:
                normalized = similarities
            
            return normalized
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse distribution: {e}")
            return {}

# === INSTANCE GLOBALE ===
enhanced_domain_detection = EnhancedDomainDetection()

# === FONCTIONS UTILITAIRES ===
def detect_domain_enhanced(transcript: str) -> Tuple[str, float]:
    """Détection de domaine renforcée"""
    return enhanced_domain_detection.detect_domain_enhanced(transcript)

def get_domain_info(domain: str) -> Dict[str, Any]:
    """Informations d'un domaine"""
    return enhanced_domain_detection.get_domain_info(domain)

def analyze_domain_distribution(transcript: str) -> Dict[str, float]:
    """Distribution des domaines"""
    return enhanced_domain_detection.analyze_domain_distribution(transcript)

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("🧪 Test de la détection de domaine renforcée...")
    
    # Test avec différents types de contenu
    test_cases = [
        ("EMDR therapy utilizes bilateral stimulation to process traumatic memories. The therapist guides the patient through eye movements while recalling distressing events.", "medical_psychology"),
        ("Start your own business and become a successful entrepreneur. Learn the strategies that top performers use to grow their companies and increase revenue.", "business_entrepreneurship"),
        ("Artificial intelligence is transforming the future of technology. Machine learning algorithms are automating complex tasks and creating new opportunities.", "technology_ai"),
        ("Transform your life with mindfulness and wellness practices. Learn how to balance work and personal life while maintaining mental and physical health.", "lifestyle_wellness"),
        ("Master your finances and build wealth through smart investment strategies. Learn how to budget effectively, save money, and invest in stocks.", "finance_investment"),
        ("This is a generic content about various topics that doesn't fit into specific categories.", "generic")
    ]
    
    for transcript, expected_domain in test_cases:
        print(f"\n📝 Test: {transcript[:60]}...")
        
        # Détection renforcée
        detected_domain, confidence = detect_domain_enhanced(transcript)
        print(f"🎯 Domaine détecté: {detected_domain} (confiance: {confidence:.3f})")
        print(f"✅ Attendu: {expected_domain}")
        
        # Analyse de distribution
        distribution = analyze_domain_distribution(transcript)
        print(f"📊 Distribution: {dict(list(distribution.items())[:3])}")
        
        # Validation
        if detected_domain == expected_domain:
            print("✅ CORRECT !")
        else:
            print("❌ INCORRECT")
    
    print("\n�� Test terminé !") 