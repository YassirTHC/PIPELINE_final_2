# 🎯 SPÉCIALISATION VIA PIPELINE - INTELLIGENCE HORS PROMPTS
# Basé sur l'analyse brillante de l'utilisateur : prompts génériques + spécialisation intelligente

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineSpecialization:
    """Système de spécialisation intelligent via pipeline (pas dans les prompts)"""
    
    def __init__(self):
        # Domaines prédéfinis avec leurs caractéristiques
        self.domain_patterns = {
            'medical_psychology': {
                'keywords': ['therapy', 'trauma', 'memory', 'brain', 'patient', 'healing', 'psychology', 'treatment', 'mental', 'health'],
                'visual_themes': ['medical', 'therapy', 'brain', 'healing', 'professional'],
                'hashtag_templates': ['#mentalhealth', '#therapy', '#healing', '#psychology', '#wellness']
            },
            'business_entrepreneurship': {
                'keywords': ['startup', 'business', 'entrepreneur', 'success', 'money', 'growth', 'strategy', 'marketing', 'sales', 'leadership'],
                'visual_themes': ['business', 'office', 'meeting', 'success', 'growth'],
                'hashtag_templates': ['#entrepreneur', '#business', '#success', '#startup', '#growth']
            },
            'technology_ai': {
                'keywords': ['ai', 'technology', 'innovation', 'future', 'digital', 'automation', 'machine', 'learning', 'data', 'software'],
                'visual_themes': ['technology', 'digital', 'innovation', 'future', 'automation'],
                'hashtag_templates': ['#ai', '#technology', '#innovation', '#future', '#digital']
            },
            'lifestyle_wellness': {
                'keywords': ['health', 'fitness', 'wellness', 'lifestyle', 'mindfulness', 'balance', 'happiness', 'growth', 'selfcare', 'motivation'],
                'visual_themes': ['lifestyle', 'wellness', 'fitness', 'nature', 'balance'],
                'hashtag_templates': ['#lifestyle', '#wellness', '#fitness', '#mindfulness', '#balance']
            },
            'education_learning': {
                'keywords': ['learning', 'education', 'knowledge', 'study', 'growth', 'skills', 'development', 'training', 'course', 'improvement'],
                'visual_themes': ['education', 'learning', 'study', 'growth', 'development'],
                'hashtag_templates': ['#education', '#learning', '#growth', '#skills', '#development']
            },
            'finance_investment': {
                'keywords': ['money', 'finance', 'investment', 'wealth', 'financial', 'budget', 'saving', 'trading', 'portfolio', 'retirement'],
                'visual_themes': ['finance', 'money', 'investment', 'wealth', 'financial'],
                'hashtag_templates': ['#finance', '#investment', '#money', '#wealth', '#financial']
            }
        }
        
        # Mots-clés de détection de domaine
        self.domain_detection = {
            'medical_psychology': ['therapy', 'trauma', 'memory', 'brain', 'patient', 'healing', 'psychology', 'mental', 'health', 'treatment', 'anxiety', 'depression', 'emdr', 'bilateral'],
            'business_entrepreneurship': ['startup', 'business', 'entrepreneur', 'success', 'money', 'profit', 'revenue', 'growth', 'strategy', 'marketing', 'sales', 'leadership', 'company'],
            'technology_ai': ['ai', 'artificial intelligence', 'technology', 'innovation', 'future', 'digital', 'automation', 'machine', 'learning', 'data', 'software', 'algorithm', 'neural'],
            'lifestyle_wellness': ['health', 'fitness', 'wellness', 'lifestyle', 'mindfulness', 'balance', 'happiness', 'growth', 'selfcare', 'motivation', 'meditation', 'yoga'],
            'education_learning': ['learning', 'education', 'knowledge', 'study', 'growth', 'skills', 'development', 'training', 'course', 'improvement', 'teaching', 'student'],
            'finance_investment': ['money', 'finance', 'investment', 'wealth', 'financial', 'budget', 'saving', 'trading', 'portfolio', 'retirement', 'stock', 'market']
        }
    
    def detect_domain(self, transcript: str) -> Tuple[str, float]:
        """
        Détecte le domaine principal du transcript
        Returns: (domain_name, confidence_score)
        """
        transcript_lower = transcript.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_detection.items():
            score = 0
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
            
            # Seuil de confiance minimum
            if best_score >= 0.1:  # Au moins 10% des mots-clés trouvés
                logger.info(f"🎯 Domaine détecté: {best_domain} (confiance: {best_score:.2f})")
                return best_domain, best_score
            else:
                logger.info("🎯 Aucun domaine spécifique détecté, utilisation du domaine générique")
                return 'generic', 0.0
        else:
            return 'generic', 0.0
    
    def enhance_keywords(self, base_keywords: List[str], domain: str, transcript: str) -> List[str]:
        """
        Améliore les mots-clés de base avec la spécialisation du domaine
        """
        if domain == 'generic' or domain not in self.domain_patterns:
            return base_keywords
        
        domain_info = self.domain_patterns[domain]
        enhanced_keywords = base_keywords.copy()
        
        # Ajouter des mots-clés spécifiques au domaine si manquants
        for domain_keyword in domain_info['keywords']:
            if domain_keyword not in enhanced_keywords and len(enhanced_keywords) < 20:
                enhanced_keywords.append(domain_keyword)
        
        # Ajouter des mots-clés contextuels du transcript
        transcript_words = re.findall(r'\b\w{4,}\b', transcript.lower())
        for word in transcript_words:
            if word not in enhanced_keywords and len(enhanced_keywords) < 25:
                # Vérifier que le mot est pertinent
                if any(domain_word in word or word in domain_word for domain_word in domain_info['keywords']):
                    enhanced_keywords.append(word)
        
        logger.info(f"🎯 Mots-clés enrichis pour le domaine {domain}: {len(enhanced_keywords)} total")
        return enhanced_keywords[:25]  # Limiter à 25 mots-clés
    
    def enhance_hashtags(self, base_hashtags: List[str], domain: str) -> List[str]:
        """
        Améliore les hashtags avec la spécialisation du domaine
        """
        if domain == 'generic' or domain not in self.domain_patterns:
            return base_hashtags
        
        domain_info = self.domain_patterns[domain]
        enhanced_hashtags = base_hashtags.copy()
        
        # Ajouter des hashtags spécifiques au domaine
        for template in domain_info['hashtag_templates']:
            if template not in enhanced_hashtags and len(enhanced_hashtags) < 15:
                enhanced_hashtags.append(template)
        
        # Ajouter des hashtags génériques populaires
        generic_hashtags = ['#viral', '#trending', '#fyp', '#foryou', '#shorts']
        for tag in generic_hashtags:
            if tag not in enhanced_hashtags and len(enhanced_hashtags) < 18:
                enhanced_hashtags.append(tag)
        
        logger.info(f"🎯 Hashtags enrichis pour le domaine {domain}: {len(enhanced_hashtags)} total")
        return enhanced_hashtags[:18]  # Limiter à 18 hashtags
    
    def suggest_visual_themes(self, domain: str) -> List[str]:
        """
        Suggère des thèmes visuels pour la sélection B-roll
        """
        if domain == 'generic' or domain not in self.domain_patterns:
            return ['general', 'lifestyle', 'people', 'nature']
        
        domain_info = self.domain_patterns[domain]
        return domain_info['visual_themes']
    
    def create_domain_specific_prompt(self, base_prompt: str, domain: str) -> str:
        """
        Crée un prompt spécifique au domaine (optionnel, pour cas avancés)
        """
        if domain == 'generic':
            return base_prompt
        
        domain_info = self.domain_patterns[domain]
        
        # Ajouter des instructions spécifiques au domaine
        domain_instruction = f"\n\nContext: This content is related to {domain.replace('_', ' ')}. Focus on relevant terminology and concepts."
        
        return base_prompt + domain_instruction
    
    def analyze_content_complexity(self, transcript: str) -> Dict[str, Any]:
        """
        Analyse la complexité du contenu pour adapter la génération
        """
        words = transcript.split()
        sentences = re.split(r'[.!?]+', transcript)
        
        analysis = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': len(words) / max(len([s for s in sentences if s.strip()]), 1),
            'complexity_level': 'medium'
        }
        
        # Déterminer le niveau de complexité
        if analysis['avg_sentence_length'] < 10:
            analysis['complexity_level'] = 'simple'
        elif analysis['avg_sentence_length'] > 20:
            analysis['complexity_level'] = 'complex'
        
        # Adapter le nombre de mots-clés selon la complexité
        if analysis['complexity_level'] == 'simple':
            analysis['recommended_keywords'] = 8
        elif analysis['complexity_level'] == 'medium':
            analysis['recommended_keywords'] = 12
        else:
            analysis['recommended_keywords'] = 18
        
        logger.info(f"📊 Analyse complexité: {analysis['complexity_level']} - {analysis['recommended_keywords']} mots-clés recommandés")
        return analysis
    
    def optimize_for_platform(self, metadata: Dict[str, Any], platform: str = 'tiktok') -> Dict[str, Any]:
        """
        Optimise les métadonnées pour une plateforme spécifique
        """
        optimized = metadata.copy()
        
        if platform == 'tiktok':
            # TikTok: hashtags populaires, titre court
            if 'title' in optimized and len(optimized['title']) > 50:
                optimized['title'] = optimized['title'][:47] + "..."
            
            # Ajouter hashtags TikTok populaires
            tiktok_tags = ['#fyp', '#foryou', '#viral', '#trending', '#shorts']
            if 'hashtags' in optimized:
                for tag in tiktok_tags:
                    if tag not in optimized['hashtags']:
                        optimized['hashtags'].append(tag)
        
        elif platform == 'instagram':
            # Instagram: description plus longue, hashtags nichés
            if 'description' in optimized and len(optimized['description']) < 100:
                optimized['description'] += " 💡 Swipe for more insights!"
        
        elif platform == 'youtube':
            # YouTube: titre descriptif, description détaillée
            if 'title' in optimized and len(optimized['title']) < 30:
                optimized['title'] += " - Complete Guide"
        
        logger.info(f"🎯 Métadonnées optimisées pour {platform}")
        return optimized

# === INSTANCE GLOBALE ===
pipeline_specialization = PipelineSpecialization()

# === FONCTIONS UTILITAIRES ===
def detect_content_domain(transcript: str) -> Tuple[str, float]:
    """Détecte le domaine du contenu"""
    return pipeline_specialization.detect_domain(transcript)

def enhance_metadata_with_domain(metadata: Dict[str, Any], transcript: str) -> Dict[str, Any]:
    """Enrichit les métadonnées avec la spécialisation du domaine"""
    domain, confidence = pipeline_specialization.detect_domain(transcript)
    
    enhanced = metadata.copy()
    
    if 'keywords' in enhanced:
        enhanced['keywords'] = pipeline_specialization.enhance_keywords(
            enhanced['keywords'], domain, transcript
        )
    
    if 'hashtags' in enhanced:
        enhanced['hashtags'] = pipeline_specialization.enhance_hashtags(
            enhanced['hashtags'], domain
        )
    
    # Ajouter des informations de domaine
    enhanced['domain'] = domain
    enhanced['domain_confidence'] = confidence
    enhanced['visual_themes'] = pipeline_specialization.suggest_visual_themes(domain)
    
    return enhanced

def analyze_content_complexity(transcript: str) -> Dict[str, Any]:
    """Analyse la complexité du contenu"""
    return pipeline_specialization.analyze_content_complexity(transcript)

def optimize_for_platform(metadata: Dict[str, Any], platform: str = 'tiktok') -> Dict[str, Any]:
    """Optimise pour une plateforme spécifique"""
    return pipeline_specialization.optimize_for_platform(metadata, platform)

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("🎯 Test du système de spécialisation via pipeline...")
    
    # Test avec différents types de contenu
    test_cases = [
        ("EMDR therapy utilizes bilateral stimulation to process traumatic memories. The therapist guides the patient through eye movements while recalling distressing events.", "medical_psychology"),
        ("Start your own business and become a successful entrepreneur. Learn the strategies that top performers use to grow their companies and increase revenue.", "business_entrepreneurship"),
        ("Artificial intelligence is transforming the future of technology. Machine learning algorithms are automating complex tasks and creating new opportunities.", "technology_ai"),
        ("This is a generic content about various topics that doesn't fit into specific categories.", "generic")
    ]
    
    for transcript, expected_domain in test_cases:
        print(f"\n📝 Test: {transcript[:50]}...")
        
        # Détection de domaine
        detected_domain, confidence = detect_content_domain(transcript)
        print(f"🎯 Domaine détecté: {detected_domain} (confiance: {confidence:.2f})")
        print(f"✅ Attendu: {expected_domain}")
        
        # Analyse de complexité
        complexity = analyze_content_complexity(transcript)
        print(f"📊 Complexité: {complexity['complexity_level']} - {complexity['recommended_keywords']} mots-clés recommandés")
        
        # Test d'enrichissement
        base_metadata = {
            'keywords': ['test', 'example'],
            'hashtags': ['#test']
        }
        
        enhanced = enhance_metadata_with_domain(base_metadata, transcript)
        print(f"🚀 Métadonnées enrichies: {len(enhanced['keywords'])} mots-clés, {len(enhanced['hashtags'])} hashtags")
        print(f"🎨 Thèmes visuels: {enhanced['visual_themes']}") 