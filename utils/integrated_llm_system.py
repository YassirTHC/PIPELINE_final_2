# 🚀 SYSTÈME LLM INTÉGRÉ COMPLET - PROMPTS MINIMALISTES + SPÉCIALISATION PIPELINE
# Architecture basée sur l'analyse brillante de l'utilisateur

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import des modules locaux
from optimized_llm import OptimizedLLM, create_optimized_llm
from pipeline_specialization import (
    detect_content_domain, 
    enhance_metadata_with_domain, 
    analyze_content_complexity,
    optimize_for_platform
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedLLMSystem:
    """
    Système LLM intégré complet :
    - Prompts minimalistes génériques
    - Spécialisation intelligente via pipeline
    - Génération de métadonnées complètes
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        self.llm = create_optimized_llm(base_url, model)
        logger.info(f"🚀 Système LLM intégré initialisé avec {self.llm.model}")
    
    def generate_complete_metadata(self, transcript: str, platform: str = 'tiktok') -> Tuple[bool, Dict[str, Any]]:
        """
        Génération complète de métadonnées avec spécialisation via pipeline
        
        Args:
            transcript: Transcription du contenu
            platform: Plateforme cible (tiktok, instagram, youtube)
            
        Returns:
            (success, metadata_dict)
        """
        logger.info("🎯 Démarrage génération métadonnées complètes...")
        
        # 1. Analyse de la complexité du contenu
        complexity_analysis = analyze_content_complexity(transcript)
        recommended_keywords = complexity_analysis['recommended_keywords']
        
        logger.info(f"📊 Complexité détectée: {complexity_analysis['complexity_level']}")
        logger.info(f"🎯 Mots-clés recommandés: {recommended_keywords}")
        
        # 2. Génération des métadonnées de base avec prompt minimaliste
        logger.info("🤖 Génération métadonnées de base avec LLM...")
        success, base_metadata = self.llm.generate_complete_metadata(transcript)
        
        if not success:
            logger.error("❌ Échec génération métadonnées de base")
            return False, {}
        
        # 3. Détection automatique du domaine
        logger.info("🎯 Détection automatique du domaine...")
        domain, confidence = detect_content_domain(transcript)
        
        # 4. Enrichissement avec la spécialisation du domaine
        logger.info(f"🚀 Enrichissement pour le domaine: {domain} (confiance: {confidence:.2f})")
        enhanced_metadata = enhance_metadata_with_domain(base_metadata, transcript)
        
        # 5. Optimisation pour la plateforme cible
        logger.info(f"🎯 Optimisation pour {platform}...")
        final_metadata = optimize_for_platform(enhanced_metadata, platform)
        
        # 6. Ajout des informations d'analyse
        final_metadata['analysis'] = {
            'complexity': complexity_analysis,
            'domain_detection': {
                'domain': domain,
                'confidence': confidence
            },
            'generation_method': 'minimalist_prompt + pipeline_specialization'
        }
        
        logger.info("✅ Génération métadonnées complètes terminée avec succès")
        return True, final_metadata
    
    def generate_keywords_only(self, transcript: str) -> Tuple[bool, List[str]]:
        """
        Génération de mots-clés uniquement avec spécialisation via pipeline
        """
        logger.info("🎯 Génération mots-clés avec spécialisation pipeline...")
        
        # 1. Analyse de la complexité
        complexity_analysis = analyze_content_complexity(transcript)
        recommended_count = complexity_analysis['recommended_keywords']
        
        # 2. Génération de base avec prompt minimaliste
        success, keywords = self.llm.generate_keywords(transcript, recommended_count)
        
        if not success:
            logger.error("❌ Échec génération mots-clés de base")
            return False, []
        
        # 3. Enrichissement via pipeline
        domain, confidence = detect_content_domain(transcript)
        
        # Créer un dictionnaire temporaire pour l'enrichissement
        temp_metadata = {'keywords': keywords}
        enhanced_metadata = enhance_metadata_with_domain(temp_metadata, transcript)
        
        final_keywords = enhanced_metadata['keywords']
        
        logger.info(f"✅ {len(final_keywords)} mots-clés générés avec spécialisation {domain}")
        return True, final_keywords
    
    def generate_title_hashtags_only(self, transcript: str, platform: str = 'tiktok') -> Tuple[bool, Dict[str, Any]]:
        """
        Génération titre + hashtags uniquement avec spécialisation via pipeline
        """
        logger.info("🎯 Génération titre + hashtags avec spécialisation pipeline...")
        
        # 1. Génération de base avec prompt minimaliste
        success, base_metadata = self.llm.generate_title_hashtags(transcript)
        
        if not success:
            logger.error("❌ Échec génération titre + hashtags de base")
            return False, {}
        
        # 2. Enrichissement via pipeline
        domain, confidence = detect_content_domain(transcript)
        
        enhanced_metadata = enhance_metadata_with_domain(base_metadata, transcript)
        
        # 3. Optimisation pour la plateforme
        final_metadata = optimize_for_platform(enhanced_metadata, platform)
        
        # 4. Ajout des informations d'analyse
        final_metadata['analysis'] = {
            'domain_detection': {
                'domain': domain,
                'confidence': confidence
            },
            'generation_method': 'minimalist_prompt + pipeline_specialization'
        }
        
        logger.info(f"✅ Titre et {len(final_metadata['hashtags'])} hashtags générés avec spécialisation {domain}")
        return True, final_metadata
    
    def batch_generate_metadata(self, transcripts: List[str], platform: str = 'tiktok') -> List[Tuple[bool, Dict[str, Any]]]:
        """
        Génération en lot de métadonnées pour plusieurs transcripts
        """
        logger.info(f"🚀 Génération en lot pour {len(transcripts)} transcripts...")
        
        results = []
        for i, transcript in enumerate(transcripts):
            logger.info(f"📝 Traitement transcript {i+1}/{len(transcripts)}...")
            
            success, metadata = self.generate_complete_metadata(transcript, platform)
            results.append((success, metadata))
            
            if success:
                logger.info(f"✅ Transcript {i+1} traité avec succès")
            else:
                logger.warning(f"⚠️ Transcript {i+1} en échec")
        
        logger.info(f"🎯 Traitement en lot terminé: {sum(1 for s, _ in results if s)}/{len(transcripts)} succès")
        return results
    
    def health_check(self) -> bool:
        """
        Vérification de la santé du système
        """
        try:
            # Test simple avec un transcript court
            test_transcript = "Test content for health check."
            success, _ = self.llm.generate_keywords(test_transcript, 3)
            return success
        except Exception as e:
            logger.error(f"❌ Échec health check: {e}")
            return False

# === FONCTIONS UTILITAIRES POUR INTÉGRATION DIRECTE ===

def create_integrated_system(base_url: str = None, model: str = None) -> IntegratedLLMSystem:
    """Factory pour créer le système intégré"""
    return IntegratedLLMSystem(base_url, model)

def generate_metadata_complete(transcript: str, platform: str = 'tiktok') -> Tuple[bool, Dict[str, Any]]:
    """Fonction utilitaire pour génération complète"""
    system = create_integrated_system()
    return system.generate_complete_metadata(transcript, platform)

def generate_keywords_enhanced(transcript: str) -> Tuple[bool, List[str]]:
    """Fonction utilitaire pour mots-clés enrichis"""
    system = create_integrated_system()
    return system.generate_keywords_only(transcript)

def generate_title_hashtags_enhanced(transcript: str, platform: str = 'tiktok') -> Tuple[bool, Dict[str, Any]]:
    """Fonction utilitaire pour titre + hashtags enrichis"""
    system = create_integrated_system()
    return system.generate_title_hashtags_only(transcript, platform)

# === TEST COMPLET DU SYSTÈME ===

if __name__ == "__main__":
    print("🚀 Test complet du système LLM intégré...")
    
    # Test avec différents types de contenu
    test_cases = [
        {
            'transcript': "EMDR therapy utilizes bilateral stimulation to process traumatic memories. The therapist guides the patient through eye movements while recalling distressing events.",
            'expected_domain': 'medical_psychology',
            'description': 'Contenu médical/psychologique'
        },
        {
            'transcript': "Start your own business and become a successful entrepreneur. Learn the strategies that top performers use to grow their companies and increase revenue.",
            'expected_domain': 'business_entrepreneurship',
            'description': 'Contenu business/entrepreneuriat'
        },
        {
            'transcript': "Artificial intelligence is transforming the future of technology. Machine learning algorithms are automating complex tasks and creating new opportunities.",
            'expected_domain': 'technology_ai',
            'description': 'Contenu technologie/IA'
        }
    ]
    
    system = create_integrated_system()
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"🧪 TEST {i+1}: {test_case['description']}")
        print(f"{'='*60}")
        
        transcript = test_case['transcript']
        expected_domain = test_case['expected_domain']
        
        print(f"📝 Transcript: {transcript[:80]}...")
        print(f"🎯 Domaine attendu: {expected_domain}")
        
        # Test 1: Mots-clés uniquement
        print(f"\n🎯 Test 1: Génération mots-clés...")
        success, keywords = system.generate_keywords_only(transcript)
        if success:
            print(f"✅ Mots-clés générés ({len(keywords)}): {keywords[:5]}...")
        else:
            print("❌ Échec génération mots-clés")
        
        # Test 2: Titre + hashtags
        print(f"\n🎯 Test 2: Génération titre + hashtags...")
        success, title_data = system.generate_title_hashtags_only(transcript, 'tiktok')
        if success:
            print(f"✅ Titre: {title_data['title']}")
            print(f"✅ Hashtags ({len(title_data['hashtags'])}): {title_data['hashtags'][:5]}...")
        else:
            print("❌ Échec génération titre + hashtags")
        
        # Test 3: Métadonnées complètes
        print(f"\n🎯 Test 3: Génération métadonnées complètes...")
        success, complete_metadata = system.generate_complete_metadata(transcript, 'tiktok')
        if success:
            print(f"✅ Titre: {complete_metadata['title']}")
            print(f"✅ Description: {complete_metadata['description'][:50]}...")
            print(f"✅ Mots-clés: {len(complete_metadata['keywords'])}")
            print(f"✅ Hashtags: {len(complete_metadata['hashtags'])}")
            print(f"🎯 Domaine détecté: {complete_metadata['analysis']['domain_detection']['domain']}")
            print(f"📊 Complexité: {complete_metadata['analysis']['complexity']['complexity_level']}")
        else:
            print("❌ Échec génération métadonnées complètes")
    
    # Test de santé
    print(f"\n{'='*60}")
    print("🏥 Test de santé du système...")
    health_ok = system.health_check()
    if health_ok:
        print("✅ Système en bonne santé")
    else:
        print("❌ Problème de santé détecté")
    
    print(f"\n🚀 Test complet terminé !") 