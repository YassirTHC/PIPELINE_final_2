ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ SYSTÃƒË†ME LLM INTÃƒâ€°GRÃƒâ€° COMPLET - PROMPTS MINIMALISTES + SPÃƒâ€°CIALISATION PIPELINE
# Architecture basÃƒÂ©e sur l'analyse brillante de l'utilisateur

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
    SystÃƒÂ¨me LLM intÃƒÂ©grÃƒÂ© complet :
    - Prompts minimalistes gÃƒÂ©nÃƒÂ©riques
    - SpÃƒÂ©cialisation intelligente via pipeline
    - GÃƒÂ©nÃƒÂ©ration de mÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        self.llm = create_optimized_llm(base_url, model)
        logger.info(f"Ã°Å¸Å¡â‚¬ SystÃƒÂ¨me LLM intÃƒÂ©grÃƒÂ© initialisÃƒÂ© avec {self.llm.model}")
    
    def generate_complete_metadata(self, transcript: str, platform: str = 'tiktok') -> Tuple[bool, Dict[str, Any]]:
        """
        GÃƒÂ©nÃƒÂ©ration complÃƒÂ¨te de mÃƒÂ©tadonnÃƒÂ©es avec spÃƒÂ©cialisation via pipeline
        
        Args:
            transcript: Transcription du contenu
            platform: Plateforme cible (tiktok, instagram, youtube)
            
        Returns:
            (success, metadata_dict)
        """
        logger.info("Ã°Å¸Å½Â¯ DÃƒÂ©marrage gÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes...")
        
        # 1. Analyse de la complexitÃƒÂ© du contenu
        complexity_analysis = analyze_content_complexity(transcript)
        recommended_keywords = complexity_analysis['recommended_keywords']
        
        logger.info(f"Ã°Å¸â€œÅ  ComplexitÃƒÂ© dÃƒÂ©tectÃƒÂ©e: {complexity_analysis['complexity_level']}")
        logger.info(f"Ã°Å¸Å½Â¯ Mots-clÃƒÂ©s recommandÃƒÂ©s: {recommended_keywords}")
        
        # 2. GÃƒÂ©nÃƒÂ©ration des mÃƒÂ©tadonnÃƒÂ©es de base avec prompt minimaliste
        logger.info("Ã°Å¸Â¤â€“ GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es de base avec LLM...")
        success, base_metadata = self.llm.generate_complete_metadata(transcript)
        
        if not success:
            logger.error("Ã¢ÂÅ’ Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es de base")
            return False, {}
        
        # 3. DÃƒÂ©tection automatique du domaine
        logger.info("Ã°Å¸Å½Â¯ DÃƒÂ©tection automatique du domaine...")
        domain, confidence = detect_content_domain(transcript)
        
        # 4. Enrichissement avec la spÃƒÂ©cialisation du domaine
        logger.info(f"Ã°Å¸Å¡â‚¬ Enrichissement pour le domaine: {domain} (confiance: {confidence:.2f})")
        enhanced_metadata = enhance_metadata_with_domain(base_metadata, transcript)
        
        # 5. Optimisation pour la plateforme cible
        logger.info(f"Ã°Å¸Å½Â¯ Optimisation pour {platform}...")
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
        
        logger.info("Ã¢Å“â€¦ GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes terminÃƒÂ©e avec succÃƒÂ¨s")
        return True, final_metadata
    
    def generate_keywords_only(self, transcript: str) -> Tuple[bool, List[str]]:
        """
        GÃƒÂ©nÃƒÂ©ration de mots-clÃƒÂ©s uniquement avec spÃƒÂ©cialisation via pipeline
        """
        logger.info("Ã°Å¸Å½Â¯ GÃƒÂ©nÃƒÂ©ration mots-clÃƒÂ©s avec spÃƒÂ©cialisation pipeline...")
        
        # 1. Analyse de la complexitÃƒÂ©
        complexity_analysis = analyze_content_complexity(transcript)
        recommended_count = complexity_analysis['recommended_keywords']
        
        # 2. GÃƒÂ©nÃƒÂ©ration de base avec prompt minimaliste
        success, keywords = self.llm.generate_keywords(transcript, recommended_count)
        
        if not success:
            logger.error("Ã¢ÂÅ’ Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration mots-clÃƒÂ©s de base")
            return False, []
        
        # 3. Enrichissement via pipeline
        domain, confidence = detect_content_domain(transcript)
        
        # CrÃƒÂ©er un dictionnaire temporaire pour l'enrichissement
        temp_metadata = {'keywords': keywords}
        enhanced_metadata = enhance_metadata_with_domain(temp_metadata, transcript)
        
        final_keywords = enhanced_metadata['keywords']
        
        logger.info(f"Ã¢Å“â€¦ {len(final_keywords)} mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©rÃƒÂ©s avec spÃƒÂ©cialisation {domain}")
        return True, final_keywords
    
    def generate_title_hashtags_only(self, transcript: str, platform: str = 'tiktok') -> Tuple[bool, Dict[str, Any]]:
        """
        GÃƒÂ©nÃƒÂ©ration titre + hashtags uniquement avec spÃƒÂ©cialisation via pipeline
        """
        logger.info("Ã°Å¸Å½Â¯ GÃƒÂ©nÃƒÂ©ration titre + hashtags avec spÃƒÂ©cialisation pipeline...")
        
        # 1. GÃƒÂ©nÃƒÂ©ration de base avec prompt minimaliste
        success, base_metadata = self.llm.generate_title_hashtags(transcript)
        
        if not success:
            logger.error("Ã¢ÂÅ’ Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration titre + hashtags de base")
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
        
        logger.info(f"Ã¢Å“â€¦ Titre et {len(final_metadata['hashtags'])} hashtags gÃƒÂ©nÃƒÂ©rÃƒÂ©s avec spÃƒÂ©cialisation {domain}")
        return True, final_metadata
    
    def batch_generate_metadata(self, transcripts: List[str], platform: str = 'tiktok') -> List[Tuple[bool, Dict[str, Any]]]:
        """
        GÃƒÂ©nÃƒÂ©ration en lot de mÃƒÂ©tadonnÃƒÂ©es pour plusieurs transcripts
        """
        logger.info(f"Ã°Å¸Å¡â‚¬ GÃƒÂ©nÃƒÂ©ration en lot pour {len(transcripts)} transcripts...")
        
        results = []
        for i, transcript in enumerate(transcripts):
            logger.info(f"Ã°Å¸â€œÂ Traitement transcript {i+1}/{len(transcripts)}...")
            
            success, metadata = self.generate_complete_metadata(transcript, platform)
            results.append((success, metadata))
            
            if success:
                logger.info(f"Ã¢Å“â€¦ Transcript {i+1} traitÃƒÂ© avec succÃƒÂ¨s")
            else:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Transcript {i+1} en ÃƒÂ©chec")
        
        logger.info(f"Ã°Å¸Å½Â¯ Traitement en lot terminÃƒÂ©: {sum(1 for s, _ in results if s)}/{len(transcripts)} succÃƒÂ¨s")
        return results
    
    def health_check(self) -> bool:
        """
        VÃƒÂ©rification de la santÃƒÂ© du systÃƒÂ¨me
        """
        try:
            # Test simple avec un transcript court
            test_transcript = "Test content for health check."
            success, _ = self.llm.generate_keywords(test_transcript, 3)
            return success
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Ãƒâ€°chec health check: {e}")
            return False

# === FONCTIONS UTILITAIRES POUR INTÃƒâ€°GRATION DIRECTE ===

def create_integrated_system(base_url: str = None, model: str = None) -> IntegratedLLMSystem:
    """Factory pour crÃƒÂ©er le systÃƒÂ¨me intÃƒÂ©grÃƒÂ©"""
    return IntegratedLLMSystem(base_url, model)

def generate_metadata_complete(transcript: str, platform: str = 'tiktok') -> Tuple[bool, Dict[str, Any]]:
    """Fonction utilitaire pour gÃƒÂ©nÃƒÂ©ration complÃƒÂ¨te"""
    system = create_integrated_system()
    return system.generate_complete_metadata(transcript, platform)

def generate_keywords_enhanced(transcript: str) -> Tuple[bool, List[str]]:
    """Fonction utilitaire pour mots-clÃƒÂ©s enrichis"""
    system = create_integrated_system()
    return system.generate_keywords_only(transcript)

def generate_title_hashtags_enhanced(transcript: str, platform: str = 'tiktok') -> Tuple[bool, Dict[str, Any]]:
    """Fonction utilitaire pour titre + hashtags enrichis"""
    system = create_integrated_system()
    return system.generate_title_hashtags_only(transcript, platform)

# === TEST COMPLET DU SYSTÃƒË†ME ===

if __name__ == "__main__":
    print("Ã°Å¸Å¡â‚¬ Test complet du systÃƒÂ¨me LLM intÃƒÂ©grÃƒÂ©...")
    
    # Test avec diffÃƒÂ©rents types de contenu
    test_cases = [
        {
            'transcript': "EMDR therapy utilizes bilateral stimulation to process traumatic memories. The therapist guides the patient through eye movements while recalling distressing events.",
            'expected_domain': 'medical_psychology',
            'description': 'Contenu mÃƒÂ©dical/psychologique'
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
        print(f"Ã°Å¸Â§Âª TEST {i+1}: {test_case['description']}")
        print(f"{'='*60}")
        
        transcript = test_case['transcript']
        expected_domain = test_case['expected_domain']
        
        print(f"Ã°Å¸â€œÂ Transcript: {transcript[:80]}...")
        print(f"Ã°Å¸Å½Â¯ Domaine attendu: {expected_domain}")
        
        # Test 1: Mots-clÃƒÂ©s uniquement
        print(f"\nÃ°Å¸Å½Â¯ Test 1: GÃƒÂ©nÃƒÂ©ration mots-clÃƒÂ©s...")
        success, keywords = system.generate_keywords_only(transcript)
        if success:
            print(f"Ã¢Å“â€¦ Mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©rÃƒÂ©s ({len(keywords)}): {keywords[:5]}...")
        else:
            print("Ã¢ÂÅ’ Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration mots-clÃƒÂ©s")
        
        # Test 2: Titre + hashtags
        print(f"\nÃ°Å¸Å½Â¯ Test 2: GÃƒÂ©nÃƒÂ©ration titre + hashtags...")
        success, title_data = system.generate_title_hashtags_only(transcript, 'tiktok')
        if success:
            print(f"Ã¢Å“â€¦ Titre: {title_data['title']}")
            print(f"Ã¢Å“â€¦ Hashtags ({len(title_data['hashtags'])}): {title_data['hashtags'][:5]}...")
        else:
            print("Ã¢ÂÅ’ Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration titre + hashtags")
        
        # Test 3: MÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes
        print(f"\nÃ°Å¸Å½Â¯ Test 3: GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes...")
        success, complete_metadata = system.generate_complete_metadata(transcript, 'tiktok')
        if success:
            print(f"Ã¢Å“â€¦ Titre: {complete_metadata['title']}")
            print(f"Ã¢Å“â€¦ Description: {complete_metadata['description'][:50]}...")
            print(f"Ã¢Å“â€¦ Mots-clÃƒÂ©s: {len(complete_metadata['keywords'])}")
            print(f"Ã¢Å“â€¦ Hashtags: {len(complete_metadata['hashtags'])}")
            print(f"Ã°Å¸Å½Â¯ Domaine dÃƒÂ©tectÃƒÂ©: {complete_metadata['analysis']['domain_detection']['domain']}")
            print(f"Ã°Å¸â€œÅ  ComplexitÃƒÂ©: {complete_metadata['analysis']['complexity']['complexity_level']}")
        else:
            print("Ã¢ÂÅ’ Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes")
    
    # Test de santÃƒÂ©
    print(f"\n{'='*60}")
    print("Ã°Å¸ÂÂ¥ Test de santÃƒÂ© du systÃƒÂ¨me...")
    health_ok = system.health_check()
    if health_ok:
        print("Ã¢Å“â€¦ SystÃƒÂ¨me en bonne santÃƒÂ©")
    else:
        print("Ã¢ÂÅ’ ProblÃƒÂ¨me de santÃƒÂ© dÃƒÂ©tectÃƒÂ©")
    
    print(f"\nÃ°Å¸Å¡â‚¬ Test complet terminÃƒÂ© !") 

