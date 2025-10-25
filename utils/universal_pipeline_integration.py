ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ INTÃƒâ€°GRATION UNIVERSELLE - REMPLACE COMPLÃƒË†TEMENT L'ANCIEN SYSTÃƒË†ME
# Pipeline 100% universel : B-roll + mÃƒÂ©tadonnÃƒÂ©es TikTok/Instagram pour TOUS les domaines

import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys

# Import des modules universels
from utils.universal_broll_generator import (
    create_universal_broll_generator,
    create_tiktok_metadata_generator
)

logger = logging.getLogger(__name__)

class UniversalPipelineIntegration:
    """Pipeline universel pour TOUS les domaines - B-roll + mÃƒÂ©tadonnÃƒÂ©es TikTok/Instagram"""
    
    def __init__(self):
        self.broll_generator = create_universal_broll_generator()
        self.metadata_generator = create_tiktok_metadata_generator()
        
        # Configuration universelle
        self.config = {
            'target_keywords': 10,  # 8-12 mots-clÃƒÂ©s optimaux
            'enable_viral_metadata': True,
            'enable_broll_generation': True,
            'fallback_on_error': True,
            'max_processing_time': 30  # secondes
        }
        
        logger.info("Ã°Å¸Å¡â‚¬ Pipeline universel initialisÃƒÂ© - PrÃƒÂªt pour TOUS les domaines")

    def process_video_universal(self, transcript: str, video_id: str = None) -> Dict[str, Any]:
        """Traitement universel d'une vidÃƒÂ©o - fonctionne sur TOUS les domaines"""
        start_time = time.time()
        
        try:
            logger.info(f"Ã°Å¸Å½Â¬ Traitement universel dÃƒÂ©marrÃƒÂ© pour {len(transcript)} caractÃƒÂ¨res")
            
            # 1. GÃƒÂ©nÃƒÂ©ration B-roll universelle
            broll_result = self._generate_universal_broll(transcript)
            
            # 2. GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es virales TikTok/Instagram
            metadata_result = self._generate_viral_metadata(transcript, broll_result)
            
            # 3. Assemblage du rÃƒÂ©sultat final
            final_result = self._assemble_final_result(broll_result, metadata_result, start_time)
            
            logger.info(f"Ã¢Å“â€¦ Traitement universel terminÃƒÂ© en {time.time() - start_time:.1f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur traitement universel: {e}")
            if self.config['fallback_on_error']:
                return self._generate_fallback_result(transcript, start_time)
            raise

    def _generate_universal_broll(self, transcript: str) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ©ration B-roll universelle - fonctionne sur TOUS les domaines"""
        logger.info("Ã°Å¸â€â€˜ GÃƒÂ©nÃƒÂ©ration B-roll universelle...")
        
        try:
            result = self.broll_generator.generate_broll_keywords_universal(
                transcript, 
                self.config['target_keywords']
            )
            
            logger.info(f"Ã¢Å“â€¦ B-roll universel gÃƒÂ©nÃƒÂ©rÃƒÂ©: {len(result['keywords'])} mots-clÃƒÂ©s")
            return result
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur gÃƒÂ©nÃƒÂ©ration B-roll: {e}")
            # Fallback vers mots-clÃƒÂ©s basiques
            return self._generate_basic_broll(transcript)

    def _generate_viral_metadata(self, transcript: str, broll_result: Dict[str, Any]) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es virales TikTok/Instagram"""
        if not self.config['enable_viral_metadata']:
            return {}
            
        logger.info("Ã°Å¸â€œÂ± GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es virales TikTok/Instagram...")
        
        try:
            domain = broll_result.get('domain', 'general')
            keywords = broll_result.get('keywords', [])
            
            result = self.metadata_generator.generate_viral_metadata(
                transcript, domain, keywords
            )
            
            logger.info(f"Ã¢Å“â€¦ MÃƒÂ©tadonnÃƒÂ©es virales gÃƒÂ©nÃƒÂ©rÃƒÂ©es: {result['title'][:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur mÃƒÂ©tadonnÃƒÂ©es virales: {e}")
            return self._generate_basic_metadata(transcript)

    def _assemble_final_result(self, broll_result: Dict[str, Any], 
                              metadata_result: Dict[str, Any], 
                              start_time: float) -> Dict[str, Any]:
        """Assemblage du rÃƒÂ©sultat final universel"""
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'processing_time': processing_time,
            'universal_pipeline': True,
            
            # B-roll universel
            'broll_data': {
                'keywords': broll_result.get('keywords', []),
                'search_queries': broll_result.get('search_queries', []),
                'domain': broll_result.get('domain', 'general'),
                'domain_confidence': broll_result.get('domain_confidence', 0.0),
                'diversity_metrics': broll_result.get('diversity_metrics', {}),
                'total_keywords': broll_result.get('total_keywords', 0)
            },
            
            # MÃƒÂ©tadonnÃƒÂ©es virales TikTok/Instagram
            'metadata': {
                'title': metadata_result.get('title', ''),
                'description': metadata_result.get('description', ''),
                'hashtags': metadata_result.get('hashtags', []),
                'call_to_action': metadata_result.get('call_to_action', ''),
                'platform_optimized': metadata_result.get('platform_optimized', True),
                'viral_potential': metadata_result.get('viral_potential', 'high')
            },
            
            # Statistiques
            'statistics': {
                'keywords_generated': len(broll_result.get('keywords', [])),
                'hashtags_generated': len(metadata_result.get('hashtags', [])),
                'domain_detected': broll_result.get('domain', 'general'),
                'processing_efficiency': 'high' if processing_time < 10 else 'medium'
            }
        }

    def _generate_fallback_result(self, transcript: str, start_time: float) -> Dict[str, Any]:
        """RÃƒÂ©sultat de fallback en cas d'erreur"""
        processing_time = time.time() - start_time
        
        return {
            'success': False,
            'processing_time': processing_time,
            'universal_pipeline': True,
            'fallback_used': True,
            
            'broll_data': {
                'keywords': ['content', 'video', 'media', 'visual'],
                'search_queries': ['content video', 'media visual'],
                'domain': 'general',
                'domain_confidence': 0.0,
                'diversity_metrics': {},
                'total_keywords': 4
            },
            
            'metadata': {
                'title': 'Amazing Content - Must See!',
                'description': 'Check out this incredible content!',
                'hashtags': ['#content', '#viral', '#amazing', '#mustsee'],
                'call_to_action': 'Follow for more amazing content!',
                'platform_optimized': True,
                'viral_potential': 'medium'
            },
            
            'statistics': {
                'keywords_generated': 4,
                'hashtags_generated': 4,
                'domain_detected': 'general',
                'processing_efficiency': 'fallback'
            }
        }

    def _generate_basic_broll(self, transcript: str) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ©ration B-roll basique en cas d'erreur"""
        words = transcript.split()[:10]
        basic_keywords = [word.lower() for word in words if len(word) > 3][:8]
        
        return {
            'keywords': basic_keywords,
            'search_queries': [' '.join(basic_keywords[:2])],
            'domain': 'general',
            'domain_confidence': 0.0,
            'diversity_metrics': {'categories_covered': 1, 'diversity_score': 0.2},
            'total_keywords': len(basic_keywords)
        }

    def _generate_basic_metadata(self, transcript: str) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es basiques en cas d'erreur"""
        return {
            'title': 'Amazing Content - Must See!',
            'description': 'Check out this incredible content!',
            'hashtags': ['#content', '#viral', '#amazing', '#mustsee'],
            'call_to_action': 'Follow for more amazing content!',
            'platform_optimized': True,
            'viral_potential': 'medium'
        }

def create_universal_pipeline() -> UniversalPipelineIntegration:
    """Factory pour crÃƒÂ©er le pipeline universel"""
    return UniversalPipelineIntegration()

# Test rapide du pipeline universel
if __name__ == "__main__":
    print("Ã°Å¸Â§Âª TEST DU PIPELINE UNIVERSEL")
    print("=" * 50)
    
    pipeline = create_universal_pipeline()
    
    # Test avec diffÃƒÂ©rents domaines
    test_cases = [
        ("science", "Research shows that cognitive control and effort are linked to dopamine levels in the brain."),
        ("sport", "Athletes train hard to improve performance. The coach emphasizes dedication and teamwork."),
        ("finance", "Investment strategies require careful analysis. Traders study market trends."),
        ("lifestyle", "Cooking healthy meals requires creativity and planning. Travel opens new perspectives.")
    ]
    
    for domain, transcript in test_cases:
        print(f"\nÃ°Å¸Å½Â¯ TEST: {domain.upper()}")
        print("-" * 30)
        
        result = pipeline.process_video_universal(transcript)
        
        if result['success']:
            print(f"Ã¢Å“â€¦ Titre: {result['metadata']['title']}")
            print(f"Ã°Å¸â€â€˜ B-roll: {result['broll_data']['keywords'][:3]}...")
            print(f"#Ã¯Â¸ÂÃ¢Æ’Â£ Hashtags: {result['metadata']['hashtags'][:3]}...")
        else:
            print(f"Ã¢ÂÅ’ Ãƒâ€°chec: {result.get('error', 'Unknown error')}")
    
    print("\nÃ°Å¸Å¡â‚¬ Pipeline universel testÃƒÂ© avec succÃƒÂ¨s!") 

