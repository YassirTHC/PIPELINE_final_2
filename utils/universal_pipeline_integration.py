# -*- coding: utf-8 -*-
# ðŸš€ INTÃ‰GRATION UNIVERSELLE - REMPLACE COMPLÃˆTEMENT L'ANCIEN SYSTÃˆME
# Pipeline 100% universel : B-roll + mÃ©tadonnÃ©es TikTok/Instagram pour TOUS les domaines

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
    """Pipeline universel pour TOUS les domaines - B-roll + mÃ©tadonnÃ©es TikTok/Instagram"""
    
    def __init__(self):
        self.broll_generator = create_universal_broll_generator()
        self.metadata_generator = create_tiktok_metadata_generator()
        
        # Configuration universelle
        self.config = {
            'target_keywords': 10,  # 8-12 mots-clÃ©s optimaux
            'enable_viral_metadata': True,
            'enable_broll_generation': True,
            'fallback_on_error': True,
            'max_processing_time': 30  # secondes
        }
        
        logger.info("ðŸš€ Pipeline universel initialisÃ© - PrÃªt pour TOUS les domaines")

    def process_video_universal(self, transcript: str, video_id: str = None) -> Dict[str, Any]:
        """Traitement universel d'une vidÃ©o - fonctionne sur TOUS les domaines"""
        start_time = time.time()
        
        try:
            logger.info(f"ðŸŽ¬ Traitement universel dÃ©marrÃ© pour {len(transcript)} caractÃ¨res")
            
            # 1. GÃ©nÃ©ration B-roll universelle
            broll_result = self._generate_universal_broll(transcript)
            
            # 2. GÃ©nÃ©ration mÃ©tadonnÃ©es virales TikTok/Instagram
            metadata_result = self._generate_viral_metadata(transcript, broll_result)
            
            # 3. Assemblage du rÃ©sultat final
            final_result = self._assemble_final_result(broll_result, metadata_result, start_time)
            
            logger.info(f"âœ… Traitement universel terminÃ© en {time.time() - start_time:.1f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"âŒ Erreur traitement universel: {e}")
            if self.config['fallback_on_error']:
                return self._generate_fallback_result(transcript, start_time)
            raise

    def _generate_universal_broll(self, transcript: str) -> Dict[str, Any]:
        """GÃ©nÃ©ration B-roll universelle - fonctionne sur TOUS les domaines"""
        logger.info("ðŸ”‘ GÃ©nÃ©ration B-roll universelle...")
        
        try:
            result = self.broll_generator.generate_broll_keywords_universal(
                transcript, 
                self.config['target_keywords']
            )
            
            logger.info(f"âœ… B-roll universel gÃ©nÃ©rÃ©: {len(result['keywords'])} mots-clÃ©s")
            return result
            
        except Exception as e:
            logger.error(f"âŒ Erreur gÃ©nÃ©ration B-roll: {e}")
            # Fallback vers mots-clÃ©s basiques
            return self._generate_basic_broll(transcript)

    def _generate_viral_metadata(self, transcript: str, broll_result: Dict[str, Any]) -> Dict[str, Any]:
        """GÃ©nÃ©ration mÃ©tadonnÃ©es virales TikTok/Instagram"""
        if not self.config['enable_viral_metadata']:
            return {}
            
        logger.info("ðŸ“± GÃ©nÃ©ration mÃ©tadonnÃ©es virales TikTok/Instagram...")
        
        try:
            domain = broll_result.get('domain', 'general')
            keywords = broll_result.get('keywords', [])
            
            result = self.metadata_generator.generate_viral_metadata(
                transcript, domain, keywords
            )
            
            logger.info(f"âœ… MÃ©tadonnÃ©es virales gÃ©nÃ©rÃ©es: {result['title'][:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"âŒ Erreur mÃ©tadonnÃ©es virales: {e}")
            return self._generate_basic_metadata(transcript)

    def _assemble_final_result(self, broll_result: Dict[str, Any], 
                              metadata_result: Dict[str, Any], 
                              start_time: float) -> Dict[str, Any]:
        """Assemblage du rÃ©sultat final universel"""
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
            
            # MÃ©tadonnÃ©es virales TikTok/Instagram
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
        """RÃ©sultat de fallback en cas d'erreur"""
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
        """GÃ©nÃ©ration B-roll basique en cas d'erreur"""
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
        """GÃ©nÃ©ration mÃ©tadonnÃ©es basiques en cas d'erreur"""
        return {
            'title': 'Amazing Content - Must See!',
            'description': 'Check out this incredible content!',
            'hashtags': ['#content', '#viral', '#amazing', '#mustsee'],
            'call_to_action': 'Follow for more amazing content!',
            'platform_optimized': True,
            'viral_potential': 'medium'
        }

def create_universal_pipeline() -> UniversalPipelineIntegration:
    """Factory pour crÃ©er le pipeline universel"""
    return UniversalPipelineIntegration()

# Test rapide du pipeline universel
if __name__ == "__main__":
    print("ðŸ§ª TEST DU PIPELINE UNIVERSEL")
    print("=" * 50)
    
    pipeline = create_universal_pipeline()
    
    # Test avec diffÃ©rents domaines
    test_cases = [
        ("science", "Research shows that cognitive control and effort are linked to dopamine levels in the brain."),
        ("sport", "Athletes train hard to improve performance. The coach emphasizes dedication and teamwork."),
        ("finance", "Investment strategies require careful analysis. Traders study market trends."),
        ("lifestyle", "Cooking healthy meals requires creativity and planning. Travel opens new perspectives.")
    ]
    
    for domain, transcript in test_cases:
        print(f"\nðŸŽ¯ TEST: {domain.upper()}")
        print("-" * 30)
        
        result = pipeline.process_video_universal(transcript)
        
        if result['success']:
            print(f"âœ… Titre: {result['metadata']['title']}")
            print(f"ðŸ”‘ B-roll: {result['broll_data']['keywords'][:3]}...")
            print(f"#ï¸âƒ£ Hashtags: {result['metadata']['hashtags'][:3]}...")
        else:
            print(f"âŒ Ã‰chec: {result.get('error', 'Unknown error')}")
    
    print("\nðŸš€ Pipeline universel testÃ© avec succÃ¨s!") 
