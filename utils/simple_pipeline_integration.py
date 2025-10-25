# -*- coding: utf-8 -*-
# ðŸš€ INTÃ‰GRATION PIPELINE SIMPLIFIÃ‰E - UTILISE DIRECTEMENT OptimizedLLM
# Compatible avec toutes nos amÃ©liorations

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Import direct d'OptimizedLLM (nos amÃ©liorations)
from utils.optimized_llm import OptimizedLLM

logger = logging.getLogger(__name__)

class SimplePipelineIntegration:
    """IntÃ©gration simplifiÃ©e avec OptimizedLLM amÃ©liorÃ©"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        # Utiliser directement OptimizedLLM avec nos amÃ©liorations
        self.llm = OptimizedLLM(model="gemma3:4b")
        
        logger.info("âœ… Pipeline simple avec OptimizedLLM amÃ©liorÃ© initialisÃ©")
    
    def _default_config(self) -> Dict[str, Any]:
        """Configuration par dÃ©faut"""
        return {
            'max_keywords_per_video': 15,
            'enable_broll_generation': True,
            'enable_metadata_generation': True,
            'fallback_on_error': True,
            'max_retries': 3
        }
    
    def process_video_transcript(self, 
                                transcript: str, 
                                video_id: str,
                                segment_timestamps: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """Traite un transcript vidÃ©o avec OptimizedLLM amÃ©liorÃ©"""
        
        start_time = time.time()
        result = {
            'success': False,
            'errors': [],
            'metadata': {},
            'broll_data': {},
            'processing_time': 0.0
        }
        
        try:
            # 1. GÃ©nÃ©ration mÃ©tadonnÃ©es avec OptimizedLLM
            if self.config['enable_metadata_generation']:
                metadata_success, metadata = self.llm.generate_complete_metadata(transcript)
                if metadata_success:
                    result['metadata'] = metadata
                    result['success'] = True
                    logger.info(f"âœ… MÃ©tadonnÃ©es gÃ©nÃ©rÃ©es pour {video_id}")
                else:
                    result['errors'].append("Ã‰chec gÃ©nÃ©ration mÃ©tadonnÃ©es")
                    logger.error(f"âŒ MÃ©tadonnÃ©es Ã©chouÃ©es pour {video_id}")
            
            # 2. GÃ©nÃ©ration B-roll avec nos amÃ©liorations hybrides
            if self.config['enable_broll_generation'] and result['success']:
                broll_data = self._optimize_broll_keywords(transcript, video_id)
                result['broll_data'] = broll_data
                logger.info(f"ðŸŽ¬ B-roll gÃ©nÃ©rÃ©: {len(broll_data.get('keywords', []))} mots-clÃ©s")
            
            # 3. Temps de traitement
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            logger.info(f"âœ… VidÃ©o {video_id} traitÃ©e en {processing_time:.1f}s")
            return result
            
        except Exception as e:
            error_msg = f"Erreur traitement vidÃ©o {video_id}: {str(e)}"
            logger.error(error_msg)
            
            result['errors'].append(error_msg)
            result['processing_time'] = time.time() - start_time
            
            # Fallback si activÃ©
            if self.config['fallback_on_error']:
                result = self._fallback_processing(transcript, video_id, result)
            
            return result
    
    def _optimize_broll_keywords(self, transcript: str, video_id: str) -> Dict[str, Any]:
        """GÃ©nÃ©ration B-roll avec OptimizedLLM amÃ©liorÃ© (hybride actions+concepts)"""
        try:
            # ðŸš€ UTILISER NOS AMÃ‰LIORATIONS HYBRIDES DIRECTEMENT
            success, broll_data = self.llm.generate_broll_keywords_and_queries(
                transcript, 
                max_keywords=self.config['max_keywords_per_video']
            )
            
            if success and broll_data:
                logger.info(f"âœ… B-roll LLM gÃ©nÃ©rÃ©: {len(broll_data.get('broll_keywords', []))} mots-clÃ©s")
                return {
                    'keywords': broll_data.get('broll_keywords', []),
                    'search_queries': broll_data.get('search_queries', []),
                    'domain': broll_data.get('domain', 'unknown'),
                    'context': broll_data.get('context', ''),
                    'hybrid_strategy': 'actions_and_concepts'  # Notre stratÃ©gie hybride
                }
            else:
                logger.warning(f"âš ï¸ LLM B-roll Ã©chouÃ©, fallback pour {video_id}")
                # Fallback intelligent
                fallback_keywords = self._extract_fallback_keywords(transcript)
                return {
                    'keywords': fallback_keywords,
                    'search_queries': [f"'{kw}'" for kw in fallback_keywords[:5]],
                    'domain': 'general',
                    'context': 'fallback_extraction',
                    'hybrid_strategy': 'fallback'
                }
                
        except Exception as e:
            logger.error(f"âŒ Erreur B-roll pour {video_id}: {e}")
            return {}
    
    def _extract_fallback_keywords(self, transcript: str) -> List[str]:
        """Extraction fallback intelligente de mots-clÃ©s"""
        # Concepts hybrides par domaine
        domain_fallbacks = {
            'brain': ['brain', 'neural_networks', 'neurons', 'mind', 'brain_scan_fmri'],
            'therapy': ['person_talking_to_therapist', 'therapy_session', 'patient_consultation'],
            'business': ['business_meeting', 'entrepreneur_presenting', 'office_workspace'],
            'science': ['laboratory_research', 'scientist_working', 'data_analysis'],
            'technology': ['programmer_coding_computer', 'tech_workspace', 'software_development']
        }
        
        text_lower = transcript.lower()
        keywords = []
        
        # DÃ©tecter le domaine et retourner les mots-clÃ©s hybrides appropriÃ©s
        for domain, kws in domain_fallbacks.items():
            if domain in text_lower:
                keywords.extend(kws)
                break
        
        # Si aucun domaine dÃ©tectÃ©, extraire des mots-clÃ©s intelligents
        if not keywords:
            words = text_lower.split()
            significant_words = [w for w in words if len(w) > 4 and w.isalpha()][:6]
            # CrÃ©er des mots-clÃ©s hybrides fallback
            keywords = [f"professional_{word}" for word in significant_words[:3]]
            keywords.extend(significant_words[:3])  # + mots directs
        
        return keywords[:8]
    
    def _fallback_processing(self, transcript: str, video_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Traitement de fallback en cas d'erreur"""
        try:
            # MÃ©tadonnÃ©es basiques
            result['metadata'] = {
                'title': f"Video {video_id}",
                'description': transcript[:100] + "...",
                'hashtags': ['#video', '#content'],
                'keywords': ['video', 'content']
            }
            
            # B-roll fallback
            result['broll_data'] = {
                'keywords': ['general_content', 'video_background'],
                'search_queries': ['general video', 'background footage'],
                'domain': 'general',
                'hybrid_strategy': 'emergency_fallback'
            }
            
            result['success'] = True
            logger.info(f"ðŸ†˜ Fallback appliquÃ© pour {video_id}")
            
        except Exception as e:
            logger.error(f"âŒ Erreur fallback pour {video_id}: {e}")
        
        return result

# Factory function pour compatibilitÃ©
def create_pipeline_integration(config: Dict[str, Any] = None) -> SimplePipelineIntegration:
    """Factory pour crÃ©er l'intÃ©gration simplifiÃ©e"""
    return SimplePipelineIntegration(config) 
