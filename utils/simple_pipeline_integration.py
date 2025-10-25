ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ INTÃƒâ€°GRATION PIPELINE SIMPLIFIÃƒâ€°E - UTILISE DIRECTEMENT OptimizedLLM
# Compatible avec toutes nos amÃƒÂ©liorations

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Import direct d'OptimizedLLM (nos amÃƒÂ©liorations)
from utils.optimized_llm import OptimizedLLM

logger = logging.getLogger(__name__)

class SimplePipelineIntegration:
    """IntÃƒÂ©gration simplifiÃƒÂ©e avec OptimizedLLM amÃƒÂ©liorÃƒÂ©"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        # Utiliser directement OptimizedLLM avec nos amÃƒÂ©liorations
        self.llm = OptimizedLLM(model="gemma3:4b")
        
        logger.info("Ã¢Å“â€¦ Pipeline simple avec OptimizedLLM amÃƒÂ©liorÃƒÂ© initialisÃƒÂ©")
    
    def _default_config(self) -> Dict[str, Any]:
        """Configuration par dÃƒÂ©faut"""
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
        """Traite un transcript vidÃƒÂ©o avec OptimizedLLM amÃƒÂ©liorÃƒÂ©"""
        
        start_time = time.time()
        result = {
            'success': False,
            'errors': [],
            'metadata': {},
            'broll_data': {},
            'processing_time': 0.0
        }
        
        try:
            # 1. GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es avec OptimizedLLM
            if self.config['enable_metadata_generation']:
                metadata_success, metadata = self.llm.generate_complete_metadata(transcript)
                if metadata_success:
                    result['metadata'] = metadata
                    result['success'] = True
                    logger.info(f"Ã¢Å“â€¦ MÃƒÂ©tadonnÃƒÂ©es gÃƒÂ©nÃƒÂ©rÃƒÂ©es pour {video_id}")
                else:
                    result['errors'].append("Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es")
                    logger.error(f"Ã¢ÂÅ’ MÃƒÂ©tadonnÃƒÂ©es ÃƒÂ©chouÃƒÂ©es pour {video_id}")
            
            # 2. GÃƒÂ©nÃƒÂ©ration B-roll avec nos amÃƒÂ©liorations hybrides
            if self.config['enable_broll_generation'] and result['success']:
                broll_data = self._optimize_broll_keywords(transcript, video_id)
                result['broll_data'] = broll_data
                logger.info(f"Ã°Å¸Å½Â¬ B-roll gÃƒÂ©nÃƒÂ©rÃƒÂ©: {len(broll_data.get('keywords', []))} mots-clÃƒÂ©s")
            
            # 3. Temps de traitement
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            logger.info(f"Ã¢Å“â€¦ VidÃƒÂ©o {video_id} traitÃƒÂ©e en {processing_time:.1f}s")
            return result
            
        except Exception as e:
            error_msg = f"Erreur traitement vidÃƒÂ©o {video_id}: {str(e)}"
            logger.error(error_msg)
            
            result['errors'].append(error_msg)
            result['processing_time'] = time.time() - start_time
            
            # Fallback si activÃƒÂ©
            if self.config['fallback_on_error']:
                result = self._fallback_processing(transcript, video_id, result)
            
            return result
    
    def _optimize_broll_keywords(self, transcript: str, video_id: str) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ©ration B-roll avec OptimizedLLM amÃƒÂ©liorÃƒÂ© (hybride actions+concepts)"""
        try:
            # Ã°Å¸Å¡â‚¬ UTILISER NOS AMÃƒâ€°LIORATIONS HYBRIDES DIRECTEMENT
            success, broll_data = self.llm.generate_broll_keywords_and_queries(
                transcript, 
                max_keywords=self.config['max_keywords_per_video']
            )
            
            if success and broll_data:
                logger.info(f"Ã¢Å“â€¦ B-roll LLM gÃƒÂ©nÃƒÂ©rÃƒÂ©: {len(broll_data.get('broll_keywords', []))} mots-clÃƒÂ©s")
                return {
                    'keywords': broll_data.get('broll_keywords', []),
                    'search_queries': broll_data.get('search_queries', []),
                    'domain': broll_data.get('domain', 'unknown'),
                    'context': broll_data.get('context', ''),
                    'hybrid_strategy': 'actions_and_concepts'  # Notre stratÃƒÂ©gie hybride
                }
            else:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â LLM B-roll ÃƒÂ©chouÃƒÂ©, fallback pour {video_id}")
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
            logger.error(f"Ã¢ÂÅ’ Erreur B-roll pour {video_id}: {e}")
            return {}
    
    def _extract_fallback_keywords(self, transcript: str) -> List[str]:
        """Extraction fallback intelligente de mots-clÃƒÂ©s"""
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
        
        # DÃƒÂ©tecter le domaine et retourner les mots-clÃƒÂ©s hybrides appropriÃƒÂ©s
        for domain, kws in domain_fallbacks.items():
            if domain in text_lower:
                keywords.extend(kws)
                break
        
        # Si aucun domaine dÃƒÂ©tectÃƒÂ©, extraire des mots-clÃƒÂ©s intelligents
        if not keywords:
            words = text_lower.split()
            significant_words = [w for w in words if len(w) > 4 and w.isalpha()][:6]
            # CrÃƒÂ©er des mots-clÃƒÂ©s hybrides fallback
            keywords = [f"professional_{word}" for word in significant_words[:3]]
            keywords.extend(significant_words[:3])  # + mots directs
        
        return keywords[:8]
    
    def _fallback_processing(self, transcript: str, video_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Traitement de fallback en cas d'erreur"""
        try:
            # MÃƒÂ©tadonnÃƒÂ©es basiques
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
            logger.info(f"Ã°Å¸â€ Ëœ Fallback appliquÃƒÂ© pour {video_id}")
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur fallback pour {video_id}: {e}")
        
        return result

# Factory function pour compatibilitÃƒÂ©
def create_pipeline_integration(config: Dict[str, Any] = None) -> SimplePipelineIntegration:
    """Factory pour crÃƒÂ©er l'intÃƒÂ©gration simplifiÃƒÂ©e"""
    return SimplePipelineIntegration(config) 

