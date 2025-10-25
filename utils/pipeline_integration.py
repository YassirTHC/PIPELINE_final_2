ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ INTÃƒâ€°GRATION PIPELINE VIDÃƒâ€°O - CONNECTEUR PRINCIPAL
# Connecte tous les modules LLM au pipeline vidÃƒÂ©o existant

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from dotenv import load_dotenv

# Import des modules locaux
from domain_detection_enhanced import detect_domain_enhanced, get_domain_info
from keyword_processing import optimize_for_broll
from optimized_llm import create_optimized_llm, generate_complete_with_broll
from metrics_and_qa import record_llm_metrics, get_system_metrics

load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoPipelineIntegration:
    """IntÃƒÂ©gration complÃƒÂ¨te avec le pipeline vidÃƒÂ©o existant"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.llm = create_optimized_llm()
        self.session_id = f"session_{int(time.time())}"
        
        # MÃƒÂ©triques de session
        self.session_metrics = {
            'videos_processed': 0,
            'total_processing_time': 0.0,
            'successful_generations': 0,
            'failed_generations': 0,
            'avg_keywords_per_video': 0.0,
            'domain_distribution': {}
        }
    

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout: Optional[int] = None,
    ) -> str:
        if not prompt:
            return ''
        if not self.llm:
            raise RuntimeError('LLM engine not initialised')
        if hasattr(self.llm, 'complete'):
            return self.llm.complete(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        success, text_response, err = self.llm._call_llm(  # type: ignore[attr-defined]
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if not success:
            raise RuntimeError(f"LLM completion failed: {err or 'unknown'}")
        return text_response

    def complete_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        response = self.complete(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        try:
            return json.loads(response)
        except Exception as exc:
            raise ValueError('Completion did not return JSON payload') from exc


    def _default_config(self) -> Dict[str, Any]:
        """Configuration par dÃƒÂ©faut"""
        return {
            'max_keywords_per_video': 15,
            'min_keywords_quality': 0.6,
            'enable_broll_generation': True,
            'enable_metadata_generation': True,
            'enable_domain_detection': True,
            'fallback_on_error': True,
            'max_retries': 3,
            'timeout_per_video': 300  # 5 minutes
        }
    
    def process_video_transcript(self, 
                                transcript: str, 
                                video_id: str,
                                segment_timestamps: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Traitement complet d'un transcript vidÃƒÂ©o
        
        Args:
            transcript: Le transcript complet ou par segments
            video_id: Identifiant unique de la vidÃƒÂ©o
            segment_timestamps: [(start_time, end_time), ...] si segment-level
        
        Returns:
            Dict avec toutes les mÃƒÂ©tadonnÃƒÂ©es gÃƒÂ©nÃƒÂ©rÃƒÂ©es
        """
        start_time = time.time()
        logger.info(f"Ã°Å¸Å½Â¬ Traitement vidÃƒÂ©o {video_id} - {len(transcript)} caractÃƒÂ¨res")
        
        try:
            result = {
                'video_id': video_id,
                'session_id': self.session_id,
                'processing_timestamp': time.time(),
                'success': False,
                'metadata': {},
                'broll_data': {},
                'domain_info': {},
                'processing_time': 0.0,
                'errors': []
            }
            
            # 1. DÃƒÂ©tection de domaine
            if self.config['enable_domain_detection']:
                domain, confidence = detect_domain_enhanced(transcript)
                domain_info = get_domain_info(domain)
                
                result['domain_info'] = {
                    'detected_domain': domain,
                    'confidence': confidence,
                    'domain_details': domain_info
                }
                
                logger.info(f"Ã°Å¸Å½Â¯ Domaine dÃƒÂ©tectÃƒÂ©: {domain} (confiance: {confidence:.3f})")
            
            # 2. GÃƒÂ©nÃƒÂ©ration LLM complÃƒÂ¨te
            if self.config['enable_metadata_generation']:
                llm_success, llm_data = self._generate_llm_content(transcript, video_id)
                
                if llm_success:
                    result['metadata'] = llm_data
                    result['success'] = True
                    self.session_metrics['successful_generations'] += 1
                    logger.info(f"Ã¢Å“â€¦ LLM rÃƒÂ©ussi pour {video_id}")
                else:
                    result['errors'].append("Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration LLM")
                    self.session_metrics['failed_generations'] += 1
                    logger.error(f"Ã¢ÂÅ’ LLM ÃƒÂ©chouÃƒÂ© pour {video_id}")
            
            # 3. Optimisation B-roll
            if self.config['enable_broll_generation'] and result['success']:
                broll_data = self._optimize_broll_keywords(transcript, video_id)
                result['broll_data'] = broll_data
                logger.info(f"Ã°Å¸Å½Â¬ B-roll optimisÃƒÂ©: {len(broll_data['keywords'])} mots-clÃƒÂ©s")
            
            # 4. Traitement par segments si timestamps fournis
            if segment_timestamps and len(segment_timestamps) > 1:
                segment_data = self._process_segments(transcript, segment_timestamps, video_id)
                result['segment_data'] = segment_data
                logger.info(f"Ã°Å¸â€œÅ  {len(segment_data)} segments traitÃƒÂ©s")
            
            # 5. MÃƒÂ©triques et validation
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            # Enregistrer les mÃƒÂ©triques
            self._record_video_metrics(video_id, transcript, result)
            
            # Mettre ÃƒÂ  jour les mÃƒÂ©triques de session
            self._update_session_metrics(result)
            
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
    
    def _generate_llm_content(self, transcript: str, video_id: str) -> Tuple[bool, Dict[str, Any]]:
        """GÃƒÂ©nÃƒÂ©ration du contenu LLM avec retry"""
        for attempt in range(self.config['max_retries']):
            try:
                success, data = generate_complete_with_broll(transcript)
                if success:
                    return True, data
                else:
                    logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Tentative {attempt + 1} ÃƒÂ©chouÃƒÂ©e pour {video_id}")
            except Exception as e:
                logger.error(f"Ã¢ÂÅ’ Erreur LLM tentative {attempt + 1}: {e}")
        
        return False, {}
    
    def _optimize_broll_keywords(self, transcript: str, video_id: str) -> Dict[str, Any]:
        """Optimisation des mots-clÃƒÂ©s B-roll avec OptimizedLLM amÃƒÂ©liorÃƒÂ©"""
        try:
            # Ã°Å¸Å¡â‚¬ NOUVEAU: Utiliser OptimizedLLM avec nos amÃƒÂ©liorations hybrides
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
                    'hybrid_strategy': 'actions_and_concepts'  # Notre nouvelle stratÃƒÂ©gie
                }
            else:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â LLM B-roll ÃƒÂ©chouÃƒÂ©, fallback vers mots-clÃƒÂ©s basiques")
                # Fallback amÃƒÂ©liorÃƒÂ©
                fallback_keywords = self._extract_fallback_keywords(transcript)
            return {
                    'keywords': fallback_keywords,
                    'search_queries': [f"'{kw}'" for kw in fallback_keywords[:5]],
                    'domain': 'general',
                    'context': 'fallback_extraction',
                    'hybrid_strategy': 'fallback'
            }
                
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur optimisation B-roll: {e}")
            return {}
    
    def _extract_fallback_keywords(self, transcript: str) -> List[str]:
        """Extraction fallback de mots-clÃƒÂ©s depuis le transcript"""
        # Concepts intelligents par domaine
        domain_fallbacks = {
            'brain': ['brain', 'neural_networks', 'neurons', 'mind'],
            'therapy': ['person_talking_to_therapist', 'therapy_session', 'patient_consultation'],
            'business': ['business_meeting', 'entrepreneur_presenting', 'office_workspace'],
            'science': ['laboratory_research', 'scientist_working', 'data_analysis']
        }
        
        text_lower = transcript.lower()
        keywords = []
        
        # DÃƒÂ©tecter le domaine et retourner les mots-clÃƒÂ©s appropriÃƒÂ©s
        for domain, kws in domain_fallbacks.items():
            if domain in text_lower:
                keywords.extend(kws)
                break
        
        # Si aucun domaine dÃƒÂ©tectÃƒÂ©, extraire des mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©riques intelligents
        if not keywords:
            words = text_lower.split()
            significant_words = [w for w in words if len(w) > 4 and w.isalpha()][:8]
            keywords = [f"professional_{word}" for word in significant_words[:4]]
        
        return keywords[:8]
    
    def _process_segments(self, transcript: str, timestamps: List[Tuple[float, float]], video_id: str) -> List[Dict[str, Any]]:
        """Traitement par segments avec timestamps"""
        segments = []
        
        for i, (start_time, end_time) in enumerate(timestamps):
            try:
                # Extraire le segment du transcript (logique ÃƒÂ  adapter)
                segment_text = f"Segment {i+1}: {transcript[:100]}..."  # Exemple
                
                # Traitement du segment
                segment_result = {
                    'segment_id': f"{video_id}_seg_{i+1}",
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': segment_text,
                    'keywords': [],
                    'domain': 'generic',
                    'confidence': 0.0
                }
                
                # DÃƒÂ©tection de domaine par segment
                if self.config['enable_domain_detection']:
                    domain, confidence = detect_domain_enhanced(segment_text)
                    segment_result['domain'] = domain
                    segment_result['confidence'] = confidence
                
                segments.append(segment_result)
                
            except Exception as e:
                logger.error(f"Ã¢ÂÅ’ Erreur segment {i+1}: {e}")
                continue
        
        return segments
    
    def _fallback_processing(self, transcript: str, video_id: str, failed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Traitement de fallback en cas d'ÃƒÂ©chec"""
        logger.info(f"Ã°Å¸â€â€ž Fallback pour {video_id}")
        
        try:
            # MÃƒÂ©thode de fallback simple
            fallback_result = failed_result.copy()
            
            # GÃƒÂ©nÃƒÂ©ration basique de mots-clÃƒÂ©s
            basic_keywords = self._extract_basic_keywords(transcript)
            fallback_result['metadata'] = {
                'title': f"Video {video_id}",
                'description': transcript[:200] + "..." if len(transcript) > 200 else transcript,
                'hashtags': ['#video', '#content'],
                'keywords': basic_keywords
            }
            
            fallback_result['success'] = True
            fallback_result['errors'].append("Fallback appliquÃƒÂ©")
            
            return fallback_result
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Fallback ÃƒÂ©chouÃƒÂ©: {e}")
            return failed_result
    
    def _extract_basic_keywords(self, transcript: str) -> List[str]:
        """Extraction basique de mots-clÃƒÂ©s (fallback)"""
        # Logique simple d'extraction
        words = transcript.lower().split()
        # Filtrer les mots courts et communs
        keywords = [word for word in words if len(word) > 3 and word not in ['the', 'and', 'for', 'with']]
        return list(set(keywords))[:10]  # Max 10 mots-clÃƒÂ©s
    
    def _record_video_metrics(self, video_id: str, transcript: str, result: Dict[str, Any]):
        """Enregistrement des mÃƒÂ©triques pour une vidÃƒÂ©o"""
        try:
            # Calculer les mÃƒÂ©triques
            keywords_count = len(result.get('metadata', {}).get('keywords', []))
            success = result.get('success', False)
            processing_time = result.get('processing_time', 0.0)
            domain = result.get('domain_info', {}).get('detected_domain', 'generic')
            confidence = result.get('domain_info', {}).get('confidence', 0.0)
            
            # Enregistrer dans le systÃƒÂ¨me de mÃƒÂ©triques
            record_llm_metrics(
                segment_id=video_id,
                transcript=transcript,
                success=success,
                response_time=processing_time,
                keywords=result.get('metadata', {}).get('keywords', []),
                domain=domain,
                confidence=confidence,
                fallback=not success,
                error_type="video_processing" if not success else None
            )
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur enregistrement mÃƒÂ©triques: {e}")
    
    def _update_session_metrics(self, result: Dict[str, Any]):
        """Mise ÃƒÂ  jour des mÃƒÂ©triques de session"""
        self.session_metrics['videos_processed'] += 1
        self.session_metrics['total_processing_time'] += result.get('processing_time', 0.0)
        
        # Mots-clÃƒÂ©s moyens
        keywords_count = len(result.get('metadata', {}).get('keywords', []))
        total_videos = self.session_metrics['videos_processed']
        current_avg = self.session_metrics['avg_keywords_per_video']
        self.session_metrics['avg_keywords_per_video'] = (current_avg * (total_videos - 1) + keywords_count) / total_videos
        
        # Distribution des domaines
        domain = result.get('domain_info', {}).get('detected_domain', 'generic')
        self.session_metrics['domain_distribution'][domain] = self.session_metrics['domain_distribution'].get(domain, 0) + 1
    
    def get_session_summary(self) -> Dict[str, Any]:
        """RÃƒÂ©sumÃƒÂ© de la session de traitement"""
        if self.session_metrics['videos_processed'] == 0:
            return self.session_metrics
        
        return {
            **self.session_metrics,
            'avg_processing_time': self.session_metrics['total_processing_time'] / self.session_metrics['videos_processed'],
            'success_rate': self.session_metrics['successful_generations'] / self.session_metrics['videos_processed']
        }
    
    def export_session_report(self, output_path: str = None) -> str:
        """Export du rapport de session"""
        if not output_path:
            timestamp = int(time.time())
            output_path = f"pipeline_session_{self.session_id}_{timestamp}.json"
        
        try:
            report_data = {
                'session_id': self.session_id,
                'timestamp': time.time(),
                'summary': self.get_session_summary(),
                'system_metrics': get_system_metrics().__dict__ if hasattr(get_system_metrics(), '__dict__') else {}
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Ã°Å¸â€œÅ  Rapport de session exportÃƒÂ©: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur export rapport: {e}")
            return ""

# === FONCTIONS UTILITAIRES ===
def create_pipeline_integration(config: Dict[str, Any] = None) -> VideoPipelineIntegration:
    """Factory pour crÃƒÂ©er une instance d'intÃƒÂ©gration"""
    return VideoPipelineIntegration(config)

def process_video_batch(transcripts: List[Tuple[str, str]], 
                       config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Traitement en lot de plusieurs vidÃƒÂ©os"""
    integration = create_pipeline_integration(config)
    results = []
    
    for transcript, video_id in transcripts:
        result = integration.process_video_transcript(transcript, video_id)
        results.append(result)
    
    return results

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("Ã°Å¸Â§Âª Test de l'intÃƒÂ©gration pipeline vidÃƒÂ©o...")
    
    # Test avec un transcript simple
    test_transcript = "EMDR therapy utilizes bilateral stimulation to process traumatic memories. The therapist guides the patient through eye movements while recalling distressing events."
    test_video_id = "test_video_001"
    
    # CrÃƒÂ©er l'intÃƒÂ©gration
    integration = create_pipeline_integration()
    
    # Traiter la vidÃƒÂ©o
    print(f"Ã°Å¸Å½Â¬ Traitement vidÃƒÂ©o: {test_video_id}")
    result = integration.process_video_transcript(test_transcript, test_video_id)
    
    # Afficher les rÃƒÂ©sultats
    print(f"Ã¢Å“â€¦ SuccÃƒÂ¨s: {result['success']}")
    print(f"Ã¢ÂÂ±Ã¯Â¸Â Temps: {result['processing_time']:.1f}s")
    
    if result['success']:
        print(f"Ã°Å¸Å½Â¯ Domaine: {result['domain_info']['detected_domain']}")
        print(f"Ã°Å¸â€œÂ Titre: {result['metadata'].get('title', 'N/A')}")
        print(f"Ã°Å¸â€â€˜ Mots-clÃƒÂ©s: {len(result['metadata'].get('keywords', []))}")
        print(f"Ã°Å¸Å½Â¬ B-roll: {len(result['broll_data'].get('keywords', []))}")
    
    # RÃƒÂ©sumÃƒÂ© de session
    summary = integration.get_session_summary()
    print(f"\nÃ°Å¸â€œÅ  RÃƒÂ©sumÃƒÂ© session:")
    print(f"   VidÃƒÂ©os traitÃƒÂ©es: {summary['videos_processed']}")
    print(f"   Taux de succÃƒÂ¨s: {summary['success_rate']:.1%}")
    print(f"   Temps moyen: {summary['avg_processing_time']:.1f}s")
    
    print("\nÃ¯Â¿Â½Ã¯Â¿Â½ Test terminÃƒÂ© !") 

