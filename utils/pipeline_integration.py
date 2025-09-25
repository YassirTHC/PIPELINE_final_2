# 🚀 INTÉGRATION PIPELINE VIDÉO - CONNECTEUR PRINCIPAL
# Connecte tous les modules LLM au pipeline vidéo existant

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
    """Intégration complète avec le pipeline vidéo existant"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.llm = create_optimized_llm()
        self.session_id = f"session_{int(time.time())}"
        
        # Métriques de session
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
        """Configuration par défaut"""
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
        Traitement complet d'un transcript vidéo
        
        Args:
            transcript: Le transcript complet ou par segments
            video_id: Identifiant unique de la vidéo
            segment_timestamps: [(start_time, end_time), ...] si segment-level
        
        Returns:
            Dict avec toutes les métadonnées générées
        """
        start_time = time.time()
        logger.info(f"🎬 Traitement vidéo {video_id} - {len(transcript)} caractères")
        
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
            
            # 1. Détection de domaine
            if self.config['enable_domain_detection']:
                domain, confidence = detect_domain_enhanced(transcript)
                domain_info = get_domain_info(domain)
                
                result['domain_info'] = {
                    'detected_domain': domain,
                    'confidence': confidence,
                    'domain_details': domain_info
                }
                
                logger.info(f"🎯 Domaine détecté: {domain} (confiance: {confidence:.3f})")
            
            # 2. Génération LLM complète
            if self.config['enable_metadata_generation']:
                llm_success, llm_data = self._generate_llm_content(transcript, video_id)
                
                if llm_success:
                    result['metadata'] = llm_data
                    result['success'] = True
                    self.session_metrics['successful_generations'] += 1
                    logger.info(f"✅ LLM réussi pour {video_id}")
                else:
                    result['errors'].append("Échec génération LLM")
                    self.session_metrics['failed_generations'] += 1
                    logger.error(f"❌ LLM échoué pour {video_id}")
            
            # 3. Optimisation B-roll
            if self.config['enable_broll_generation'] and result['success']:
                broll_data = self._optimize_broll_keywords(transcript, video_id)
                result['broll_data'] = broll_data
                logger.info(f"🎬 B-roll optimisé: {len(broll_data['keywords'])} mots-clés")
            
            # 4. Traitement par segments si timestamps fournis
            if segment_timestamps and len(segment_timestamps) > 1:
                segment_data = self._process_segments(transcript, segment_timestamps, video_id)
                result['segment_data'] = segment_data
                logger.info(f"📊 {len(segment_data)} segments traités")
            
            # 5. Métriques et validation
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            # Enregistrer les métriques
            self._record_video_metrics(video_id, transcript, result)
            
            # Mettre à jour les métriques de session
            self._update_session_metrics(result)
            
            logger.info(f"✅ Vidéo {video_id} traitée en {processing_time:.1f}s")
            return result
            
        except Exception as e:
            error_msg = f"Erreur traitement vidéo {video_id}: {str(e)}"
            logger.error(error_msg)
            
            result['errors'].append(error_msg)
            result['processing_time'] = time.time() - start_time
            
            # Fallback si activé
            if self.config['fallback_on_error']:
                result = self._fallback_processing(transcript, video_id, result)
            
            return result
    
    def _generate_llm_content(self, transcript: str, video_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Génération du contenu LLM avec retry"""
        for attempt in range(self.config['max_retries']):
            try:
                success, data = generate_complete_with_broll(transcript)
                if success:
                    return True, data
                else:
                    logger.warning(f"⚠️ Tentative {attempt + 1} échouée pour {video_id}")
            except Exception as e:
                logger.error(f"❌ Erreur LLM tentative {attempt + 1}: {e}")
        
        return False, {}
    
    def _optimize_broll_keywords(self, transcript: str, video_id: str) -> Dict[str, Any]:
        """Optimisation des mots-clés B-roll avec OptimizedLLM amélioré"""
        try:
            # 🚀 NOUVEAU: Utiliser OptimizedLLM avec nos améliorations hybrides
            success, broll_data = self.llm.generate_broll_keywords_and_queries(
                transcript, 
                max_keywords=self.config['max_keywords_per_video']
            )
            
            if success and broll_data:
                logger.info(f"✅ B-roll LLM généré: {len(broll_data.get('broll_keywords', []))} mots-clés")
                return {
                    'keywords': broll_data.get('broll_keywords', []),
                    'search_queries': broll_data.get('search_queries', []),
                    'domain': broll_data.get('domain', 'unknown'),
                    'context': broll_data.get('context', ''),
                    'hybrid_strategy': 'actions_and_concepts'  # Notre nouvelle stratégie
                }
            else:
                logger.warning(f"⚠️ LLM B-roll échoué, fallback vers mots-clés basiques")
                # Fallback amélioré
                fallback_keywords = self._extract_fallback_keywords(transcript)
            return {
                    'keywords': fallback_keywords,
                    'search_queries': [f"'{kw}'" for kw in fallback_keywords[:5]],
                    'domain': 'general',
                    'context': 'fallback_extraction',
                    'hybrid_strategy': 'fallback'
            }
                
        except Exception as e:
            logger.error(f"❌ Erreur optimisation B-roll: {e}")
            return {}
    
    def _extract_fallback_keywords(self, transcript: str) -> List[str]:
        """Extraction fallback de mots-clés depuis le transcript"""
        # Concepts intelligents par domaine
        domain_fallbacks = {
            'brain': ['brain', 'neural_networks', 'neurons', 'mind'],
            'therapy': ['person_talking_to_therapist', 'therapy_session', 'patient_consultation'],
            'business': ['business_meeting', 'entrepreneur_presenting', 'office_workspace'],
            'science': ['laboratory_research', 'scientist_working', 'data_analysis']
        }
        
        text_lower = transcript.lower()
        keywords = []
        
        # Détecter le domaine et retourner les mots-clés appropriés
        for domain, kws in domain_fallbacks.items():
            if domain in text_lower:
                keywords.extend(kws)
                break
        
        # Si aucun domaine détecté, extraire des mots-clés génériques intelligents
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
                # Extraire le segment du transcript (logique à adapter)
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
                
                # Détection de domaine par segment
                if self.config['enable_domain_detection']:
                    domain, confidence = detect_domain_enhanced(segment_text)
                    segment_result['domain'] = domain
                    segment_result['confidence'] = confidence
                
                segments.append(segment_result)
                
            except Exception as e:
                logger.error(f"❌ Erreur segment {i+1}: {e}")
                continue
        
        return segments
    
    def _fallback_processing(self, transcript: str, video_id: str, failed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Traitement de fallback en cas d'échec"""
        logger.info(f"🔄 Fallback pour {video_id}")
        
        try:
            # Méthode de fallback simple
            fallback_result = failed_result.copy()
            
            # Génération basique de mots-clés
            basic_keywords = self._extract_basic_keywords(transcript)
            fallback_result['metadata'] = {
                'title': f"Video {video_id}",
                'description': transcript[:200] + "..." if len(transcript) > 200 else transcript,
                'hashtags': ['#video', '#content'],
                'keywords': basic_keywords
            }
            
            fallback_result['success'] = True
            fallback_result['errors'].append("Fallback appliqué")
            
            return fallback_result
            
        except Exception as e:
            logger.error(f"❌ Fallback échoué: {e}")
            return failed_result
    
    def _extract_basic_keywords(self, transcript: str) -> List[str]:
        """Extraction basique de mots-clés (fallback)"""
        # Logique simple d'extraction
        words = transcript.lower().split()
        # Filtrer les mots courts et communs
        keywords = [word for word in words if len(word) > 3 and word not in ['the', 'and', 'for', 'with']]
        return list(set(keywords))[:10]  # Max 10 mots-clés
    
    def _record_video_metrics(self, video_id: str, transcript: str, result: Dict[str, Any]):
        """Enregistrement des métriques pour une vidéo"""
        try:
            # Calculer les métriques
            keywords_count = len(result.get('metadata', {}).get('keywords', []))
            success = result.get('success', False)
            processing_time = result.get('processing_time', 0.0)
            domain = result.get('domain_info', {}).get('detected_domain', 'generic')
            confidence = result.get('domain_info', {}).get('confidence', 0.0)
            
            # Enregistrer dans le système de métriques
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
            logger.error(f"❌ Erreur enregistrement métriques: {e}")
    
    def _update_session_metrics(self, result: Dict[str, Any]):
        """Mise à jour des métriques de session"""
        self.session_metrics['videos_processed'] += 1
        self.session_metrics['total_processing_time'] += result.get('processing_time', 0.0)
        
        # Mots-clés moyens
        keywords_count = len(result.get('metadata', {}).get('keywords', []))
        total_videos = self.session_metrics['videos_processed']
        current_avg = self.session_metrics['avg_keywords_per_video']
        self.session_metrics['avg_keywords_per_video'] = (current_avg * (total_videos - 1) + keywords_count) / total_videos
        
        # Distribution des domaines
        domain = result.get('domain_info', {}).get('detected_domain', 'generic')
        self.session_metrics['domain_distribution'][domain] = self.session_metrics['domain_distribution'].get(domain, 0) + 1
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Résumé de la session de traitement"""
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
            
            logger.info(f"📊 Rapport de session exporté: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur export rapport: {e}")
            return ""

# === FONCTIONS UTILITAIRES ===
def create_pipeline_integration(config: Dict[str, Any] = None) -> VideoPipelineIntegration:
    """Factory pour créer une instance d'intégration"""
    return VideoPipelineIntegration(config)

def process_video_batch(transcripts: List[Tuple[str, str]], 
                       config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Traitement en lot de plusieurs vidéos"""
    integration = create_pipeline_integration(config)
    results = []
    
    for transcript, video_id in transcripts:
        result = integration.process_video_transcript(transcript, video_id)
        results.append(result)
    
    return results

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("🧪 Test de l'intégration pipeline vidéo...")
    
    # Test avec un transcript simple
    test_transcript = "EMDR therapy utilizes bilateral stimulation to process traumatic memories. The therapist guides the patient through eye movements while recalling distressing events."
    test_video_id = "test_video_001"
    
    # Créer l'intégration
    integration = create_pipeline_integration()
    
    # Traiter la vidéo
    print(f"🎬 Traitement vidéo: {test_video_id}")
    result = integration.process_video_transcript(test_transcript, test_video_id)
    
    # Afficher les résultats
    print(f"✅ Succès: {result['success']}")
    print(f"⏱️ Temps: {result['processing_time']:.1f}s")
    
    if result['success']:
        print(f"🎯 Domaine: {result['domain_info']['detected_domain']}")
        print(f"📝 Titre: {result['metadata'].get('title', 'N/A')}")
        print(f"🔑 Mots-clés: {len(result['metadata'].get('keywords', []))}")
        print(f"🎬 B-roll: {len(result['broll_data'].get('keywords', []))}")
    
    # Résumé de session
    summary = integration.get_session_summary()
    print(f"\n📊 Résumé session:")
    print(f"   Vidéos traitées: {summary['videos_processed']}")
    print(f"   Taux de succès: {summary['success_rate']:.1%}")
    print(f"   Temps moyen: {summary['avg_processing_time']:.1f}s")
    
    print("\n�� Test terminé !") 