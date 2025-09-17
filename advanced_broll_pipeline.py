"""
Pipeline B-roll Avancé Complet avec Intégration Intelligente
Version de production avec NLP, ML, gestion vidéo et base de données
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor
import time

# Configuration du logging avancé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedBrollPipeline:
    """Pipeline B-roll avancé avec intégration complète des composants intelligents"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.context_analyzer = None
        self.broll_selector = None
        self.processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
            "last_request_time": None
        }
        
        # Initialisation synchrone des composants de base
        self._initialize_sync_components()
        
        logger.info("Pipeline B-roll avancé initialisé (initialisation asynchrone en attente)")

    def _initialize_sync_components(self):
        """Initialise les composants de base de manière synchrone"""
        try:
            logger.info("Initialisation des composants de base...")
            
            # Vérifier la disponibilité des modules
            try:
                import advanced_context_analyzer
                logger.info("✅ Module advanced_context_analyzer disponible")
            except ImportError as e:
                logger.warning(f"⚠️ Module advanced_context_analyzer non disponible: {e}")
            
            try:
                import advanced_broll_selector
                logger.info("✅ Module advanced_broll_selector disponible")
            except ImportError as e:
                logger.warning(f"⚠️ Module advanced_broll_selector non disponible: {e}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des composants de base: {e}")

    async def initialize_async_components(self):
        """Initialise les composants asynchrones (à appeler après création de l'instance)"""
        try:
            logger.info("🔄 Initialisation asynchrone des composants intelligents...")
            
            # Initialiser l'analyseur contextuel
            try:
                from advanced_context_analyzer import AdvancedContextAnalyzer
                self.context_analyzer = AdvancedContextAnalyzer()
                logger.info("✅ Analyseur contextuel avancé initialisé")
            except ImportError as e:
                logger.warning(f"⚠️ Analyseur contextuel avancé non disponible: {e}")
                self.context_analyzer = None
            
            # Initialiser le sélecteur B-roll
            try:
                from advanced_broll_selector import AdvancedBrollSelector
                self.broll_selector = AdvancedBrollSelector()
                logger.info("✅ Sélecteur B-roll avancé initialisé")
            except ImportError as e:
                logger.warning(f"⚠️ Sélecteur B-roll avancé non disponible: {e}")
                self.broll_selector = None
            
            # Vérifier l'état des composants
            if self.context_analyzer and self.broll_selector:
                logger.info("🎉 Tous les composants intelligents sont opérationnels!")
            else:
                logger.warning("⚠️ Certains composants intelligents ne sont pas disponibles")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des composants: {e}")
            traceback.print_exc()
            return False

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration du pipeline"""
        default_config = {
            "pipeline": {
                "name": "Advanced B-roll Pipeline",
                "version": "2.0.0-production",
                "max_concurrent_requests": 5,
                "request_timeout": 300,  # secondes
                "enable_caching": True,
                "cache_ttl": 3600
            },
            "analysis": {
                "min_confidence_threshold": 0.6,
                "max_segments_per_request": 100,
                "enable_semantic_analysis": True,
                "enable_visual_analysis": True,
                "enable_sentiment_analysis": True
            },
            "broll_selection": {
                "max_candidates_per_segment": 10,
                "diversity_weight": 0.2,
                "quality_weight": 0.3,
                "context_weight": 0.3,
                "semantic_weight": 0.2
            },
            "performance": {
                "enable_profiling": True,
                "log_performance_metrics": True,
                "max_memory_usage": "2GB"
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self._merge_configs(default_config, user_config)
                    logger.info(f"Configuration chargée depuis: {config_path}")
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de la config: {e}")
        
        return default_config

    def _merge_configs(self, default: Dict, user: Dict):
        """Fusionne les configurations par défaut et utilisateur"""
        for key, value in user.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_configs(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value

    async def process_transcript_advanced(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Traite une transcription avec le pipeline avancé complet"""
        start_time = time.time()
        request_id = f"req_{int(start_time)}"
        
        try:
            logger.info(f"🚀 Démarrage du pipeline avancé pour la requête {request_id}")
            
            # Validation des données d'entrée
            if not self._validate_transcript_data(transcript_data):
                raise ValueError("Données de transcription invalides")
            
            # Attendre que les composants soient initialisés
            components_ready = await self._wait_for_components(timeout=30)
            if not components_ready:
                logger.warning("⚠️ Composants non disponibles, utilisation du mode fallback")
            
            # Analyse contextuelle avancée
            context_analysis = await self._perform_advanced_context_analysis(transcript_data)
            
            # Sélection B-roll intelligente
            broll_selections = await self._perform_intelligent_broll_selection(
                transcript_data, context_analysis
            )
            
            # Analyse des résultats
            results_analysis = await self._analyze_broll_selection_results(
                broll_selections, context_analysis
            )
            
            # Enrichissement des métadonnées
            enriched_metadata = await self._enrich_metadata(
                transcript_data, context_analysis, broll_selections
            )
            
            # Sauvegarder les métadonnées
            await self._save_metadata(request_id, transcript_data, context_analysis, broll_selections, results_analysis)
            
            processing_time = time.time() - start_time
            
            # Créer la réponse finale
            final_results = {
                "pipeline_status": "success",
                "request_id": request_id,
                "processing_time": processing_time,
                "context_analysis": context_analysis,
                "broll_selections": broll_selections,
                "results_analysis": results_analysis,
                "enriched_metadata": enriched_metadata,
                "performance_metrics": {
                    "total_processing_time": processing_time,
                    "segments_processed": len(transcript_data.get('segments', [])),
                    "brolls_selected": len(broll_selections),
                    "context_confidence": context_analysis.get('context_scores', {}).get('overall_confidence', 0.0),
                    "selection_quality": results_analysis.get('overall_quality_score', 0.0),
                    "components_status": {
                        "context_analyzer": "operational" if self.context_analyzer else "fallback",
                        "broll_selector": "operational" if self.broll_selector else "fallback"
                    }
                }
            }
            
            self.processing_stats["successful_requests"] += 1
            logger.info(f"SUCCESS: Pipeline avance termine avec succes en {processing_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.processing_stats["failed_requests"] += 1
            
            error_response = self._create_error_response(
                request_id, str(e), processing_time
            )
            
            logger.error(f"❌ Erreur pipeline avancé: {e}")
            traceback.print_exc()
            
            return error_response

    def _validate_transcript_data(self, transcript_data: Dict[str, Any]) -> bool:
        """Valide les données de transcription d'entrée"""
        try:
            required_fields = ['segments', 'metadata']
            
            # Vérifier les champs requis
            for field in required_fields:
                if field not in transcript_data:
                    logger.error(f"Champ requis manquant: {field}")
                    return False
            
            # Vérifier les segments
            segments = transcript_data.get('segments', [])
            if not segments or not isinstance(segments, list):
                logger.error("Segments invalides ou vides")
                return False
            
            # Vérifier la structure des segments
            for i, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    logger.error(f"Segment {i} invalide")
                    return False
                
                required_segment_fields = ['text', 'start', 'end']
                for field in required_segment_fields:
                    if field not in segment:
                        logger.error(f"Champ segment manquant: {field} dans segment {i}")
                        return False
            
            # Vérifier le nombre de segments
            max_segments = self.config["analysis"]["max_segments_per_request"]
            if len(segments) > max_segments:
                logger.warning(f"Nombre de segments ({len(segments)}) dépasse la limite ({max_segments})")
            
            logger.info(f"✅ Validation des données réussie: {len(segments)} segments")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur validation: {e}")
            return False

    async def _wait_for_components(self, timeout: int = 30):
        """Attend que les composants soient initialisés avec gestion d'erreur robuste"""
        start_time = time.time()
        
        # Si les composants ne sont pas initialisés, les initialiser
        if not self.context_analyzer or not self.broll_selector:
            logger.info("🔄 Initialisation des composants asynchrones...")
            try:
                await self.initialize_async_components()
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'initialisation des composants: {e}")
        
        # Attendre que les composants soient prêts
        while time.time() - start_time < timeout:
            if self.context_analyzer and self.broll_selector:
                logger.info("🎉 Tous les composants intelligents sont opérationnels!")
                return True
            await asyncio.sleep(1)
        
        # Timeout atteint, vérifier l'état des composants
        if not self.context_analyzer and not self.broll_selector:
            logger.error("❌ Aucun composant intelligent disponible après timeout")
            return False
        elif not self.context_analyzer:
            logger.warning("⚠️ Analyseur contextuel non disponible, utilisation du fallback")
            return False
        elif not self.broll_selector:
            logger.warning("⚠️ Sélecteur B-roll non disponible, utilisation du fallback")
            return False
        
        return False

    async def _perform_advanced_context_analysis(self, transcript_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Effectue l'analyse contextuelle avancée avec fallback robuste"""
        try:
            if not self.context_analyzer:
                logger.warning("⚠️ Analyseur contextuel avancé non disponible, utilisation du fallback")
                return self._create_enhanced_fallback_analysis(transcript_data)
            
            # Extraire les segments pour l'analyse
            segments = transcript_data.get('segments', [])
            
            # Effectuer l'analyse contextuelle avancée
            context_analysis = await self.context_analyzer.analyze_transcript_advanced(segments)
            
            if not context_analysis:
                logger.warning("⚠️ Analyse contextuelle avancée échouée, utilisation du fallback")
                return self._create_enhanced_fallback_analysis(transcript_data)
            
            # Vérifier la confiance de l'analyse
            confidence = context_analysis.get('context_scores', {}).get('overall_confidence', 0.0)
            min_confidence = self.config["analysis"]["min_confidence_threshold"]
            
            if confidence < min_confidence:
                logger.warning(f"WARNING: Confiance contextuelle faible ({confidence:.2f}), utilisation du fallback")
                return self._create_enhanced_fallback_analysis(transcript_data)
            
            logger.info(f"SUCCESS: Analyse contextuelle avancee reussie (confiance: {confidence:.2f})")
            return context_analysis
            
        except Exception as e:
            logger.error(f"ERROR: Erreur analyse contextuelle avancee: {e}")
            return self._create_enhanced_fallback_analysis(transcript_data)

    def _create_enhanced_fallback_analysis(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crée une analyse contextuelle de fallback améliorée"""
        try:
            segments = transcript_data.get('segments', [])
            
            # Analyse intelligente basée sur le contenu
            main_themes = self._extract_main_themes_from_segments(segments)
            overall_tone = self._analyze_overall_tone(segments)
            complexity_profile = self._analyze_complexity_profile(segments)
            
            enhanced_analysis = {
                "global_analysis": {
                    "main_theme": main_themes[0] if main_themes else "general",
                    "sub_themes": main_themes[1:3] if len(main_themes) > 1 else [],
                    "overall_tone": overall_tone,
                    "complexity_profile": complexity_profile,
                    "target_audience": self._determine_target_audience(complexity_profile),
                    "narrative_structure": "linear",
                    "context_coherence": 0.7  # Score amélioré
                },
                "segments_analysis": [
                    {
                        "text": seg.get('text', ''),
                        "local_context": self._determine_local_context(seg.get('text', '')),
                        "global_relevance": 0.7,
                        "main_keywords": self._extract_keywords_from_text(seg.get('text', '')),
                        "sentiment": self._analyze_sentiment_simple(seg.get('text', '')),
                        "complexity_level": self._analyze_complexity_simple(seg.get('text', '')),
                        "appropriate_brolls": self._suggest_appropriate_brolls(seg.get('text', '')),
                        "forbidden_brolls": [],
                        "context_score": 0.7
                    }
                    for seg in segments
                ],
                "narrative_coherence": {
                    "coherence_score": 0.7,
                    "transitions": self._analyze_narrative_transitions(segments),
                    "consistency": "good"
                },
                "context_scores": {
                    "overall_confidence": 0.7,
                    "theme_consistency": 0.7,
                    "complexity_match": 0.7,
                    "average_segment_score": 0.7
                }
            }
            
            logger.info("SUCCESS: Analyse de fallback amelioree creee avec succes")
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur création analyse de fallback: {e}")
            return self._create_basic_context_analysis(transcript_data)

    def _create_basic_context_analysis(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crée une analyse contextuelle basique en cas d'échec"""
        try:
            segments = transcript_data.get('segments', [])
            
            basic_analysis = {
                "global_analysis": {
                    "main_theme": "general",
                    "sub_themes": ["basic"],
                    "overall_tone": "neutral",
                    "complexity_profile": "medium",
                    "target_audience": "general_public",
                    "narrative_structure": "linear",
                    "context_coherence": 0.5
                },
                "segments_analysis": [
                    {
                        "text": seg.get('text', ''),
                        "local_context": "general",
                        "global_relevance": 0.5,
                        "main_keywords": [],
                        "sentiment": "neutral",
                        "complexity_level": "medium",
                        "appropriate_brolls": [],
                        "forbidden_brolls": [],
                        "context_score": 0.5
                    }
                    for seg in segments
                ],
                "narrative_coherence": {
                    "coherence_score": 0.5,
                    "transitions": [],
                    "consistency": "unknown"
                },
                "context_scores": {
                    "overall_confidence": 0.5,
                    "theme_consistency": 0.5,
                    "complexity_match": 0.5,
                    "average_segment_score": 0.5
                }
            }
            
            return basic_analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur création analyse basique: {e}")
            return {}

    async def _perform_intelligent_broll_selection(self, 
                                                 transcript_data: Dict[str, Any],
                                                 context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Effectue la sélection B-roll intelligente"""
        try:
            if not self.broll_selector:
                logger.warning("⚠️ Sélecteur B-roll avancé non disponible, utilisation du fallback")
                return self._create_basic_broll_selections(transcript_data)
            
            segments = transcript_data.get('segments', [])
            segments_analysis = context_analysis.get('segments_analysis', [])
            global_analysis = context_analysis.get('global_analysis', {})
            
            broll_selections = []
            
            for i, (segment, segment_analysis) in enumerate(zip(segments, segments_analysis)):
                try:
                    logger.debug(f"🎯 Traitement segment {i+1}/{len(segments)}: {segment.get('text', '')[:50]}...")
                    
                    # Sélection B-roll intelligente
                    selection = await self.broll_selector.select_contextual_brolls(
                        context_analysis, segment_analysis
                    )
                    
                    if selection:
                        broll_selections.append({
                            "segment_index": i,
                            "segment_text": segment.get('text', ''),
                            "segment_context": segment_analysis.get('local_context', 'unknown'),
                            "selected_broll": {
                                "id": selection.primary_broll.metadata.id,
                                "title": selection.primary_broll.metadata.title,
                                "file_path": str(selection.primary_broll.metadata.file_path),
                                "duration": selection.primary_broll.metadata.duration,
                                "context_relevance": selection.primary_broll.context_relevance,
                                "final_score": selection.primary_broll.final_score,
                                "selection_reason": selection.primary_broll.selection_reason
                            },
                            "alternative_brolls": [
                                {
                                    "id": alt.metadata.id,
                                    "title": alt.metadata.title,
                                    "file_path": str(alt.metadata.file_path),
                                    "duration": alt.metadata.duration,
                                    "final_score": alt.final_score
                                }
                                for alt in selection.alternative_brolls
                            ],
                            "selection_metadata": selection.selection_metadata,
                            "context_match_score": selection.context_match_score,
                            "diversity_score": selection.diversity_score
                        })
                        
                        logger.debug(f"SUCCESS: B-roll selectionne pour segment {i+1}: {selection.primary_broll.metadata.title}")
                    else:
                        logger.warning(f"WARNING: Aucune selection B-roll pour le segment {i+1}")
                        broll_selections.append({
                            "segment_index": i,
                            "segment_text": segment.get('text', ''),
                            "segment_context": segment_analysis.get('local_context', 'unknown'),
                            "selected_broll": None,
                            "alternative_brolls": [],
                            "selection_metadata": {"reason": "Aucune sélection possible"},
                            "context_match_score": 0.0,
                            "diversity_score": 0.0
                        })
                    
                except Exception as e:
                    logger.error(f"❌ Erreur traitement segment {i+1}: {e}")
                    
                    broll_selections.append({
                        "segment_index": i,
                        "segment_text": segment.get('text', ''),
                        "segment_context": "error",
                        "selected_broll": None,
                        "alternative_brolls": [],
                        "selection_metadata": {"reason": f"Erreur: {str(e)}"},
                        "context_match_score": 0.0,
                        "diversity_score": 0.0
                    })
            
            logger.info(f"SUCCESS: Selection B-roll intelligente terminee: {len(broll_selections)} segments traites")
            return broll_selections
            
        except Exception as e:
            logger.error(f"❌ Erreur sélection B-roll intelligente: {e}")
            return self._create_basic_broll_selections(transcript_data)

    def _create_basic_broll_selections(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Crée des sélections B-roll basiques en cas d'échec"""
        try:
            segments = transcript_data.get('segments', [])
            
            basic_selections = []
            for i, segment in enumerate(segments):
                basic_selections.append({
                    "segment_index": i,
                    "segment_text": segment.get('text', ''),
                    "segment_context": "general",
                    "selected_broll": {
                        "id": f"fallback_{i}",
                        "title": "Sélection par défaut",
                        "file_path": "",
                        "duration": 0.0,
                        "context_relevance": 0.5,
                        "final_score": 0.5,
                        "selection_reason": "Sélection basique - mode fallback"
                    },
                    "alternative_brolls": [],
                    "selection_metadata": {"reason": "Mode fallback"},
                    "context_match_score": 0.5,
                    "diversity_score": 0.5
                })
            
            return basic_selections
            
        except Exception as e:
            logger.error(f"❌ Erreur création sélections basiques: {e}")
            return []

    async def _analyze_broll_selection_results(self, broll_selections: List[Dict[str, Any]], 
                                            context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse les résultats de la sélection B-roll"""
        try:
            if not broll_selections:
                return {
                    "overall_quality_score": 0.0,
                    "context_relevance_score": 0.0,
                    "diversity_score": 0.0,
                    "analysis_summary": "Aucune sélection B-roll"
                }
            
            # Calculer le score de qualité global
            context_scores = [s.get('context_match_score', 0.0) for s in broll_selections]
            avg_context_score = sum(context_scores) / len(context_scores) if context_scores else 0.0
            
            # Calculer le score de diversité
            diversity_score = self._calculate_diversity_score(broll_selections)
            
            # Calculer le score de pertinence contextuelle
            context_relevance_score = self._calculate_context_relevance_score(broll_selections, context_analysis)
            
            # Score de qualité global pondéré
            overall_quality_score = (
                avg_context_score * 0.4 +
                diversity_score * 0.3 +
                context_relevance_score * 0.3
            )
            
            analysis_results = {
                "overall_quality_score": overall_quality_score,
                "context_relevance_score": context_relevance_score,
                "diversity_score": diversity_score,
                "average_context_score": avg_context_score,
                "selections_count": len(broll_selections),
                "analysis_summary": self._generate_analysis_summary(broll_selections, overall_quality_score),
                "quality_breakdown": {
                    "context_quality": avg_context_score,
                    "diversity_quality": diversity_score,
                    "relevance_quality": context_relevance_score
                }
            }
            
            logger.info(f"SUCCESS: Analyse des resultats terminee - Score global: {overall_quality_score:.2f}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse des résultats: {e}")
            return {
                "overall_quality_score": 0.0,
                "context_relevance_score": 0.0,
                "diversity_score": 0.0,
                "analysis_summary": f"Erreur d'analyse: {e}"
            }

    def _calculate_diversity_score(self, broll_selections: List[Dict[str, Any]]) -> float:
        """Calcule le score de diversité des B-rolls sélectionnés"""
        try:
            if not broll_selections:
                return 0.0
            
            broll_types = set()
            broll_sources = set()
            broll_themes = set()
            
            for selection in broll_selections:
                if not isinstance(selection, dict):
                    continue
                selected_broll = selection.get('selected_broll') or {}
                if not isinstance(selected_broll, dict):
                    selected_broll = {}
                
                file_path = selected_broll.get('file_path') or ''
                if isinstance(file_path, str) and file_path:
                    file_type = Path(file_path).suffix.lower()
                    if file_type:
                        broll_types.add(file_type)
                
                source = selected_broll.get('source') or 'unknown'
                if isinstance(source, str) and source:
                    broll_sources.add(source)
                else:
                    broll_sources.add('unknown')
                
                context = selection.get('segment_context') or 'unknown'
                if isinstance(context, str) and context:
                    broll_themes.add(context)
                else:
                    broll_themes.add('unknown')
            
            type_diversity = min(1.0, len(broll_types) / 3.0)
            source_diversity = min(1.0, len(broll_sources) / 2.0)
            theme_diversity = min(1.0, len(broll_themes) / 5.0)
            
            diversity_score = (
                type_diversity * 0.4 +
                source_diversity * 0.3 +
                theme_diversity * 0.3
            )
            
            return diversity_score
        except Exception as e:
            logger.warning(f"Erreur calcul diversité: {e}")
            return 0.5

    def _calculate_context_relevance_score(self, broll_selections: List[Dict[str, Any]], 
                                        context_analysis: Dict[str, Any]) -> float:
        """Calcule le score de pertinence contextuelle"""
        try:
            if not broll_selections:
                return 0.0
            
            # Extraire le thème principal du contexte
            main_theme = context_analysis.get('global_analysis', {}).get('main_theme', 'general')
            
            # Calculer la pertinence de chaque sélection
            relevance_scores = []
            
            for selection in broll_selections:
                segment_context = selection.get('segment_context', 'unknown')
                context_match_score = selection.get('context_match_score', 0.0)
                
                # Bonus si le contexte du segment correspond au thème principal
                theme_bonus = 0.2 if segment_context == main_theme else 0.0
                
                # Score de pertinence final
                relevance_score = min(1.0, context_match_score + theme_bonus)
                relevance_scores.append(relevance_score)
            
            # Retourner le score moyen
            return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Erreur calcul pertinence contextuelle: {e}")
            return 0.5

    def _generate_analysis_summary(self, broll_selections: List[Dict[str, Any]], 
                                 overall_quality_score: float) -> str:
        """Génère un résumé de l'analyse"""
        try:
            if not broll_selections:
                return "Aucune sélection B-roll disponible"
            
            # Compter les sélections par contexte
            context_counts = {}
            for selection in broll_selections:
                context = selection.get('segment_context', 'unknown')
                context_counts[context] = context_counts.get(context, 0) + 1
            
            # Générer le résumé
            if overall_quality_score >= 0.8:
                quality_level = "excellente"
            elif overall_quality_score >= 0.6:
                quality_level = "bonne"
            elif overall_quality_score >= 0.4:
                quality_level = "moyenne"
            else:
                quality_level = "faible"
            
            summary = f"Qualité {quality_level} ({overall_quality_score:.2f}/1.0) - "
            summary += f"{len(broll_selections)} B-rolls sélectionnés dans {len(context_counts)} contextes"
            
            return summary
            
        except Exception as e:
            logger.warning(f"Erreur génération résumé: {e}")
            return "Résumé non disponible"

    async def _enrich_metadata(self, transcript_data: Dict[str, Any], 
                             context_analysis: Dict[str, Any], 
                             broll_selections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrichit les métadonnées avec les informations du pipeline"""
        try:
            enriched_metadata = {
                "transcript_metadata": transcript_data.get('metadata', {}),
                "pipeline_metadata": {
                    "version": self.config["pipeline"]["version"],
                    "processing_timestamp": datetime.now().isoformat(),
                    "components_used": {
                        "context_analyzer": "advanced" if self.context_analyzer else "fallback",
                        "broll_selector": "advanced" if self.broll_selector else "fallback"
                    }
                },
                "context_metadata": {
                    "main_theme": context_analysis.get('global_analysis', {}).get('main_theme', 'unknown'),
                    "sub_themes": context_analysis.get('global_analysis', {}).get('sub_themes', []),
                    "overall_tone": context_analysis.get('global_analysis', {}).get('overall_tone', 'neutral'),
                    "target_audience": context_analysis.get('global_analysis', {}).get('target_audience', 'general_public')
                },
                "broll_metadata": {
                    "total_selections": len(broll_selections),
                    "contexts_covered": list(set(s.get('segment_context', 'unknown') for s in broll_selections)),
                    "average_context_score": sum(s.get('context_match_score', 0.0) for s in broll_selections) / len(broll_selections) if broll_selections else 0.0
                },
                "quality_indicators": {
                    "context_coherence": context_analysis.get('context_scores', {}).get('theme_consistency', 0.0),
                    "narrative_flow": context_analysis.get('narrative_coherence', {}).get('coherence_score', 0.0),
                    "broll_relevance": sum(s.get('context_match_score', 0.0) for s in broll_selections) / len(broll_selections) if broll_selections else 0.0
                }
            }
            
            logger.info("SUCCESS: Metadonnees enrichies generees avec succes")
            return enriched_metadata
            
        except Exception as e:
            logger.error(f"❌ Erreur enrichissement métadonnées: {e}")
            return {
                "error": f"Erreur enrichissement: {e}",
                "timestamp": datetime.now().isoformat()
            }

    def _create_error_response(self, request_id: str, error_message: str, 
                             processing_time: float) -> Dict[str, Any]:
        """Crée une réponse d'erreur"""
        return {
            "request_id": request_id,
            "pipeline_status": "error",
            "error_message": error_message,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": self.config["pipeline"]["version"]
        }

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Obtient le statut actuel du pipeline"""
        return {
            "status": "operational" if self.context_analyzer and self.broll_selector else "degraded",
            "version": self.config["pipeline"]["version"],
            "components_status": {
                "context_analyzer": "operational" if self.context_analyzer else "unavailable",
                "broll_selector": "operational" if self.broll_selector else "unavailable"
            },
            "processing_stats": self.processing_stats,
            "configuration": {
                "analysis": self.config["analysis"],
                "broll_selection": self.config["broll_selection"],
                "performance": self.config["performance"]
            },
            "last_update": datetime.now().isoformat()
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """Obtient les statistiques de la base de données B-roll"""
        try:
            if self.broll_selector:
                return self.broll_selector.get_database_stats()
            else:
                return {"error": "Sélecteur B-roll non disponible"}
        except Exception as e:
            logger.error(f"❌ Erreur obtention stats base: {e}")
            return {"error": str(e)}

    def update_pipeline_config(self, new_config: Dict[str, Any]) -> bool:
        """Met à jour la configuration du pipeline"""
        try:
            for key, value in new_config.items():
                if key in self.config:
                    if isinstance(value, dict) and isinstance(self.config[key], dict):
                        self._merge_configs(self.config[key], value)
                    else:
                        self.config[key] = value
                    logger.info(f"✅ Configuration mise à jour: {key}")
                else:
                    logger.warning(f"⚠️ Clé de configuration inconnue: {key}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour configuration: {e}")
            return False

    def close_pipeline(self):
        """Ferme le pipeline et libère les ressources"""
        try:
            if self.broll_selector:
                self.broll_selector.close_database()
            
            logger.info("Pipeline B-roll avancé fermé")
            
        except Exception as e:
            logger.error(f"❌ Erreur fermeture pipeline: {e}")

    def _extract_main_themes_from_segments(self, segments: List[Dict]) -> List[str]:
        """Extrait les thèmes principaux des segments de transcription"""
        try:
            # Mots-clés thématiques avec pondération
            theme_keywords = {
                'neuroscience': ['brain', 'neural', 'cognitive', 'mind', 'psychology', 'mental'],
                'technology': ['ai', 'artificial', 'intelligence', 'computer', 'digital', 'tech'],
                'health': ['health', 'medical', 'doctor', 'treatment', 'patient', 'surgery'],
                'education': ['learn', 'study', 'education', 'knowledge', 'teaching', 'student'],
                'business': ['business', 'company', 'work', 'career', 'money', 'finance'],
                'science': ['science', 'research', 'discovery', 'experiment', 'laboratory']
            }
            
            theme_scores = {}
            all_text = ' '.join([seg.get('text', '').lower() for seg in segments])
            
            for theme, keywords in theme_keywords.items():
                score = sum(all_text.count(keyword) for keyword in keywords)
                if score > 0:
                    theme_scores[theme] = score
            
            # Trier par score et retourner les 3 premiers
            sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
            return [theme for theme, score in sorted_themes[:3]]
            
        except Exception as e:
            logger.warning(f"Erreur extraction thèmes: {e}")
            return ['general']

    def _analyze_overall_tone(self, segments: List[Dict]) -> str:
        """Analyse le ton global de la transcription"""
        try:
            # Mots indicateurs de ton
            positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'enjoy', 'success'}
            negative_words = {'bad', 'terrible', 'awful', 'hate', 'fail', 'problem', 'difficult', 'pain'}
            neutral_words = {'think', 'believe', 'consider', 'analyze', 'discuss', 'explain', 'describe'}
            
            all_text = ' '.join([seg.get('text', '').lower() for seg in segments])
            words = set(all_text.split())
            
            positive_count = len(words.intersection(positive_words))
            negative_count = len(words.intersection(negative_words))
            neutral_count = len(words.intersection(neutral_words))
            
            if positive_count > negative_count and positive_count > neutral_count:
                return 'positive'
            elif negative_count > positive_count and negative_count > neutral_count:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.warning(f"Erreur analyse ton: {e}")
            return 'neutral'

    def _analyze_complexity_profile(self, segments: List[Dict]) -> str:
        """Analyse le profil de complexité de la transcription"""
        try:
            total_words = sum(len(seg.get('text', '').split()) for seg in segments)
            avg_words_per_segment = total_words / len(segments) if segments else 0
            
            if avg_words_per_segment < 10:
                return 'simple'
            elif avg_words_per_segment < 20:
                return 'medium'
            else:
                return 'complex'
                
        except Exception as e:
            logger.warning(f"Erreur analyse complexité: {e}")
            return 'medium'

    def _determine_target_audience(self, complexity_profile: str) -> str:
        """Détermine le public cible basé sur la complexité"""
        complexity_mapping = {
            'simple': 'general_public',
            'medium': 'educated_public',
            'complex': 'specialists'
        }
        return complexity_mapping.get(complexity_profile, 'general_public')

    def _determine_local_context(self, text: str) -> str:
        """Détermine le contexte local d'un segment"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['ai', 'artificial', 'intelligence']):
            return 'technology'
        elif any(word in text_lower for word in ['brain', 'mind', 'think']):
            return 'neuroscience'
        elif any(word in text_lower for word in ['learn', 'study', 'education']):
            return 'education'
        else:
            return 'general'

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extrait les mots-clés d'un texte"""
        try:
            # Mots-clés importants avec pondération
            important_words = ['ai', 'artificial', 'intelligence', 'brain', 'technology', 'future', 'important', 'key']
            text_lower = text.lower()
            
            keywords = []
            for word in important_words:
                if word in text_lower:
                    keywords.append(word)
            
            # Ajouter des mots uniques du texte
            words = text_lower.split()
            unique_words = [word for word in words if len(word) > 4 and word not in keywords]
            keywords.extend(unique_words[:3])
            
            return keywords[:5]  # Limiter à 5 mots-clés
            
        except Exception as e:
            logger.warning(f"Erreur extraction mots-clés: {e}")
            return []

    def _analyze_sentiment_simple(self, text: str) -> str:
        """Analyse de sentiment simple basée sur des règles"""
        try:
            positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'enjoy', 'success'}
            negative_words = {'bad', 'terrible', 'awful', 'hate', 'fail', 'problem', 'difficult', 'pain'}
            
            text_lower = text.lower()
            words = set(text_lower.split())
            
            positive_count = len(words.intersection(positive_words))
            negative_count = len(words.intersection(negative_words))
            
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.warning(f"Erreur analyse sentiment: {e}")
            return 'neutral'

    def _analyze_complexity_simple(self, text: str) -> str:
        """Analyse de complexité simple basée sur des règles"""
        try:
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            if avg_word_length < 5:
                return 'simple'
            elif avg_word_length < 7:
                return 'medium'
            else:
                return 'complex'
                
        except Exception as e:
            logger.warning(f"Erreur analyse complexité: {e}")
            return 'medium'

    def _suggest_appropriate_brolls(self, text: str) -> List[str]:
        """Suggère des types de B-roll appropriés pour un segment"""
        try:
            text_lower = text.lower()
            suggestions = []
            
            if any(word in text_lower for word in ['ai', 'artificial', 'intelligence']):
                suggestions.extend(['technology', 'digital', 'innovation'])
            if any(word in text_lower for word in ['brain', 'mind', 'think']):
                suggestions.extend(['neuroscience', 'research', 'laboratory'])
            if any(word in text_lower for word in ['future', 'tomorrow', 'next']):
                suggestions.extend(['futuristic', 'modern', 'progress'])
            
            return list(set(suggestions))[:3]  # Limiter à 3 suggestions
            
        except Exception as e:
            logger.warning(f"Erreur suggestions B-roll: {e}")
            return ['general']

    def _analyze_narrative_transitions(self, segments: List[Dict]) -> List[Dict]:
        """Analyse les transitions narratives entre segments"""
        try:
            transitions = []
            
            for i in range(1, len(segments)):
                prev_text = segments[i-1].get('text', '').lower()
                curr_text = segments[i].get('text', '').lower()
                
                # Détecter les transitions logiques
                transition_words = ['but', 'however', 'therefore', 'meanwhile', 'then', 'next', 'finally']
                
                for word in transition_words:
                    if word in curr_text:
                        transitions.append({
                            'from_segment': i-1,
                            'to_segment': i,
                            'transition_type': word,
                            'strength': 0.7
                        })
                        break
            
            return transitions
            
        except Exception as e:
            logger.warning(f"Erreur analyse transitions: {e}")
            return [] 

    async def _save_metadata(self, request_id: str, transcript_data: Dict[str, Any], 
                            context_analysis: Dict[str, Any], broll_selections: List[Dict[str, Any]], 
                            results_analysis: Dict[str, Any]):
        """Sauvegarde les métadonnées de traitement"""
        try:
            # Créer le répertoire de sortie s'il n'existe pas
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Préparer les métadonnées à sauvegarder
            metadata = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "transcript_info": {
                    "title": transcript_data.get("metadata", {}).get("title", "Unknown"),
                    "duration": transcript_data.get("metadata", {}).get("duration", 0.0),
                    "language": transcript_data.get("metadata", {}).get("language", "en"),
                    "segments_count": len(transcript_data.get("segments", []))
                },
                "context_analysis": {
                    "overall_confidence": context_analysis.get("context_scores", {}).get("overall_confidence", 0.0),
                    "main_theme": context_analysis.get("global_analysis", {}).get("main_theme", "unknown"),
                    "analysis_type": "advanced" if context_analysis.get("context_scores", {}).get("overall_confidence", 0.0) > 0.6 else "fallback"
                },
                "broll_selections": {
                    "total_selections": len(broll_selections),
                    "selections_per_segment": [len(sel.get("brolls", [])) for sel in broll_selections]
                },
                "results_analysis": {
                    "overall_quality_score": results_analysis.get("overall_quality_score", 0.0),
                    "diversity_score": results_analysis.get("diversity_score", 0.0),
                    "context_relevance_score": results_analysis.get("context_relevance_score", 0.0)
                },
                "pipeline_status": "success"
            }
            
            # Sauvegarder dans report.json
            report_file = output_dir / "report.json"
            
            # Charger le rapport existant s'il existe
            existing_report = {}
            if report_file.exists():
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        existing_report = json.load(f)
                except Exception as e:
                    logger.warning(f"⚠️ Impossible de charger le rapport existant: {e}")
            
            # Ajouter la nouvelle entrée
            if "requests" not in existing_report:
                existing_report["requests"] = []
            
            existing_report["requests"].append(metadata)
            existing_report["last_updated"] = datetime.now().isoformat()
            existing_report["total_requests"] = len(existing_report["requests"])
            
            # Sauvegarder le rapport mis à jour
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(existing_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"SUCCESS: Metadonnees sauvegardees dans {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde des métadonnées: {e}")
            # Ne pas faire échouer le pipeline pour une erreur de sauvegarde 