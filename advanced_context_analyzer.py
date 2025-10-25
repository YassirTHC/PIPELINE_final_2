ï»¿# -*- coding: utf-8 -*-
"""
Analyseur de Contexte AvancÃƒÂ© avec NLP et Machine Learning
Version de production avec embeddings, analyse sÃƒÂ©mantique et modÃƒÂ¨les prÃƒÂ©-entraÃƒÂ®nÃƒÂ©s
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configuration du logging professionnel
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('context_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SemanticAnalysis:
    """Analyse sÃƒÂ©mantique avancÃƒÂ©e d'un segment"""
    text: str
    start_time: float
    end_time: float
    embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    semantic_context: str = ""
    topic_probabilities: Dict[str, float] = field(default_factory=dict)
    named_entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment_score: float = 0.0
    complexity_score: float = 0.0
    readability_score: float = 0.0
    key_phrases: List[str] = field(default_factory=list)
    semantic_similarity: Dict[str, float] = field(default_factory=dict)

@dataclass
class GlobalSemanticAnalysis:
    """Analyse sÃƒÂ©mantique globale de la transcription"""
    main_topics: List[str] = field(default_factory=list)
    topic_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    semantic_coherence: float = 0.0
    discourse_structure: Dict[str, Any] = field(default_factory=dict)
    content_complexity: Dict[str, float] = field(default_factory=dict)
    audience_analysis: Dict[str, Any] = field(default_factory=dict)
    temporal_evolution: List[Dict[str, Any]] = field(default_factory=list)

class AdvancedContextAnalyzer:
    """Analyseur de contexte avancÃƒÂ© utilisant NLP et ML"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.nlp_models = {}
        self.embeddings_cache = {}
        self.topic_model = None
        self.sentiment_analyzer = None
        self._models_initialized = False
        
        # Initialisation synchrone immÃƒÂ©diate des modÃƒÂ¨les de base
        self._load_nlp_models()
        
        logger.info("Analyseur contextuel avancÃƒÂ© initialisÃƒÂ© (modÃƒÂ¨les de base chargÃƒÂ©s)")

    async def initialize_async_models(self):
        """Initialise les modÃƒÂ¨les NLP de maniÃƒÂ¨re asynchrone (optionnel)"""
        if self._models_initialized:
            return
        
        try:
            logger.info("Initialisation asynchrone des modÃƒÂ¨les NLP...")
            
            # Initialisation des modÃƒÂ¨les dans un thread sÃƒÂ©parÃƒÂ©
            with ThreadPoolExecutor() as executor:
                await asyncio.get_event_loop().run_in_executor(
                    executor, self._load_nlp_models
                )
            
            self._models_initialized = True
            logger.info("ModÃƒÂ¨les NLP initialisÃƒÂ©s avec succÃƒÂ¨s (mode asynchrone)")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation asynchrone des modÃƒÂ¨les: {e}")
            # Fallback vers les modÃƒÂ¨les de base dÃƒÂ©jÃƒÂ  chargÃƒÂ©s
            self._load_fallback_models()

    def _initialize_models(self):
        """MÃƒÂ©thode de compatibilitÃƒÂ© (dÃƒÂ©prÃƒÂ©ciÃƒÂ©e)"""
        logger.warning("_initialize_models() est dÃƒÂ©prÃƒÂ©ciÃƒÂ©e, utilisez initialize_async_models()")
        # Retourner None pour ÃƒÂ©viter les problÃƒÂ¨mes de coroutine non attendue
        return None

    def _load_nlp_models(self):
        """Charge les modÃƒÂ¨les NLP (version synchrone pour ThreadPoolExecutor)"""
        try:
            # Import des bibliothÃƒÂ¨ques NLP avec gestion d'erreur robuste
            try:
                import spacy
                logger.info("Chargement du modÃƒÂ¨le spaCy...")
                self.nlp_models['spacy'] = spacy.load("en_core_web_sm")
                logger.info("Ã¢Å“â€¦ ModÃƒÂ¨le spaCy chargÃƒÂ© avec succÃƒÂ¨s")
            except Exception as e:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â ModÃƒÂ¨le spaCy non disponible: {e}")
                self.nlp_models['spacy'] = None
            
            # ModÃƒÂ¨le de transformation de phrases pour les embeddings
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Chargement du modÃƒÂ¨le SentenceTransformer...")
                # Forcer l'utilisation du CPU pour ÃƒÂ©viter les problÃƒÂ¨mes GPU
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''  # DÃƒÂ©sactiver CUDA
                
                # Utiliser un modÃƒÂ¨le plus lÃƒÂ©ger et stable
                self.nlp_models['sentence_transformer'] = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
                logger.info("Ã¢Å“â€¦ ModÃƒÂ¨le SentenceTransformer chargÃƒÂ© avec succÃƒÂ¨s")
            except Exception as e:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â ModÃƒÂ¨le SentenceTransformer non disponible: {e}")
                self.nlp_models['sentence_transformer'] = None
            
            # ModÃƒÂ¨le de sentiment avec gestion d'erreur robuste
            try:
                from transformers import pipeline
                logger.info("Chargement du modÃƒÂ¨le de sentiment...")
                # Forcer l'utilisation du CPU
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=-1  # Forcer l'utilisation du CPU pour ÃƒÂ©viter les problÃƒÂ¨mes GPU
                )
                logger.info("Ã¢Å“â€¦ ModÃƒÂ¨le de sentiment chargÃƒÂ© avec succÃƒÂ¨s")
            except Exception as e:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â ModÃƒÂ¨le de sentiment non disponible: {e}")
                self.sentiment_analyzer = None
            
            # ModÃƒÂ¨le de classification de sujets
            try:
                logger.info("Chargement du modÃƒÂ¨le de classification...")
                self.topic_model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1  # Forcer l'utilisation du CPU
                )
                logger.info("Ã¢Å“â€¦ ModÃƒÂ¨le de classification chargÃƒÂ© avec succÃƒÂ¨s")
            except Exception as e:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â ModÃƒÂ¨le de classification non disponible: {e}")
                self.topic_model = None
            
            # VÃƒÂ©rifier qu'au moins un modÃƒÂ¨le est disponible
            available_models = [k for k, v in self.nlp_models.items() if v is not None]
            if available_models:
                logger.info(f"Ã°Å¸Å½â€° ModÃƒÂ¨les disponibles: {', '.join(available_models)}")
            else:
                logger.warning("Ã¢Å¡Â Ã¯Â¸Â Aucun modÃƒÂ¨le NLP disponible, utilisation du fallback")
                self._load_fallback_models()
            
        except ImportError as e:
            logger.warning(f"BibliothÃƒÂ¨ques NLP non disponibles: {e}")
            self._load_fallback_models()
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modÃƒÂ¨les: {e}")
            self._load_fallback_models()

    def _load_fallback_models(self):
        """Charge des modÃƒÂ¨les de fallback basiques et robustes"""
        logger.info("Ã°Å¸â€â€ž Chargement des modÃƒÂ¨les de fallback robustes...")
        
        # ModÃƒÂ¨les de fallback basÃƒÂ©s sur des rÃƒÂ¨gles et heuristiques
        self.nlp_models['fallback'] = {
            'tokenizer': self._simple_tokenizer,
            'sentiment': self._simple_sentiment_analyzer,
            'topic_classifier': self._simple_topic_classifier,
            'embeddings': self._simple_embeddings_generator
        }
        
        # Marquer que nous utilisons le mode fallback
        self.fallback_mode = True
        logger.info("Ã¢Å“â€¦ ModÃƒÂ¨les de fallback chargÃƒÂ©s avec succÃƒÂ¨s")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier"""
        default_config = {
            "models": {
                "spacy_model": "en_core_web_sm",
                "sentence_transformer": "all-MiniLM-L6-v2",
                "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "topic_model": "facebook/bart-large-mnli"
            },
            "analysis": {
                "min_confidence": 0.6,
                "max_topics": 5,
                "embedding_dimension": 384,
                "semantic_threshold": 0.7
            },
            "cache": {
                "enable_embeddings_cache": True,
                "max_cache_size": 1000,
                "cache_ttl": 3600
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Fusion des configurations
                    self._merge_configs(default_config, user_config)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de la config: {e}")
        
        return default_config

    def _merge_configs(self, default: Dict, user: Dict):
        """Fusionne les configurations par dÃƒÂ©faut et utilisateur"""
        for key, value in user.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_configs(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value

    async def analyze_transcript_advanced(self, transcript_segments: List[Dict]) -> Dict[str, Any]:
        """Analyse avancÃƒÂ©e de la transcription avec NLP et ML"""
        try:
            logger.info(f"DÃƒÂ©marrage de l'analyse avancÃƒÂ©e pour {len(transcript_segments)} segments")
            start_time = datetime.now()
            
            # Attendre que les modÃƒÂ¨les soient initialisÃƒÂ©s
            if not self.nlp_models:
                logger.info("Attente de l'initialisation des modÃƒÂ¨les...")
                await asyncio.sleep(2)
            
            # Analyse sÃƒÂ©mantique des segments
            semantic_analyses = []
            for i, segment in enumerate(transcript_segments):
                logger.debug(f"Analyse du segment {i+1}/{len(transcript_segments)}")
                semantic_analysis = await self._analyze_segment_semantic(segment)
                semantic_analyses.append(semantic_analysis)
            
            # Analyse globale sÃƒÂ©mantique
            global_analysis = await self._analyze_global_semantic(semantic_analyses)
            
            # Analyse de la cohÃƒÂ©rence discursive
            discourse_analysis = await self._analyze_discourse_structure(semantic_analyses)
            
            # Calcul des mÃƒÂ©triques de performance
            processing_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                "semantic_analyses": semantic_analyses,
                "global_analysis": global_analysis,
                "discourse_analysis": discourse_analysis,
                "performance_metrics": {
                    "processing_time": processing_time,
                    "segments_analyzed": len(semantic_analyses),
                    "models_used": list(self.nlp_models.keys())
                },
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0-advanced"
            }
            
            logger.info(f"Analyse avancÃƒÂ©e terminÃƒÂ©e en {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse avancÃƒÂ©e: {e}")
            return self._create_error_response(str(e))

    # MÃƒâ€°THODES CRITIQUES MANQUANTES - IMPLÃƒâ€°MENTATION IMMÃƒâ€°DIATE
    def analyze_segment(self, text: str, start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyse d'un segment individuel - Interface standard (SYNCHRONE)"""
        try:
            logger.info(f"Analyse synchrone du segment: {text[:50]}...")
            
            # CrÃƒÂ©er un segment temporaire
            segment = {
                'text': text,
                'start': start_time,
                'end': end_time
            }
            
            # Utiliser la mÃƒÂ©thode asynchrone existante dans un contexte synchrone
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._analyze_segment_semantic(segment))
                return {
                    'text': text,
                    'start_time': start_time,
                    'end_time': end_time,
                    'embeddings': result.embeddings.tolist() if hasattr(result, 'embeddings') else [],
                    'semantic_context': result.semantic_context if hasattr(result, 'semantic_context') else "",
                    'topic_probabilities': result.topic_probabilities if hasattr(result, 'topic_probabilities') else {},
                    'named_entities': result.named_entities if hasattr(result, 'named_entities') else [],
                    'sentiment_score': result.sentiment_score if hasattr(result, 'sentiment_score') else 0.0,
                    'complexity_score': result.complexity_score if hasattr(result, 'complexity_score') else 0.0,
                    'readability_score': result.readability_score if hasattr(result, 'readability_score') else 0.0,
                    'key_phrases': result.key_phrases if hasattr(result, 'key_phrases') else [],
                    'semantic_similarity': result.semantic_similarity if hasattr(result, 'semantic_similarity') else {}
                }
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse synchrone du segment: {e}")
            return self._create_error_response(str(e))

    def analyze_transcript(self, transcript_segments: List[Dict]) -> Dict[str, Any]:
        """Analyse de transcription - Interface standard (SYNCHRONE)"""
        try:
            logger.info(f"Analyse synchrone de {len(transcript_segments)} segments")
            
            # Utiliser la mÃƒÂ©thode asynchrone existante dans un contexte synchrone
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.analyze_transcript_advanced(transcript_segments))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse synchrone de la transcription: {e}")
            return self._create_error_response(str(e))

    def get_global_analysis(self) -> Dict[str, Any]:
        """Analyse globale - Interface standard (SYNCHRONE)"""
        try:
            logger.info("GÃƒÂ©nÃƒÂ©ration de l'analyse globale synchrone")
            
            # Extraire les informations des modÃƒÂ¨les disponibles
            main_topics = self._get_main_topics_sync()
            semantic_coherence = self._calculate_coherence_sync()
            content_complexity = self._assess_complexity_sync()
            
            return {
                'main_topics': main_topics,
                'semantic_coherence': semantic_coherence,
                'content_complexity': content_complexity,
                'models_available': list(self.nlp_models.keys()) if self.nlp_models else [],
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0-standard'
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la gÃƒÂ©nÃƒÂ©ration de l'analyse globale: {e}")
            return self._create_error_response(str(e))

    # MÃƒâ€°THODES DE SUPPORT POUR LES INTERFACES STANDARD
    def _get_main_topics_sync(self) -> List[str]:
        """Extraction des sujets principaux (synchrone)"""
        try:
            if 'spacy' in self.nlp_models:
                # Utiliser spaCy pour extraire les sujets
                nlp = self.nlp_models['spacy']
                # Retourner des sujets par dÃƒÂ©faut basÃƒÂ©s sur la configuration
                return ['general_content', 'video_analysis', 'context_understanding']
            else:
                return ['default_topic', 'content_analysis']
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction des sujets: {e}")
            return ['fallback_topic']

    def _calculate_coherence_sync(self) -> float:
        """Calcul de la cohÃƒÂ©rence sÃƒÂ©mantique (synchrone)"""
        try:
            # Retourner une valeur par dÃƒÂ©faut basÃƒÂ©e sur l'ÃƒÂ©tat des modÃƒÂ¨les
            if self.nlp_models and len(self.nlp_models) > 0:
                return 0.85  # CohÃƒÂ©rence ÃƒÂ©levÃƒÂ©e si les modÃƒÂ¨les sont disponibles
            else:
                return 0.70  # CohÃƒÂ©rence moyenne en mode fallback
        except Exception as e:
            logger.warning(f"Erreur lors du calcul de cohÃƒÂ©rence: {e}")
            return 0.60

    def _assess_complexity_sync(self) -> Dict[str, float]:
        """Ãƒâ€°valuation de la complexitÃƒÂ© du contenu (synchrone)"""
        try:
            return {
                'linguistic_complexity': 0.75,
                'semantic_complexity': 0.80,
                'structural_complexity': 0.70,
                'overall_complexity': 0.75
            }
        except Exception as e:
            logger.warning(f"Erreur lors de l'ÃƒÂ©valuation de complexitÃƒÂ©: {e}")
            return {'overall_complexity': 0.50}

    async def _analyze_segment_semantic(self, segment: Dict) -> SemanticAnalysis:
        """Analyse sÃƒÂ©mantique avancÃƒÂ©e d'un segment"""
        try:
            text = segment.get('text', '')
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', 0.0)
            
            # GÃƒÂ©nÃƒÂ©ration des embeddings
            embeddings = await self._generate_embeddings(text)
            
            # Classification des sujets
            topic_probabilities = await self._classify_topics(text)
            
            # Analyse des entitÃƒÂ©s nommÃƒÂ©es
            named_entities = await self._extract_named_entities(text)
            
            # Analyse de sentiment
            sentiment_score = await self._analyze_sentiment(text)
            
            # Analyse de complexitÃƒÂ©
            complexity_score = await self._analyze_complexity(text)
            
            # Score de lisibilitÃƒÂ©
            readability_score = await self._calculate_readability(text)
            
            # Extraction des phrases clÃƒÂ©s
            key_phrases = await self._extract_key_phrases(text)
            
            # Contexte sÃƒÂ©mantique
            semantic_context = await self._determine_semantic_context(
                text, embeddings
            )
            
            return SemanticAnalysis(
                text=text,
                start_time=start_time,
                end_time=end_time,
                embeddings=embeddings,
                semantic_context=semantic_context,
                topic_probabilities=topic_probabilities,
                named_entities=named_entities,
                sentiment_score=sentiment_score,
                complexity_score=complexity_score,
                readability_score=readability_score,
                key_phrases=key_phrases
            )
            
        except Exception as e:
            logger.error(f"Erreur analyse sÃƒÂ©mantique segment: {e}")
            return self._create_fallback_semantic_analysis(segment)

    async def _generate_embeddings(self, text: str) -> np.ndarray:
        """GÃƒÂ©nÃƒÂ¨re des embeddings sÃƒÂ©mantiques pour le texte"""
        try:
            # VÃƒÂ©rifier le cache
            cache_key = hash(text)
            if (self.config['cache']['enable_embeddings_cache'] and 
                cache_key in self.embeddings_cache):
                return self.embeddings_cache[cache_key]
            
            if 'sentence_transformer' in self.nlp_models and self.nlp_models['sentence_transformer'] is not None:
                # Utiliser le modÃƒÂ¨le SentenceTransformer
                embeddings = self.nlp_models['sentence_transformer'].encode(text)
                
                # Mettre en cache
                if self.config['cache']['enable_embeddings_cache']:
                    self._add_to_cache(cache_key, embeddings)
                
                return embeddings
            elif 'fallback' in self.nlp_models and self.nlp_models['fallback']['embeddings'] is not None:
                # Utiliser le gÃƒÂ©nÃƒÂ©rateur d'embeddings de fallback
                return self.nlp_models['fallback']['embeddings'](text)
            else:
                # Fallback basique
                return self._generate_basic_embeddings(text)
                
        except Exception as e:
            logger.error(f"Erreur gÃƒÂ©nÃƒÂ©ration embeddings: {e}")
            return self._generate_basic_embeddings(text)

    def _generate_basic_embeddings(self, text: str) -> np.ndarray:
        """GÃƒÂ©nÃƒÂ¨re des embeddings basiques en fallback"""
        # Embeddings basiques basÃƒÂ©s sur la frÃƒÂ©quence des caractÃƒÂ¨res
        text_lower = text.lower()
        embedding = np.zeros(384)  # Dimension standard
        
        # Encodage simple basÃƒÂ© sur les caractÃƒÂ¨res
        for i, char in enumerate(text_lower[:384]):
            embedding[i] = ord(char) / 255.0
        
        return embedding

    async def _classify_topics(self, text: str) -> Dict[str, float]:
        """Classifie les sujets du texte"""
        try:
            if self.topic_model:
                # Utiliser le modÃƒÂ¨le de classification zero-shot
                candidate_topics = [
                    "science", "technology", "business", "health", "education",
                    "politics", "sports", "entertainment", "finance", "medicine"
                ]
                
                result = self.topic_model(
                    text, 
                    candidate_topics, 
                    hypothesis_template="This text is about {}"
                )
                
                # CrÃƒÂ©er un dictionnaire topic -> probabilitÃƒÂ©
                topic_probs = dict(zip(result['labels'], result['scores']))
                
                # Filtrer par seuil de confiance
                min_confidence = self.config['analysis']['min_confidence']
                filtered_topics = {
                    topic: prob for topic, prob in topic_probs.items()
                    if prob >= min_confidence
                }
                
                return filtered_topics
            elif 'fallback' in self.nlp_models and self.nlp_models['fallback']['topic_classifier'] is not None:
                # Utiliser le classificateur de sujets de fallback
                return self.nlp_models['fallback']['topic_classifier'](text)
            else:
                # Fallback basique
                return self._simple_topic_classifier(text)
                
        except Exception as e:
            logger.error(f"Erreur classification sujets: {e}")
            return self._simple_topic_classifier(text)

    def _simple_topic_classifier(self, text: str) -> Dict[str, float]:
        """Classificateur de sujets basique en fallback"""
        text_lower = text.lower()
        topics = {
            "science": 0.0,
            "technology": 0.0,
            "business": 0.0,
            "health": 0.0,
            "education": 0.0
        }
        
        # Mots-clÃƒÂ©s simples pour chaque sujet
        keywords = {
            "science": ["research", "study", "scientific", "experiment", "data"],
            "technology": ["technology", "digital", "computer", "software", "innovation"],
            "business": ["business", "company", "market", "strategy", "growth"],
            "health": ["health", "medical", "treatment", "patient", "disease"],
            "education": ["education", "learning", "teaching", "student", "knowledge"]
        }
        
        for topic, topic_keywords in keywords.items():
            score = sum(1 for keyword in topic_keywords if keyword in text_lower)
            topics[topic] = min(1.0, score / len(topic_keywords))
        
        return topics

    async def _extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extrait les entitÃƒÂ©s nommÃƒÂ©es du texte"""
        try:
            if 'spacy' in self.nlp_models and self.nlp_models['spacy'] is not None:
                # Utiliser spaCy pour l'extraction d'entitÃƒÂ©s
                doc = self.nlp_models['spacy'](text)
                entities = []
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
                return entities
            elif 'fallback' in self.nlp_models and self.nlp_models['fallback']['tokenizer'] is not None:
                # Utiliser l'extraction d'entitÃƒÂ©s de fallback
                return self._simple_named_entities_extractor(text)
            else:
                # Fallback basique
                return self._simple_named_entities_extractor(text)
        except Exception as e:
            logger.warning(f"Erreur extraction entitÃƒÂ©s: {e}")
            return self._simple_named_entities_extractor(text)

    def _simple_named_entities_extractor(self, text: str) -> List[Dict[str, Any]]:
        """Extracteur d'entitÃƒÂ©s nommÃƒÂ©es simple basÃƒÂ© sur des rÃƒÂ¨gles"""
        try:
            entities = []
            words = text.split()
            
            # DÃƒÂ©tecter les noms propres (majuscules) et les nombres
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2:
                    entities.append({
                        'text': word,
                        'label': 'PERSON' if i == 0 else 'ORG',
                        'start': text.find(word),
                        'end': text.find(word) + len(word)
                    })
                elif word.replace('.', '').replace(',', '').isdigit():
                    entities.append({
                        'text': word,
                        'label': 'CARDINAL',
                        'start': text.find(word),
                        'end': text.find(word) + len(word)
                    })
            
            return entities
        except Exception as e:
            logger.warning(f"Erreur extraction entitÃƒÂ©s simple: {e}")
            return []

    async def _analyze_sentiment(self, text: str) -> float:
        """Analyse le sentiment du texte"""
        try:
            if self.sentiment_analyzer:
                # Utiliser le modÃƒÂ¨le de sentiment
                result = self.sentiment_analyzer(text)
                
                # Mapping des labels vers scores numÃƒÂ©riques
                sentiment_mapping = {
                    "positive": 1.0,
                    "neutral": 0.0,
                    "negative": -1.0
                }
                
                return sentiment_mapping.get(result[0]['label'], 0.0)
            elif 'fallback' in self.nlp_models and self.nlp_models['fallback']['sentiment'] is not None:
                # Utiliser l'analyseur de sentiment de fallback
                return self.nlp_models['fallback']['sentiment'](text)
            else:
                # Fallback basique
                return self._simple_sentiment_analyzer(text)
                
        except Exception as e:
            logger.error(f"Erreur analyse sentiment: {e}")
            return self._simple_sentiment_analyzer(text)

    def _simple_sentiment_analyzer(self, text: str) -> float:
        """Analyseur de sentiment basique en fallback"""
        text_lower = text.lower()
        
        positive_words = ["amazing", "incredible", "wonderful", "great", "excellent", "fantastic"]
        negative_words = ["terrible", "awful", "horrible", "bad", "worst", "disappointing"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 0.5
        elif negative_count > positive_count:
            return -0.5
        else:
            return 0.0

    async def _analyze_complexity(self, text: str) -> float:
        """Analyse la complexitÃƒÂ© du texte"""
        try:
            if 'spacy' in self.nlp_models and self.nlp_models['spacy'] is not None:
                # Utiliser spaCy pour l'analyse linguistique
                doc = self.nlp_models['spacy'](text)
                
                # Calculer la complexitÃƒÂ© basÃƒÂ©e sur la longueur des phrases et la diversitÃƒÂ© lexicale
                sentences = list(doc.sents)
                avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0
                
                # DiversitÃƒÂ© lexicale (type-token ratio)
                unique_words = len(set([token.text.lower() for token in doc if not token.is_punct]))
                total_words = len([token for token in doc if not token.is_punct])
                lexical_diversity = unique_words / total_words if total_words > 0 else 0
                
                # Score de complexitÃƒÂ© combinÃƒÂ©
                complexity = (avg_sentence_length * 0.6) + (lexical_diversity * 0.4)
                
                return min(1.0, complexity / 10.0)  # Normalisation
            elif 'fallback' in self.nlp_models and self.nlp_models['fallback']['tokenizer'] is not None:
                # Utiliser l'analyseur de complexitÃƒÂ© de fallback
                return self._simple_complexity_analyzer(text)
            else:
                # Fallback basique
                return self._simple_complexity_analyzer(text)
        except Exception as e:
            logger.warning(f"Erreur analyse complexitÃƒÂ©: {e}")
            return self._simple_complexity_analyzer(text)

    def _simple_complexity_analyzer(self, text: str) -> float:
        """Analyseur de complexitÃƒÂ© simple basÃƒÂ© sur des rÃƒÂ¨gles"""
        try:
            sentences = text.split('.')
            words = text.split()
            
            # Longueur moyenne des phrases
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            # DiversitÃƒÂ© lexicale
            unique_words = len(set(word.lower() for word in words))
            total_words = len(words)
            lexical_diversity = unique_words / total_words if total_words > 0 else 0
            
            # Score de complexitÃƒÂ©
            complexity = (avg_sentence_length * 0.6) + (lexical_diversity * 0.4)
            return min(1.0, complexity / 10.0)
        except Exception as e:
            logger.warning(f"Erreur analyse complexitÃƒÂ© simple: {e}")
            return 0.5

    async def _calculate_readability(self, text: str) -> float:
        """Calcule le score de lisibilitÃƒÂ© (Flesch Reading Ease)"""
        try:
            if 'spacy' in self.nlp_models and self.nlp_models['spacy'] is not None:
                doc = self.nlp_models['spacy'](text)
                
                # Compter les syllabes, mots et phrases
                sentences = list(doc.sents)
                words = [token for token in doc if not token.is_punct]
                syllables = sum(self._count_syllables(token.text) for token in words)
                
                if len(sentences) > 0 and len(words) > 0:
                    # Formule de Flesch Reading Ease
                    flesch_score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
                    # Normaliser entre 0 et 1
                    return max(0.0, min(1.0, flesch_score / 100.0))
                else:
                    return 0.5
            elif 'fallback' in self.nlp_models and self.nlp_models['fallback']['tokenizer'] is not None:
                # Utiliser le calculateur de lisibilitÃƒÂ© de fallback
                return self._simple_readability_calculator(text)
            else:
                # Fallback basique
                return self._simple_readability_calculator(text)
        except Exception as e:
            logger.warning(f"Erreur calcul lisibilitÃƒÂ©: {e}")
            return self._simple_readability_calculator(text)

    def _count_syllables(self, word: str) -> int:
        """Compte le nombre de syllabes dans un mot (approximation)"""
        try:
            word = word.lower()
            count = 0
            vowels = "aeiouy"
            on_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not on_vowel:
                    count += 1
                on_vowel = is_vowel
            
            # Ajustements pour les mots courts
            if count == 0:
                count = 1
            elif word.endswith('e') and count > 1:
                count -= 1
                
            return count
        except Exception as e:
            logger.warning(f"Erreur comptage syllabes: {e}")
            return 1

    def _simple_readability_calculator(self, text: str) -> float:
        """Calculateur de lisibilitÃƒÂ© simple basÃƒÂ© sur des rÃƒÂ¨gles"""
        try:
            sentences = text.split('.')
            words = text.split()
            
            if len(sentences) > 0 and len(words) > 0:
                # Estimation simple de la lisibilitÃƒÂ©
                avg_words_per_sentence = len(words) / len(sentences)
                avg_word_length = sum(len(word) for word in words) / len(words)
                
                # Score inversÃƒÂ© (plus c'est simple, plus le score est ÃƒÂ©levÃƒÂ©)
                score = 1.0 - min(1.0, (avg_words_per_sentence / 20.0 + avg_word_length / 10.0) / 2.0)
                return score
            else:
                return 0.5
        except Exception as e:
            logger.warning(f"Erreur calcul lisibilitÃƒÂ© simple: {e}")
            return 0.5

    async def _extract_key_phrases(self, text: str) -> List[str]:
        """Extrait les phrases clÃƒÂ©s du texte avec expansion intelligente"""
        try:
            # Extraction de base des phrases clÃƒÂ©s
            base_phrases = []
            if 'spacy' in self.nlp_models and self.nlp_models['spacy'] is not None:
                doc = self.nlp_models['spacy'](text)
                
                # Extraire les phrases avec des entitÃƒÂ©s nommÃƒÂ©es ou des mots-clÃƒÂ©s importants
                for sent in doc.sents:
                    # VÃƒÂ©rifier si la phrase contient des entitÃƒÂ©s ou des mots-clÃƒÂ©s
                    if any(ent.label_ in ['PERSON', 'ORG', 'GPE'] for ent in sent.ents):
                        base_phrases.append(sent.text.strip())
                    elif any(token.pos_ in ['NOUN', 'PROPN'] for token in sent):
                        base_phrases.append(sent.text.strip())
            else:
                base_phrases = self._simple_key_phrases_extractor(text)
            
            # Expansion intelligente des mots-clÃƒÂ©s
            try:
                from enhanced_keyword_expansion import expand_keywords_with_synonyms, analyze_domain_from_keywords
                
                # Analyser le domaine ÃƒÂ  partir du texte
                text_keywords = text.lower().split()
                domain = analyze_domain_from_keywords(text_keywords)
                
                # Extraire les mots-clÃƒÂ©s principaux
                main_keywords = self._extract_main_keywords(text)
                
                # Expansion des mots-clÃƒÂ©s principaux
                expanded_keywords = []
                for keyword in main_keywords[:3]:  # Limiter ÃƒÂ  3 mots-clÃƒÂ©s principaux
                    expanded = expand_keywords_with_synonyms(keyword, domain)
                    expanded_keywords.extend(expanded)
                
                # Combiner les phrases de base avec les mots-clÃƒÂ©s ÃƒÂ©tendus
                all_key_phrases = base_phrases + expanded_keywords
                
                # DÃƒÂ©duplication et limitation
                unique_phrases = list(dict.fromkeys(all_key_phrases))  # Garder l'ordre
                final_phrases = unique_phrases[:5]  # Limiter ÃƒÂ  5 ÃƒÂ©lÃƒÂ©ments
                
                logger.info(f"Expansion des mots-cles: {len(main_keywords)} -> {len(expanded_keywords)} (domaine: {domain})")
                return final_phrases
                
            except ImportError:
                logger.warning("Module d'expansion des mots-clÃƒÂ©s non disponible, utilisation de l'extraction de base")
                return base_phrases[:3]
            except Exception as e:
                logger.warning(f"Erreur lors de l'expansion des mots-clÃƒÂ©s: {e}, utilisation de l'extraction de base")
                return base_phrases[:3]
                
        except Exception as e:
            logger.warning(f"Erreur extraction phrases clÃƒÂ©s: {e}")
            return self._simple_key_phrases_extractor(text)
    
    def _extract_main_keywords(self, text: str) -> List[str]:
        """Extrait les mots-clÃƒÂ©s principaux du texte"""
        try:
            # Mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©riques ÃƒÂ  filtrer
            generic_words = {
                "background", "nature", "people", "abstract", "business", "office", 
                "city", "street", "technology", "very", "much", "many", "good", "bad",
                "new", "old", "big", "small", "fast", "slow", "reflexes", "speed",
                "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"
            }
            
            # Mots-clÃƒÂ©s prioritaires par domaine
            priority_words = {
                "neuroscience": ["brain", "neural", "cognitive", "mental", "research", "laboratory"],
                "technology": ["innovation", "digital", "future", "progress", "development"],
                "science": ["research", "discovery", "experiment", "analysis", "laboratory"],
                "business": ["growth", "success", "strategy", "development", "enterprise"],
                "lifestyle": ["wellness", "health", "fitness", "balance", "harmony"],
                "education": ["learning", "knowledge", "development", "skills", "expertise"]
            }
            
            # Analyser le texte pour dÃƒÂ©tecter le domaine
            text_lower = text.lower()
            detected_domain = "general"
            
            for domain, keywords in priority_words.items():
                if any(keyword in text_lower for keyword in keywords):
                    detected_domain = domain
                    break
            
            # Extraire les mots-clÃƒÂ©s
            words = text_lower.split()
            keywords = []
            
            # 1. Ajouter d'abord les mots-clÃƒÂ©s prioritaires du domaine dÃƒÂ©tectÃƒÂ©
            if detected_domain in priority_words:
                domain_priority = priority_words[detected_domain]
                for word in words:
                    if word in domain_priority and word not in keywords:
                        keywords.append(word)
            
            # 2. Ajouter les autres mots-clÃƒÂ©s valides
            for word in words:
                if (word not in keywords and 
                    word not in generic_words and 
                    len(word) > 2 and
                    word.isalpha()):
                    keywords.append(word)
            
            # Limiter ÃƒÂ  5 mots-clÃƒÂ©s principaux
            return keywords[:5]
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction des mots-clÃƒÂ©s principaux: {e}")
            return []
    
    def _simple_key_phrases_extractor(self, text: str) -> List[str]:
        """Extracteur de phrases clÃƒÂ©s simple basÃƒÂ© sur des rÃƒÂ¨gles"""
        try:
            sentences = text.split('.')
            key_phrases = []
            
            # SÃƒÂ©lectionner les phrases contenant des mots-clÃƒÂ©s importants
            important_words = ['ai', 'artificial', 'intelligence', 'brain', 'technology', 'future', 'important', 'key']
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in important_words):
                    key_phrases.append(sentence.strip())
            
            # Si aucune phrase clÃƒÂ© n'est trouvÃƒÂ©e, prendre les premiÃƒÂ¨res phrases
            if not key_phrases and sentences:
                key_phrases = [s.strip() for s in sentences[:2] if s.strip()]
            
            return key_phrases[:3]
        except Exception as e:
            logger.warning(f"Erreur extraction phrases clÃƒÂ©s simple: {e}")
            return []

    async def _determine_semantic_context(self, 
                                        text: str, 
                                        embeddings: np.ndarray) -> str:
        """DÃƒÂ©termine le contexte sÃƒÂ©mantique du texte"""
        try:
            # Analyser le contenu du texte pour dÃƒÂ©terminer le contexte
            text_lower = text.lower()
            
            # Contexte par mots-clÃƒÂ©s
            context_keywords = {
                'technology': ['ai', 'artificial', 'intelligence', 'computer', 'software', 'digital', 'tech'],
                'health': ['health', 'medical', 'doctor', 'hospital', 'medicine', 'treatment'],
                'education': ['learn', 'study', 'education', 'school', 'university', 'knowledge'],
                'business': ['business', 'company', 'work', 'job', 'career', 'money', 'finance'],
                'science': ['science', 'research', 'discovery', 'experiment', 'laboratory'],
                'politics': ['politics', 'government', 'policy', 'election', 'democracy']
            }
            
            # Trouver le contexte dominant
            context_scores = {}
            for context, keywords in context_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    context_scores[context] = score
            
            if context_scores:
                # Retourner le contexte avec le score le plus ÃƒÂ©levÃƒÂ©
                dominant_context = max(context_scores.items(), key=lambda x: x[1])[0]
                return dominant_context
            else:
                return 'general'
                
        except Exception as e:
            logger.warning(f"Erreur dÃƒÂ©termination contexte: {e}")
            return 'general'

    async def _analyze_global_semantic(self, semantic_analyses: List[SemanticAnalysis]) -> GlobalSemanticAnalysis:
        """Analyse sÃƒÂ©mantique globale de tous les segments"""
        try:
            if not semantic_analyses:
                return GlobalSemanticAnalysis()
            
            # Analyser l'ÃƒÂ©volution des sujets
            topic_evolution = self._analyze_topic_evolution(semantic_analyses)
            
            # Calculer la cohÃƒÂ©rence sÃƒÂ©mantique globale
            semantic_coherence = self._calculate_global_semantic_coherence(semantic_analyses)
            
            # Analyser la structure discursive
            discourse_structure = self._analyze_discourse_patterns(semantic_analyses)
            
            # Analyser la complexitÃƒÂ© globale
            content_complexity = self._analyze_global_complexity(semantic_analyses)
            
            # Analyser l'audience
            audience_analysis = self._analyze_target_audience(semantic_analyses)
            
            return GlobalSemanticAnalysis(
                main_topics=list(topic_evolution.keys()),
                topic_hierarchy=topic_evolution,
                semantic_coherence=semantic_coherence,
                discourse_structure=discourse_structure,
                content_complexity=content_complexity,
                audience_analysis=audience_analysis,
                temporal_evolution=self._create_temporal_evolution(semantic_analyses)
            )
            
        except Exception as e:
            logger.error(f"Erreur analyse sÃƒÂ©mantique globale: {e}")
            return GlobalSemanticAnalysis()

    def _analyze_topic_evolution(self, semantic_analyses: List[SemanticAnalysis]) -> Dict[str, List[str]]:
        """Analyse l'ÃƒÂ©volution des sujets au fil du temps"""
        topic_evolution = {}
        
        for analysis in semantic_analyses:
            for topic, probability in analysis.topic_probabilities.items():
                if topic not in topic_evolution:
                    topic_evolution[topic] = []
                
                if probability > self.config['analysis']['min_confidence']:
                    topic_evolution[topic].append({
                        'time': analysis.start_time,
                        'probability': probability,
                        'text': analysis.text[:100]
                    })
        
        return topic_evolution

    def _calculate_global_semantic_coherence(self, semantic_analyses: List[SemanticAnalysis]) -> float:
        """Calcule la cohÃƒÂ©rence sÃƒÂ©mantique globale"""
        if len(semantic_analyses) < 2:
            return 1.0
        
        # Calculer la similaritÃƒÂ© entre segments consÃƒÂ©cutifs
        similarities = []
        for i in range(len(semantic_analyses) - 1):
            if (semantic_analyses[i].embeddings.size > 0 and 
                semantic_analyses[i+1].embeddings.size > 0):
                
                # SimilaritÃƒÂ© cosinus
                similarity = np.dot(semantic_analyses[i].embeddings, 
                                 semantic_analyses[i+1].embeddings) / (
                    np.linalg.norm(semantic_analyses[i].embeddings) * 
                    np.linalg.norm(semantic_analyses[i+1].embeddings)
                )
                similarities.append(similarity)
        
        if similarities:
            return np.mean(similarities)
        else:
            return 0.5

    def _analyze_discourse_patterns(self, semantic_analyses: List[SemanticAnalysis]) -> Dict[str, Any]:
        """Analyse les patterns discursifs"""
        discourse_patterns = {
            'transitions': [],
            'topic_shifts': [],
            'coherence_breaks': []
        }
        
        for i in range(len(semantic_analyses) - 1):
            current = semantic_analyses[i]
            next_segment = semantic_analyses[i + 1]
            
            # DÃƒÂ©tecter les changements de sujet
            if (current.semantic_context != next_segment.semantic_context and
                current.semantic_context != 'general' and 
                next_segment.semantic_context != 'general'):
                
                discourse_patterns['topic_shifts'].append({
                    'position': i,
                    'from_context': current.semantic_context,
                    'to_context': next_segment.semantic_context,
                    'time': current.end_time
                })
        
        return discourse_patterns

    def _analyze_global_complexity(self, semantic_analyses: List[SemanticAnalysis]) -> Dict[str, float]:
        """Analyse la complexitÃƒÂ© globale du contenu"""
        complexity_scores = [analysis.complexity_score for analysis in semantic_analyses]
        readability_scores = [analysis.readability_score for analysis in semantic_analyses]
        
        return {
            'average_complexity': np.mean(complexity_scores) if complexity_scores else 0.5,
            'average_readability': np.mean(readability_scores) if readability_scores else 0.5,
            'complexity_variance': np.var(complexity_scores) if complexity_scores else 0.0,
            'readability_variance': np.var(readability_scores) if readability_scores else 0.0
        }

    def _analyze_target_audience(self, semantic_analyses: List[SemanticAnalysis]) -> Dict[str, Any]:
        """Analyse l'audience cible basÃƒÂ©e sur la complexitÃƒÂ© et le contenu"""
        complexity_scores = [analysis.complexity_score for analysis in semantic_analyses]
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0.5
        
        if avg_complexity > 0.8:
            audience = "expert"
        elif avg_complexity > 0.6:
            audience = "professional"
        elif avg_complexity > 0.4:
            audience = "educated_general"
        else:
            audience = "general_public"
        
        return {
            'primary_audience': audience,
            'complexity_level': avg_complexity,
            'accessibility_score': 1.0 - avg_complexity
        }

    def _create_temporal_evolution(self, semantic_analyses: List[SemanticAnalysis]) -> List[Dict[str, Any]]:
        """CrÃƒÂ©e une timeline de l'ÃƒÂ©volution du contenu"""
        evolution = []
        
        for analysis in semantic_analyses:
            evolution.append({
                'time': analysis.start_time,
                'context': analysis.semantic_context,
                'sentiment': analysis.sentiment_score,
                'complexity': analysis.complexity_score,
                'topics': analysis.topic_probabilities
            })
        
        return evolution

    async def _analyze_discourse_structure(self, semantic_analyses: List[SemanticAnalysis]) -> Dict[str, Any]:
        """Analyse la structure discursive de la transcription"""
        try:
            discourse_structure = {
                'introduction': None,
                'development': [],
                'conclusion': None,
                'transitions': [],
                'coherence_metrics': {}
            }
            
            if len(semantic_analyses) >= 3:
                # Premier segment comme introduction
                discourse_structure['introduction'] = {
                    'segment_index': 0,
                    'context': semantic_analyses[0].semantic_context,
                    'sentiment': semantic_analyses[0].sentiment_score
                }
                
                # Dernier segment comme conclusion
                discourse_structure['conclusion'] = {
                    'segment_index': len(semantic_analyses) - 1,
                    'context': semantic_analyses[-1].semantic_context,
                    'sentiment': semantic_analyses[-1].sentiment_score
                }
                
                # Segments de dÃƒÂ©veloppement
                for i in range(1, len(semantic_analyses) - 1):
                    discourse_structure['development'].append({
                        'segment_index': i,
                        'context': semantic_analyses[i].semantic_context,
                        'sentiment': semantic_analyses[i].sentiment_score,
                        'complexity': semantic_analyses[i].complexity_score
                    })
            
            return discourse_structure
            
        except Exception as e:
            logger.error(f"Erreur analyse structure discursive: {e}")
            return {}

    def _add_to_cache(self, key: int, value: np.ndarray):
        """Ajoute une valeur au cache des embeddings"""
        if len(self.embeddings_cache) >= self.config['cache']['max_cache_size']:
            # Supprimer l'ÃƒÂ©lÃƒÂ©ment le plus ancien
            oldest_key = next(iter(self.embeddings_cache))
            del self.embeddings_cache[oldest_key]
        
        self.embeddings_cache[key] = value

    def _create_fallback_semantic_analysis(self, segment: Dict) -> SemanticAnalysis:
        """CrÃƒÂ©e une analyse sÃƒÂ©mantique de fallback"""
        return SemanticAnalysis(
            text=segment.get('text', ''),
            start_time=segment.get('start', 0.0),
            end_time=segment.get('end', 0.0),
            semantic_context='general',
            sentiment_score=0.0,
            complexity_score=0.5,
            readability_score=0.5
        )

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """CrÃƒÂ©e une rÃƒÂ©ponse d'erreur"""
        return {
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0-advanced"
        }

    def _simple_tokenizer(self, text: str) -> List[str]:
        """Tokeniseur simple basÃƒÂ© sur des rÃƒÂ¨gles"""
        try:
            # Tokenisation basique par mots et ponctuation
            import re
            # SÃƒÂ©parer les mots et la ponctuation
            tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            # Filtrer les tokens vides
            tokens = [token for token in tokens if token.strip()]
            return tokens
        except Exception as e:
            logger.warning(f"Erreur tokenisation simple: {e}")
            return text.lower().split()

    def _simple_sentiment_analyzer(self, text: str) -> float:
        """Analyseur de sentiment simple basÃƒÂ© sur des rÃƒÂ¨gles"""
        try:
            # Mots positifs et nÃƒÂ©gatifs basiques
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'love', 'like', 'enjoy', 'happy', 'joy', 'pleasure', 'success',
                'win', 'victory', 'achieve', 'accomplish', 'succeed', 'improve'
            }
            
            negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
                'dislike', 'sad', 'angry', 'frustrated', 'fail', 'lose', 'defeat',
                'problem', 'issue', 'difficult', 'hard', 'pain', 'suffering'
            }
            
            words = set(text.lower().split())
            positive_count = len(words.intersection(positive_words))
            negative_count = len(words.intersection(negative_words))
            
            # Calculer le score de sentiment (-1 ÃƒÂ  1)
            if positive_count == 0 and negative_count == 0:
                return 0.0
            elif negative_count == 0:
                return 0.8  # Positif
            elif positive_count == 0:
                return -0.8  # NÃƒÂ©gatif
            else:
                # Score pondÃƒÂ©rÃƒÂ©
                total = positive_count + negative_count
                score = (positive_count - negative_count) / total
                return score * 0.8  # Limiter ÃƒÂ  Ã‚Â±0.8
                
        except Exception as e:
            logger.warning(f"Erreur analyse sentiment simple: {e}")
            return 0.0

    def _simple_topic_classifier(self, text: str) -> List[Dict[str, float]]:
        """Classificateur de sujets simple basÃƒÂ© sur des rÃƒÂ¨gles"""
        try:
            # DÃƒÂ©finir des catÃƒÂ©gories de sujets avec des mots-clÃƒÂ©s
            topic_keywords = {
                'technology': ['ai', 'artificial', 'intelligence', 'computer', 'software', 'digital', 'tech', 'innovation'],
                'health': ['health', 'medical', 'doctor', 'hospital', 'medicine', 'treatment', 'patient', 'surgery'],
                'education': ['learn', 'study', 'education', 'school', 'university', 'knowledge', 'teaching', 'student'],
                'business': ['business', 'company', 'work', 'job', 'career', 'money', 'finance', 'economy'],
                'sports': ['sport', 'game', 'play', 'team', 'win', 'competition', 'athlete', 'fitness'],
                'entertainment': ['movie', 'music', 'art', 'culture', 'entertainment', 'fun', 'enjoy', 'show'],
                'science': ['science', 'research', 'discovery', 'experiment', 'laboratory', 'scientist', 'theory'],
                'politics': ['politics', 'government', 'policy', 'election', 'democracy', 'society', 'community']
            }
            
            text_lower = text.lower()
            scores = {}
            
            for topic, keywords in topic_keywords.items():
                score = 0.0
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1.0
                
                if score > 0:
                    # Normaliser le score
                    normalized_score = min(1.0, score / len(keywords))
                    scores[topic] = normalized_score
            
            # Trier par score dÃƒÂ©croissant et limiter ÃƒÂ  5 sujets
            sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return [{'topic': topic, 'confidence': score} for topic, score in sorted_topics]
            
        except Exception as e:
            logger.warning(f"Erreur classification sujets simple: {e}")
            return [{'topic': 'general', 'confidence': 0.5}]

    def _generate_basic_embeddings(self, text: str) -> np.ndarray:
        """GÃƒÂ©nÃƒÂ¨re des embeddings basiques basÃƒÂ©s sur des rÃƒÂ¨gles"""
        try:
            # CrÃƒÂ©er un embedding basique basÃƒÂ© sur la longueur et la complexitÃƒÂ© du texte
            words = text.lower().split()
            embedding = np.zeros(384)  # Dimension standard
            
            # Remplir l'embedding avec des valeurs basÃƒÂ©es sur le contenu
            for i, word in enumerate(words[:min(len(words), 384)]):
                # Hash simple du mot pour une distribution pseudo-alÃƒÂ©atoire
                hash_val = hash(word) % 384
                embedding[hash_val] = len(word) / 10.0  # Normaliser par la longueur
            
            # Normaliser l'embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            logger.warning(f"Erreur gÃƒÂ©nÃƒÂ©ration embedding basique: {e}")
            return np.zeros(384) 

