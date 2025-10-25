ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ PIPELINE INTELLIGENT AVEC LLM DIRECT
# IntÃƒÂ©gration des modules LLM pour une vraie comprÃƒÂ©hension contextuelle

import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Import des modules LLM intelligents
from .llm_broll_generator import create_llm_broll_generator
from .llm_metadata_generator import create_llm_metadata_generator

logger = logging.getLogger(__name__)

class LLMIntelligentPipeline:
    """Pipeline intelligent utilisant directement le LLM pour une vraie comprÃƒÂ©hension"""
    
    def __init__(self, model: str = "gemma3:4b"):
        self.model = model
        
        # Ã°Å¸Â§Â  Initialisation des composants LLM
        self.broll_generator = create_llm_broll_generator(model)
        self.metadata_generator = create_llm_metadata_generator(model)
        
        # Configuration intelligente
        self.config = {
            'enable_llm_analysis': True,
            'enable_fallback': True,
            'max_processing_time': 60,  # secondes
            'confidence_threshold': 0.7
        }
        
        logger.info(f"Ã°Å¸Å¡â‚¬ Pipeline intelligent LLM initialisÃƒÂ© - ModÃƒÂ¨le: {model}")
        print(f"Ã°Å¸Â§Â  [PIPELINE INTELLIGENT] InitialisÃƒÂ© avec {model}")
    
    def process_video_intelligent(self, transcript: str, video_id: str = None) -> Dict[str, Any]:
        """Traitement intelligent d'une vidÃƒÂ©o avec le LLM"""
        
        if not video_id:
            video_id = f"video_{int(time.time())}"
        
        print(f"\nÃ°Å¸Â§Â  [PIPELINE INTELLIGENT] Traitement de {video_id}")
        print(f"Ã°Å¸â€œÂ Transcript: {len(transcript)} caractÃƒÂ¨res")
        print(f"Ã°Å¸Å½Â¯ ModÃƒÂ¨le LLM: {self.model}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Ã°Å¸Å½Â¯ Ãƒâ€°TAPE 1: GÃƒÂ©nÃƒÂ©ration B-roll intelligente avec LLM
            print("Ã°Å¸Å½Â¬ [Ãƒâ€°TAPE 1] GÃƒÂ©nÃƒÂ©ration B-roll intelligente...")
            broll_result = self.broll_generator.generate_broll_keywords(transcript)
            
            if not broll_result.get('success', False):
                print(f"Ã¢ÂÅ’ [B-ROLL] Ãƒâ€°chec de la gÃƒÂ©nÃƒÂ©ration")
                return self._create_error_response("Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration B-roll", video_id)
            
            broll_keywords = broll_result.get('keywords', [])
            detected_domain = broll_result.get('domain', 'unknown')
            broll_time = broll_result.get('processing_time', 0)
            
            print(f"Ã¢Å“â€¦ [B-ROLL] {len(broll_keywords)} mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
            print(f"Ã°Å¸Å’Â Domaine dÃƒÂ©tectÃƒÂ©: {detected_domain}")
            print(f"Ã¢ÂÂ±Ã¯Â¸Â Temps: {broll_time:.1f}s")
            
            # Ã°Å¸â€œÂ Ãƒâ€°TAPE 2: GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es virales avec LLM
            print("\nÃ°Å¸â€œÂ [Ãƒâ€°TAPE 2] GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es virales...")
            metadata_result = self.metadata_generator.generate_viral_metadata(transcript)
            
            if not metadata_result.get('success', False):
                print(f"Ã¢ÂÅ’ [MÃƒâ€°TADONNÃƒâ€°ES] Ãƒâ€°chec de la gÃƒÂ©nÃƒÂ©ration")
                return self._create_error_response("Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es", video_id)
            
            title = metadata_result.get('title', '')
            description = metadata_result.get('description', '')
            hashtags = metadata_result.get('hashtags', [])
            metadata_time = metadata_result.get('processing_time', 0)
            
            print(f"Ã¢Å“â€¦ [MÃƒâ€°TADONNÃƒâ€°ES] Titre: {title}")
            print(f"Ã°Å¸â€œâ€“ Description: {description[:50]}...")
            print(f"#Ã¯Â¸ÂÃ¢Æ’Â£ Hashtags: {len(hashtags)} gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
            print(f"Ã¢ÂÂ±Ã¯Â¸Â Temps: {metadata_time:.1f}s")
            
            # Ã°Å¸â€œÅ  Ãƒâ€°TAPE 3: Analyse de l'intelligence
            print("\nÃ°Å¸â€œÅ  [Ãƒâ€°TAPE 3] Analyse de l'intelligence...")
            intelligence_score = self._analyze_intelligence(
                broll_keywords, detected_domain, title, hashtags
            )
            
            total_time = time.time() - start_time
            
            # Ã°Å¸Å½â€° RÃƒâ€°SULTAT FINAL
            result = {
                'success': True,
                'video_id': video_id,
                'processing_time': total_time,
                'intelligence_score': intelligence_score,
                'llm_model': self.model,
                
                # DonnÃƒÂ©es B-roll intelligentes
                'broll_data': {
                    'keywords': broll_keywords,
                    'domain': detected_domain,
                    'generation_time': broll_time,
                    'quality_score': self._assess_broll_quality(broll_keywords)
                },
                
                # MÃƒÂ©tadonnÃƒÂ©es virales
                'metadata': {
                    'title': title,
                    'description': description,
                    'hashtags': hashtags,
                    'generation_time': metadata_time,
                    'viral_score': self._assess_viral_potential(title, description, hashtags)
                },
                
                # MÃƒÂ©triques d'intelligence
                'intelligence_metrics': {
                    'domain_detection': detected_domain != 'fallback',
                    'keyword_specificity': self._assess_keyword_specificity(broll_keywords),
                    'title_engagement': self._assess_title_engagement(title),
                    'hashtag_relevance': self._assess_hashtag_relevance(hashtags, transcript)
                }
            }
            
            print(f"\nÃ°Å¸Å½â€° [PIPELINE INTELLIGENT] Traitement terminÃƒÂ© avec succÃƒÂ¨s!")
            print(f"Ã°Å¸â€œÅ  Score d'intelligence: {intelligence_score:.1f}%")
            print(f"Ã¢ÂÂ±Ã¯Â¸Â Temps total: {total_time:.1f}s")
            print(f"Ã°Å¸Å½Â¯ Domaine: {detected_domain}")
            print(f"Ã°Å¸â€â€˜ Mots-clÃƒÂ©s B-roll: {len(broll_keywords)}")
            print(f"Ã°Å¸â€œÂ Titre viral: {title}")
            print(f"#Ã¯Â¸ÂÃ¢Æ’Â£ Hashtags: {len(hashtags)}")
            
            return result
            
        except Exception as e:
            error_msg = f"Erreur dans le pipeline intelligent: {str(e)}"
            print(f"Ã¢ÂÅ’ [PIPELINE] {error_msg}")
            logger.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg, video_id)
    
    def _analyze_intelligence(self, broll_keywords: List[str], domain: str, title: str, hashtags: List[str]) -> float:
        """Analyse le score d'intelligence du traitement"""
        
        scores = []
        
        # Score de dÃƒÂ©tection de domaine
        if domain != 'fallback' and domain != 'unknown':
            scores.append(100)  # Domaine dÃƒÂ©tectÃƒÂ© avec succÃƒÂ¨s
        else:
            scores.append(0)    # Fallback utilisÃƒÂ©
        
        # Score de qualitÃƒÂ© des mots-clÃƒÂ©s B-roll
        if len(broll_keywords) >= 8:
            # VÃƒÂ©rifier la spÃƒÂ©cificitÃƒÂ© (ÃƒÂ©viter les mots gÃƒÂ©nÃƒÂ©riques)
            generic_words = ['content', 'media', 'engaging', 'professional', 'interesting']
            specific_count = sum(1 for kw in broll_keywords if not any(gw in kw.lower() for gw in generic_words))
            specificity_score = (specific_count / len(broll_keywords)) * 100
            scores.append(specificity_score)
        else:
            scores.append(0)
        
        # Score du titre viral
        if title and len(title) <= 60:
            # VÃƒÂ©rifier la prÃƒÂ©sence d'emojis viraux
            viral_emojis = ['Ã°Å¸â€Â¥', 'Ã°Å¸â€™Â¡', 'Ã°Å¸Å¡â‚¬', 'Ã°Å¸â€™Âª', 'Ã°Å¸Å½Â¯', 'Ã°Å¸ËœÂ±', 'Ã°Å¸Â¤Â¯']
            emoji_score = 100 if any(emoji in title for emoji in viral_emojis) else 50
            scores.append(emoji_score)
        else:
            scores.append(0)
        
        # Score des hashtags
        if len(hashtags) >= 10:
            # VÃƒÂ©rifier la diversitÃƒÂ© des hashtags
            hashtag_score = min(100, len(hashtags) * 10)
            scores.append(hashtag_score)
        else:
            scores.append(0)
        
        # Score moyen
        return sum(scores) / len(scores) if scores else 0
    
    def _assess_broll_quality(self, keywords: List[str]) -> float:
        """Ãƒâ€°value la qualitÃƒÂ© des mots-clÃƒÂ©s B-roll"""
        
        if not keywords:
            return 0.0
        
        # CritÃƒÂ¨res de qualitÃƒÂ©
        scores = []
        
        # Longueur appropriÃƒÂ©e
        for kw in keywords:
            if 2 <= len(kw.split()) <= 5:  # 2-5 mots par phrase
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        # Ãƒâ€°viter les mots gÃƒÂ©nÃƒÂ©riques
        generic_words = ['content', 'media', 'engaging', 'professional', 'interesting']
        for kw in keywords:
            if not any(gw in kw.lower() for gw in generic_words):
                scores.append(1.0)
            else:
                scores.append(0.3)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _assess_viral_potential(self, title: str, description: str, hashtags: List[str]) -> float:
        """Ãƒâ€°value le potentiel viral du contenu"""
        
        scores = []
        
        # Titre viral
        if title:
            viral_emojis = ['Ã°Å¸â€Â¥', 'Ã°Å¸â€™Â¡', 'Ã°Å¸Å¡â‚¬', 'Ã°Å¸â€™Âª', 'Ã°Å¸Å½Â¯', 'Ã°Å¸ËœÂ±', 'Ã°Å¸Â¤Â¯']
            if any(emoji in title for emoji in viral_emojis):
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        # Description engageante
        if description and len(description) <= 180:
            if any(word in description.lower() for word in ['watch', 'learn', 'try', 'discover']):
                scores.append(1.0)
            else:
                scores.append(0.7)
        
        # Hashtags appropriÃƒÂ©s
        if len(hashtags) >= 10:
            scores.append(1.0)
        elif len(hashtags) >= 5:
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _assess_keyword_specificity(self, keywords: List[str]) -> float:
        """Ãƒâ€°value la spÃƒÂ©cificitÃƒÂ© des mots-clÃƒÂ©s"""
        
        if not keywords:
            return 0.0
        
        generic_words = ['content', 'media', 'engaging', 'professional', 'interesting', 'motivational']
        specific_count = 0
        
        for kw in keywords:
            if not any(gw in kw.lower() for gw in generic_words):
                specific_count += 1
        
        return (specific_count / len(keywords)) * 100
    
    def _assess_title_engagement(self, title: str) -> float:
        """Ãƒâ€°value l'engagement du titre"""
        
        if not title:
            return 0.0
        
        scores = []
        
        # Longueur appropriÃƒÂ©e
        if len(title) <= 60:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # PrÃƒÂ©sence d'emojis viraux
        viral_emojis = ['Ã°Å¸â€Â¥', 'Ã°Å¸â€™Â¡', 'Ã°Å¸Å¡â‚¬', 'Ã°Å¸â€™Âª', 'Ã°Å¸Å½Â¯', 'Ã°Å¸ËœÂ±', 'Ã°Å¸Â¤Â¯']
        if any(emoji in title for emoji in viral_emojis):
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Mots d'action
        action_words = ['how', 'why', 'what', 'when', 'where', 'this', 'that']
        if any(word in title.lower() for word in action_words):
            scores.append(1.0)
        else:
            scores.append(0.7)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _assess_hashtag_relevance(self, hashtags: List[str], transcript: str) -> float:
        """Ãƒâ€°value la pertinence des hashtags par rapport au transcript"""
        
        if not hashtags or not transcript:
            return 0.0
        
        # Extraire les mots-clÃƒÂ©s du transcript
        transcript_words = set(transcript.lower().split())
        
        # Compter les hashtags pertinents
        relevant_count = 0
        for hashtag in hashtags:
            # Enlever le # et vÃƒÂ©rifier la pertinence
            tag = hashtag.replace('#', '').lower()
            if tag in transcript_words or any(word in transcript.lower() for word in tag.split('_')):
                relevant_count += 1
        
        return (relevant_count / len(hashtags)) * 100
    
    def _create_error_response(self, error_message: str, video_id: str) -> Dict[str, Any]:
        """CrÃƒÂ©e une rÃƒÂ©ponse d'erreur structurÃƒÂ©e"""
        
        return {
            'success': False,
            'video_id': video_id,
            'error': error_message,
            'processing_time': 0,
            'intelligence_score': 0,
            'llm_model': self.model,
            'broll_data': {'keywords': [], 'domain': 'error', 'generation_time': 0, 'quality_score': 0},
            'metadata': {'title': '', 'description': '', 'hashtags': [], 'generation_time': 0, 'viral_score': 0},
            'intelligence_metrics': {
                'domain_detection': False,
                'keyword_specificity': 0,
                'title_engagement': 0,
                'hashtag_relevance': 0
            }
        }

def create_llm_intelligent_pipeline(model: str = "gemma3:4b") -> LLMIntelligentPipeline:
    """Factory pour crÃƒÂ©er un pipeline intelligent LLM"""
    return LLMIntelligentPipeline(model=model) 

