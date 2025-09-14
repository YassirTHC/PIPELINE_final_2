# 🚀 PIPELINE INTELLIGENT AVEC LLM DIRECT
# Intégration des modules LLM pour une vraie compréhension contextuelle

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
    """Pipeline intelligent utilisant directement le LLM pour une vraie compréhension"""
    
    def __init__(self, model: str = "gemma3:4b"):
        self.model = model
        
        # 🧠 Initialisation des composants LLM
        self.broll_generator = create_llm_broll_generator(model)
        self.metadata_generator = create_llm_metadata_generator(model)
        
        # Configuration intelligente
        self.config = {
            'enable_llm_analysis': True,
            'enable_fallback': True,
            'max_processing_time': 60,  # secondes
            'confidence_threshold': 0.7
        }
        
        logger.info(f"🚀 Pipeline intelligent LLM initialisé - Modèle: {model}")
        print(f"🧠 [PIPELINE INTELLIGENT] Initialisé avec {model}")
    
    def process_video_intelligent(self, transcript: str, video_id: str = None) -> Dict[str, Any]:
        """Traitement intelligent d'une vidéo avec le LLM"""
        
        if not video_id:
            video_id = f"video_{int(time.time())}"
        
        print(f"\n🧠 [PIPELINE INTELLIGENT] Traitement de {video_id}")
        print(f"📝 Transcript: {len(transcript)} caractères")
        print(f"🎯 Modèle LLM: {self.model}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 🎯 ÉTAPE 1: Génération B-roll intelligente avec LLM
            print("🎬 [ÉTAPE 1] Génération B-roll intelligente...")
            broll_result = self.broll_generator.generate_broll_keywords(transcript)
            
            if not broll_result.get('success', False):
                print(f"❌ [B-ROLL] Échec de la génération")
                return self._create_error_response("Échec génération B-roll", video_id)
            
            broll_keywords = broll_result.get('keywords', [])
            detected_domain = broll_result.get('domain', 'unknown')
            broll_time = broll_result.get('processing_time', 0)
            
            print(f"✅ [B-ROLL] {len(broll_keywords)} mots-clés générés")
            print(f"🌍 Domaine détecté: {detected_domain}")
            print(f"⏱️ Temps: {broll_time:.1f}s")
            
            # 📝 ÉTAPE 2: Génération métadonnées virales avec LLM
            print("\n📝 [ÉTAPE 2] Génération métadonnées virales...")
            metadata_result = self.metadata_generator.generate_viral_metadata(transcript)
            
            if not metadata_result.get('success', False):
                print(f"❌ [MÉTADONNÉES] Échec de la génération")
                return self._create_error_response("Échec génération métadonnées", video_id)
            
            title = metadata_result.get('title', '')
            description = metadata_result.get('description', '')
            hashtags = metadata_result.get('hashtags', [])
            metadata_time = metadata_result.get('processing_time', 0)
            
            print(f"✅ [MÉTADONNÉES] Titre: {title}")
            print(f"📖 Description: {description[:50]}...")
            print(f"#️⃣ Hashtags: {len(hashtags)} générés")
            print(f"⏱️ Temps: {metadata_time:.1f}s")
            
            # 📊 ÉTAPE 3: Analyse de l'intelligence
            print("\n📊 [ÉTAPE 3] Analyse de l'intelligence...")
            intelligence_score = self._analyze_intelligence(
                broll_keywords, detected_domain, title, hashtags
            )
            
            total_time = time.time() - start_time
            
            # 🎉 RÉSULTAT FINAL
            result = {
                'success': True,
                'video_id': video_id,
                'processing_time': total_time,
                'intelligence_score': intelligence_score,
                'llm_model': self.model,
                
                # Données B-roll intelligentes
                'broll_data': {
                    'keywords': broll_keywords,
                    'domain': detected_domain,
                    'generation_time': broll_time,
                    'quality_score': self._assess_broll_quality(broll_keywords)
                },
                
                # Métadonnées virales
                'metadata': {
                    'title': title,
                    'description': description,
                    'hashtags': hashtags,
                    'generation_time': metadata_time,
                    'viral_score': self._assess_viral_potential(title, description, hashtags)
                },
                
                # Métriques d'intelligence
                'intelligence_metrics': {
                    'domain_detection': detected_domain != 'fallback',
                    'keyword_specificity': self._assess_keyword_specificity(broll_keywords),
                    'title_engagement': self._assess_title_engagement(title),
                    'hashtag_relevance': self._assess_hashtag_relevance(hashtags, transcript)
                }
            }
            
            print(f"\n🎉 [PIPELINE INTELLIGENT] Traitement terminé avec succès!")
            print(f"📊 Score d'intelligence: {intelligence_score:.1f}%")
            print(f"⏱️ Temps total: {total_time:.1f}s")
            print(f"🎯 Domaine: {detected_domain}")
            print(f"🔑 Mots-clés B-roll: {len(broll_keywords)}")
            print(f"📝 Titre viral: {title}")
            print(f"#️⃣ Hashtags: {len(hashtags)}")
            
            return result
            
        except Exception as e:
            error_msg = f"Erreur dans le pipeline intelligent: {str(e)}"
            print(f"❌ [PIPELINE] {error_msg}")
            logger.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg, video_id)
    
    def _analyze_intelligence(self, broll_keywords: List[str], domain: str, title: str, hashtags: List[str]) -> float:
        """Analyse le score d'intelligence du traitement"""
        
        scores = []
        
        # Score de détection de domaine
        if domain != 'fallback' and domain != 'unknown':
            scores.append(100)  # Domaine détecté avec succès
        else:
            scores.append(0)    # Fallback utilisé
        
        # Score de qualité des mots-clés B-roll
        if len(broll_keywords) >= 8:
            # Vérifier la spécificité (éviter les mots génériques)
            generic_words = ['content', 'media', 'engaging', 'professional', 'interesting']
            specific_count = sum(1 for kw in broll_keywords if not any(gw in kw.lower() for gw in generic_words))
            specificity_score = (specific_count / len(broll_keywords)) * 100
            scores.append(specificity_score)
        else:
            scores.append(0)
        
        # Score du titre viral
        if title and len(title) <= 60:
            # Vérifier la présence d'emojis viraux
            viral_emojis = ['🔥', '💡', '🚀', '💪', '🎯', '😱', '🤯']
            emoji_score = 100 if any(emoji in title for emoji in viral_emojis) else 50
            scores.append(emoji_score)
        else:
            scores.append(0)
        
        # Score des hashtags
        if len(hashtags) >= 10:
            # Vérifier la diversité des hashtags
            hashtag_score = min(100, len(hashtags) * 10)
            scores.append(hashtag_score)
        else:
            scores.append(0)
        
        # Score moyen
        return sum(scores) / len(scores) if scores else 0
    
    def _assess_broll_quality(self, keywords: List[str]) -> float:
        """Évalue la qualité des mots-clés B-roll"""
        
        if not keywords:
            return 0.0
        
        # Critères de qualité
        scores = []
        
        # Longueur appropriée
        for kw in keywords:
            if 2 <= len(kw.split()) <= 5:  # 2-5 mots par phrase
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        # Éviter les mots génériques
        generic_words = ['content', 'media', 'engaging', 'professional', 'interesting']
        for kw in keywords:
            if not any(gw in kw.lower() for gw in generic_words):
                scores.append(1.0)
            else:
                scores.append(0.3)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _assess_viral_potential(self, title: str, description: str, hashtags: List[str]) -> float:
        """Évalue le potentiel viral du contenu"""
        
        scores = []
        
        # Titre viral
        if title:
            viral_emojis = ['🔥', '💡', '🚀', '💪', '🎯', '😱', '🤯']
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
        
        # Hashtags appropriés
        if len(hashtags) >= 10:
            scores.append(1.0)
        elif len(hashtags) >= 5:
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _assess_keyword_specificity(self, keywords: List[str]) -> float:
        """Évalue la spécificité des mots-clés"""
        
        if not keywords:
            return 0.0
        
        generic_words = ['content', 'media', 'engaging', 'professional', 'interesting', 'motivational']
        specific_count = 0
        
        for kw in keywords:
            if not any(gw in kw.lower() for gw in generic_words):
                specific_count += 1
        
        return (specific_count / len(keywords)) * 100
    
    def _assess_title_engagement(self, title: str) -> float:
        """Évalue l'engagement du titre"""
        
        if not title:
            return 0.0
        
        scores = []
        
        # Longueur appropriée
        if len(title) <= 60:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Présence d'emojis viraux
        viral_emojis = ['🔥', '💡', '🚀', '💪', '🎯', '😱', '🤯']
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
        """Évalue la pertinence des hashtags par rapport au transcript"""
        
        if not hashtags or not transcript:
            return 0.0
        
        # Extraire les mots-clés du transcript
        transcript_words = set(transcript.lower().split())
        
        # Compter les hashtags pertinents
        relevant_count = 0
        for hashtag in hashtags:
            # Enlever le # et vérifier la pertinence
            tag = hashtag.replace('#', '').lower()
            if tag in transcript_words or any(word in transcript.lower() for word in tag.split('_')):
                relevant_count += 1
        
        return (relevant_count / len(hashtags)) * 100
    
    def _create_error_response(self, error_message: str, video_id: str) -> Dict[str, Any]:
        """Crée une réponse d'erreur structurée"""
        
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
    """Factory pour créer un pipeline intelligent LLM"""
    return LLMIntelligentPipeline(model=model) 