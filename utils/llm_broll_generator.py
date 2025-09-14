# 🚀 GÉNÉRATEUR B-ROLL INTELLIGENT AVEC LLM DIRECT
# Utilise directement le LLM pour une vraie compréhension contextuelle

import json
import logging
import time
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class LLMBrollGenerator:
    """Générateur B-roll intelligent utilisant directement le LLM local"""
    
    def __init__(self, model: str = "gemma3:4b", timeout: int = 120):
        self.model = model
        self.timeout = timeout
        self.api_url = "http://localhost:11434/api/generate"
        
        # 🧠 PROMPT SYSTÈME OPTIMISÉ pour B-roll parfait
        self.system_prompt = """Generate 8-12 B-roll search keywords optimized for stock footage platforms.

CRITICAL REQUIREMENTS:
- Each keyword/phrase must be 2-5 words maximum
- Focus on VISUAL elements that can be filmed/photographed
- Optimize for Pexels, Pixabay, Storyblocks search algorithms
- Cover: People, Actions, Objects, Environments, Concepts
- Avoid generic words like "content", "media", "interesting"

EXAMPLES OF PERFECT B-roll keywords:
- Science: "scientist examining test tubes", "brain scan MRI", "research lab equipment"
- Sports: "athlete running outdoors", "team huddle celebration", "coach giving instructions"
- Business: "person typing on laptop", "meeting room discussion", "professional handshake"
- Lifestyle: "person cooking in kitchen", "sunset over mountains", "friends laughing together"

OUTPUT: JSON only with exact format:
{"keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6", "keyword7", "keyword8"]}

Transcript:"""
        
        # 🚀 PROMPT CONTEXTUEL INTELLIGENT pour modèles 4B
        self.fast_prompt = """Analyze this transcript and generate B-roll keywords that VISUALLY REPRESENT the specific content.

REQUIREMENTS:
- ONLY keywords/phrases that DIRECTLY relate to what's being discussed
- NO generic concepts, NO unrelated visuals
- Focus on: people, actions, objects, environments mentioned in the transcript
- Each keyword must be searchable on stock footage platforms

EXAMPLES:
- Transcript: "brain focus, sleep deprivation" → ["person sleeping", "brain scan", "tired person"]
- Transcript: "panoramic vision, space time" → ["wide landscape view", "time concept", "spatial awareness"]

OUTPUT: {"keywords": ["keyword1", "keyword2", "keyword3"]}

Transcript:"""

    def generate_broll_keywords(self, transcript: str) -> Dict[str, Any]:
        """Génère des mots-clés B-roll intelligents avec le LLM"""
        
        try:
            # 🧠 Choix du prompt selon le modèle
            if self.model in ["gemma3:4b", "qwen3:4b"]:
                # Modèles 4B = prompt ultra-optimisé
                full_prompt = self.fast_prompt + transcript
                print(f"🧠 [LLM] Prompt ULTRA-OPTIMISÉ pour {self.model}")
            else:
                # Modèles plus puissants = prompt complet optimisé
                full_prompt = self.system_prompt + transcript
                print(f"🧠 [LLM] Prompt COMPLET OPTIMISÉ pour {self.model}")
            
            print(f"🧠 [LLM] Génération B-roll intelligente pour {len(transcript)} caractères")
            print(f"🎯 Modèle: {self.model}")
            print(f"📝 Taille prompt: {len(full_prompt)} caractères")
            
            start_time = time.time()
            
            # 🚀 Appel direct au LLM avec timeout adaptatif
            timeout = min(self.timeout, 60 if self.model in ["gemma3:4b", "qwen3:4b"] else 120)
            
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '').strip()
                
                print(f"✅ [LLM] Réponse reçue en {duration:.1f}s")
                print(f"📝 Taille réponse: {len(llm_response)} caractères")
                
                # 🔍 Extraction et validation JSON
                try:
                    # Nettoyer la réponse (enlever markdown, etc.)
                    cleaned_response = self._clean_llm_response(llm_response)
                    
                    # Parser le JSON
                    parsed_data = json.loads(cleaned_response)
                    
                    if 'keywords' in parsed_data and isinstance(parsed_data['keywords'], list):
                        keywords = parsed_data['keywords']
                        
                        # Validation des mots-clés
                        validated_keywords = self._validate_keywords(keywords)
                        
                        print(f"🎯 [LLM] {len(validated_keywords)} mots-clés B-roll générés")
                        print(f"🔍 Exemples: {', '.join(validated_keywords[:3])}...")
                        
                        return {
                            'success': True,
                            'keywords': validated_keywords,
                            'domain': self._detect_domain_from_keywords(validated_keywords),
                            'processing_time': duration,
                            'model_used': self.model
                        }
                    else:
                        raise ValueError("Format JSON invalide: 'keywords' manquant")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ [LLM] Erreur parsing JSON: {e}")
                    print(f"📝 Réponse brute: {llm_response[:200]}...")
                    return self._fallback_generation(transcript, f"Erreur JSON: {e}")
                    
            else:
                print(f"❌ [LLM] Erreur HTTP: {response.status_code}")
                return self._fallback_generation(transcript, f"Erreur HTTP: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"⏱️ [LLM] Timeout après {timeout}s")
            return self._fallback_generation(transcript, f"Timeout LLM ({timeout}s)")
            
        except Exception as e:
            print(f"❌ [LLM] Erreur générale: {e}")
            return self._fallback_generation(transcript, f"Erreur: {e}")
    
    def _clean_llm_response(self, response: str) -> str:
        """Nettoie la réponse du LLM pour extraire le JSON"""
        
        # Chercher le JSON dans la réponse
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_part = response[start_idx:end_idx + 1]
            return json_part
        
        # Si pas de JSON trouvé, essayer de nettoyer
        cleaned = response.replace('```json', '').replace('```', '').strip()
        return cleaned
    
    def _validate_keywords(self, keywords: List[str]) -> List[str]:
        """Valide et nettoie les mots-clés générés"""
        
        validated = []
        
        for keyword in keywords:
            if isinstance(keyword, str) and keyword.strip():
                # Nettoyer le mot-clé
                clean_keyword = keyword.strip()
                
                # Éviter les mots-clés trop génériques
                generic_words = ['content', 'media', 'engaging', 'professional', 'interesting', 'video', 'footage']
                if clean_keyword.lower() not in generic_words:
                    # Vérifier la longueur (2-5 mots)
                    word_count = len(clean_keyword.split())
                    if 2 <= word_count <= 5:
                        validated.append(clean_keyword)
        
        # Garantir au moins 8 mots-clés
        if len(validated) < 8:
            # Ajouter des mots-clés de fallback intelligents
            fallback_keywords = ['person working', 'professional environment', 'modern technology', 'natural landscape', 'urban setting', 'creative process', 'daily activity', 'social interaction']
            for i in range(8 - len(validated)):
                if fallback_keywords[i] not in validated:
                    validated.append(fallback_keywords[i])
        
        # Limiter à 12 maximum
        return validated[:12]
    
    def _detect_domain_from_keywords(self, keywords: List[str]) -> str:
        """Détecte le domaine basé sur les mots-clés générés"""
        
        # Analyse simple basée sur les mots-clés
        science_words = ['scientist', 'research', 'lab', 'experiment', 'test', 'analysis', 'microscope', 'test tubes']
        sport_words = ['athlete', 'training', 'competition', 'game', 'sport', 'coach', 'team', 'running']
        business_words = ['meeting', 'office', 'business', 'professional', 'corporate', 'handshake', 'presentation']
        tech_words = ['computer', 'technology', 'digital', 'screen', 'device', 'coding', 'software']
        medical_words = ['doctor', 'patient', 'medical', 'therapy', 'treatment', 'hospital', 'clinic']
        
        keyword_text = ' '.join(keywords).lower()
        
        if any(word in keyword_text for word in medical_words):
            return 'medical'
        elif any(word in keyword_text for word in science_words):
            return 'science'
        elif any(word in keyword_text for word in sport_words):
            return 'sport'
        elif any(word in keyword_text for word in business_words):
            return 'business'
        elif any(word in keyword_text for word in tech_words):
            return 'technology'
        else:
            return 'lifestyle'  # Domaine par défaut plus spécifique que "general"
    
    def _fallback_generation(self, transcript: str, error_reason: str) -> Dict[str, Any]:
        """Génération de fallback intelligente basée sur le transcript"""
        
        print(f"🔄 [FALLBACK] Génération intelligente de fallback: {error_reason}")
        
        # Analyse simple du transcript pour extraire des mots-clés
        words = transcript.lower().split()
        
        # Filtrer les mots pertinents
        relevant_words = []
        for word in words:
            if len(word) > 3 and word.isalpha():
                # Éviter les mots trop communs
                common_words = ['the', 'and', 'that', 'this', 'with', 'from', 'they', 'have', 'been', 'will', 'would', 'could', 'should']
                if word not in common_words:
                    relevant_words.append(word)
        
        # Prendre les mots les plus fréquents
        from collections import Counter
        word_counts = Counter(relevant_words)
        top_words = [word for word, _ in word_counts.most_common(10)]
        
        # Transformer en mots-clés B-roll optimisés
        broll_keywords = []
        for word in top_words[:8]:
            # Ajouter du contexte pour rendre plus visuel et optimisé
            if word in ['person', 'people', 'man', 'woman']:
                broll_keywords.append(f"person {word}ing")
            elif word in ['work', 'study', 'research']:
                broll_keywords.append(f"person {word}ing")
            elif word in ['therapy', 'treatment']:
                broll_keywords.append(f"therapy session")
            elif word in ['brain', 'mind']:
                broll_keywords.append(f"brain activity")
            else:
                broll_keywords.append(word)
        
        print(f"🔄 [FALLBACK] {len(broll_keywords)} mots-clés générés par fallback")
        
        return {
            'success': True,
            'keywords': broll_keywords,
            'domain': 'fallback',
            'processing_time': 0.1,
            'model_used': 'fallback_system',
            'fallback_reason': error_reason
        }

def create_llm_broll_generator(model: str = "gemma3:4b") -> LLMBrollGenerator:
    """Factory pour créer un générateur B-roll LLM"""
    return LLMBrollGenerator(model=model) 