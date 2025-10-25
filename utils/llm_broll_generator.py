ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ GÃƒâ€°NÃƒâ€°RATEUR B-ROLL INTELLIGENT AVEC LLM DIRECT
# Utilise directement le LLM pour une vraie comprÃƒÂ©hension contextuelle

import json
import logging
import time
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class LLMBrollGenerator:
    """GÃƒÂ©nÃƒÂ©rateur B-roll intelligent utilisant directement le LLM local"""
    
    def __init__(self, model: str = "gemma3:4b", timeout: int = 120):
        self.model = model
        self.timeout = timeout
        self.api_url = "http://localhost:11434/api/generate"
        
        # Ã°Å¸Â§Â  PROMPT SYSTÃƒË†ME OPTIMISÃƒâ€° pour B-roll parfait
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
        
        # Ã°Å¸Å¡â‚¬ PROMPT CONTEXTUEL INTELLIGENT pour modÃƒÂ¨les 4B
        self.fast_prompt = """Analyze this transcript and generate B-roll keywords that VISUALLY REPRESENT the specific content.

REQUIREMENTS:
- ONLY keywords/phrases that DIRECTLY relate to what's being discussed
- NO generic concepts, NO unrelated visuals
- Focus on: people, actions, objects, environments mentioned in the transcript
- Each keyword must be searchable on stock footage platforms

EXAMPLES:
- Transcript: "brain focus, sleep deprivation" Ã¢â€ â€™ ["person sleeping", "brain scan", "tired person"]
- Transcript: "panoramic vision, space time" Ã¢â€ â€™ ["wide landscape view", "time concept", "spatial awareness"]

OUTPUT: {"keywords": ["keyword1", "keyword2", "keyword3"]}

Transcript:"""

    def generate_broll_keywords(self, transcript: str) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ¨re des mots-clÃƒÂ©s B-roll intelligents avec le LLM"""
        
        try:
            # Ã°Å¸Â§Â  Choix du prompt selon le modÃƒÂ¨le
            if self.model in ["gemma3:4b", "qwen3:4b"]:
                # ModÃƒÂ¨les 4B = prompt ultra-optimisÃƒÂ©
                full_prompt = self.fast_prompt + transcript
                print(f"Ã°Å¸Â§Â  [LLM] Prompt ULTRA-OPTIMISÃƒâ€° pour {self.model}")
            else:
                # ModÃƒÂ¨les plus puissants = prompt complet optimisÃƒÂ©
                full_prompt = self.system_prompt + transcript
                print(f"Ã°Å¸Â§Â  [LLM] Prompt COMPLET OPTIMISÃƒâ€° pour {self.model}")
            
            print(f"Ã°Å¸Â§Â  [LLM] GÃƒÂ©nÃƒÂ©ration B-roll intelligente pour {len(transcript)} caractÃƒÂ¨res")
            print(f"Ã°Å¸Å½Â¯ ModÃƒÂ¨le: {self.model}")
            print(f"Ã°Å¸â€œÂ Taille prompt: {len(full_prompt)} caractÃƒÂ¨res")
            
            start_time = time.time()
            
            # Ã°Å¸Å¡â‚¬ Appel direct au LLM avec timeout adaptatif
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
                
                print(f"Ã¢Å“â€¦ [LLM] RÃƒÂ©ponse reÃƒÂ§ue en {duration:.1f}s")
                print(f"Ã°Å¸â€œÂ Taille rÃƒÂ©ponse: {len(llm_response)} caractÃƒÂ¨res")
                
                # Ã°Å¸â€Â Extraction et validation JSON
                try:
                    # Nettoyer la rÃƒÂ©ponse (enlever markdown, etc.)
                    cleaned_response = self._clean_llm_response(llm_response)
                    
                    # Parser le JSON
                    parsed_data = json.loads(cleaned_response)
                    
                    if 'keywords' in parsed_data and isinstance(parsed_data['keywords'], list):
                        keywords = parsed_data['keywords']
                        
                        # Validation des mots-clÃƒÂ©s
                        validated_keywords = self._validate_keywords(keywords)
                        
                        print(f"Ã°Å¸Å½Â¯ [LLM] {len(validated_keywords)} mots-clÃƒÂ©s B-roll gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
                        print(f"Ã°Å¸â€Â Exemples: {', '.join(validated_keywords[:3])}...")
                        
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
                    print(f"Ã¢ÂÅ’ [LLM] Erreur parsing JSON: {e}")
                    print(f"Ã°Å¸â€œÂ RÃƒÂ©ponse brute: {llm_response[:200]}...")
                    return self._fallback_generation(transcript, f"Erreur JSON: {e}")
                    
            else:
                print(f"Ã¢ÂÅ’ [LLM] Erreur HTTP: {response.status_code}")
                return self._fallback_generation(transcript, f"Erreur HTTP: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"Ã¢ÂÂ±Ã¯Â¸Â [LLM] Timeout aprÃƒÂ¨s {timeout}s")
            return self._fallback_generation(transcript, f"Timeout LLM ({timeout}s)")
            
        except Exception as e:
            print(f"Ã¢ÂÅ’ [LLM] Erreur gÃƒÂ©nÃƒÂ©rale: {e}")
            return self._fallback_generation(transcript, f"Erreur: {e}")
    
    def _clean_llm_response(self, response: str) -> str:
        """Nettoie la rÃƒÂ©ponse du LLM pour extraire le JSON"""
        
        # Chercher le JSON dans la rÃƒÂ©ponse
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_part = response[start_idx:end_idx + 1]
            return json_part
        
        # Si pas de JSON trouvÃƒÂ©, essayer de nettoyer
        cleaned = response.replace('```json', '').replace('```', '').strip()
        return cleaned
    
    def _validate_keywords(self, keywords: List[str]) -> List[str]:
        """Valide et nettoie les mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©rÃƒÂ©s"""
        
        validated = []
        
        for keyword in keywords:
            if isinstance(keyword, str) and keyword.strip():
                # Nettoyer le mot-clÃƒÂ©
                clean_keyword = keyword.strip()
                
                # Ãƒâ€°viter les mots-clÃƒÂ©s trop gÃƒÂ©nÃƒÂ©riques
                generic_words = ['content', 'media', 'engaging', 'professional', 'interesting', 'video', 'footage']
                if clean_keyword.lower() not in generic_words:
                    # VÃƒÂ©rifier la longueur (2-5 mots)
                    word_count = len(clean_keyword.split())
                    if 2 <= word_count <= 5:
                        validated.append(clean_keyword)
        
        # Garantir au moins 8 mots-clÃƒÂ©s
        if len(validated) < 8:
            # Ajouter des mots-clÃƒÂ©s de fallback intelligents
            fallback_keywords = ['person working', 'professional environment', 'modern technology', 'natural landscape', 'urban setting', 'creative process', 'daily activity', 'social interaction']
            for i in range(8 - len(validated)):
                if fallback_keywords[i] not in validated:
                    validated.append(fallback_keywords[i])
        
        # Limiter ÃƒÂ  12 maximum
        return validated[:12]
    
    def _detect_domain_from_keywords(self, keywords: List[str]) -> str:
        """DÃƒÂ©tecte le domaine basÃƒÂ© sur les mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©rÃƒÂ©s"""
        
        # Analyse simple basÃƒÂ©e sur les mots-clÃƒÂ©s
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
            return 'lifestyle'  # Domaine par dÃƒÂ©faut plus spÃƒÂ©cifique que "general"
    
    def _fallback_generation(self, transcript: str, error_reason: str) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ©ration de fallback intelligente basÃƒÂ©e sur le transcript"""
        
        print(f"Ã°Å¸â€â€ž [FALLBACK] GÃƒÂ©nÃƒÂ©ration intelligente de fallback: {error_reason}")
        
        # Analyse simple du transcript pour extraire des mots-clÃƒÂ©s
        words = transcript.lower().split()
        
        # Filtrer les mots pertinents
        relevant_words = []
        for word in words:
            if len(word) > 3 and word.isalpha():
                # Ãƒâ€°viter les mots trop communs
                common_words = ['the', 'and', 'that', 'this', 'with', 'from', 'they', 'have', 'been', 'will', 'would', 'could', 'should']
                if word not in common_words:
                    relevant_words.append(word)
        
        # Prendre les mots les plus frÃƒÂ©quents
        from collections import Counter
        word_counts = Counter(relevant_words)
        top_words = [word for word, _ in word_counts.most_common(10)]
        
        # Transformer en mots-clÃƒÂ©s B-roll optimisÃƒÂ©s
        broll_keywords = []
        for word in top_words[:8]:
            # Ajouter du contexte pour rendre plus visuel et optimisÃƒÂ©
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
        
        print(f"Ã°Å¸â€â€ž [FALLBACK] {len(broll_keywords)} mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©rÃƒÂ©s par fallback")
        
        return {
            'success': True,
            'keywords': broll_keywords,
            'domain': 'fallback',
            'processing_time': 0.1,
            'model_used': 'fallback_system',
            'fallback_reason': error_reason
        }

def create_llm_broll_generator(model: str = "gemma3:4b") -> LLMBrollGenerator:
    """Factory pour crÃƒÂ©er un gÃƒÂ©nÃƒÂ©rateur B-roll LLM"""
    return LLMBrollGenerator(model=model) 

