ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ GÃƒâ€°NÃƒâ€°RATEUR DE MÃƒâ€°TADONNÃƒâ€°ES VIRALES AVEC LLM DIRECT
# Titres, descriptions et hashtags TikTok/Instagram optimisÃƒÂ©s

import json
import logging
import time
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class LLMMetadataGenerator:
    """GÃƒÂ©nÃƒÂ©rateur de mÃƒÂ©tadonnÃƒÂ©es virales utilisant directement le LLM local"""
    
    def __init__(self, model: str = "gemma3:4b", timeout: int = 120):
        self.model = model
        self.timeout = timeout
        self.api_url = "http://localhost:11434/api/generate"
        
        # Ã°Å¸Â§Â  PROMPT SYSTÃƒË†ME OPTIMISÃƒâ€° pour viralitÃƒÂ© maximale
        self.system_prompt = """Create ULTRA-VIRAL TikTok/Instagram metadata that will explode on social media.

CRITICAL VIRAL REQUIREMENTS:
- Title: Ã¢â€°Â¤60 chars, MUST start with Ã°Å¸â€Â¥Ã°Å¸â€™Â¡Ã°Å¸Å¡â‚¬Ã°Å¸ËœÂ±Ã°Å¸Â¤Â¯, be EXTREMELY catchy
- Description: Ã¢â€°Â¤180 chars, include STRONG CTA like "Watch NOW", "You won't BELIEVE"
- Hashtags: 10-15 total, MIX trending + niche + viral + community

VIRAL TITLE PATTERNS:
- "Ã°Å¸â€Â¥ This [Topic] Will BLOW Your Mind"
- "Ã°Å¸ËœÂ± [Number] Things About [Topic] You Never Knew"
- "Ã°Å¸Å¡â‚¬ How I [Achievement] in [Timeframe]"
- "Ã°Å¸â€™Â¡ [Topic] Secrets That Changed Everything"

VIRAL DESCRIPTION PATTERNS:
- "You won't BELIEVE what happens next! Watch NOW to discover the truth about [topic]"
- "This [topic] revelation will SHOCK you! Share if you agree! Ã°Å¸â€Â¥"
- "The [topic] hack that changed my life! Try it NOW and thank me later! Ã°Å¸â€™Âª"

HASHTAG STRATEGY:
- 3-4 TRENDING: #fyp #viral #trending #foryou
- 3-4 NICHE: specific to your topic
- 3-4 ENGAGEMENT: #fypage #explore #shorts #reels
- 2-3 COMMUNITY: #tiktok #content #video

OUTPUT: JSON only with exact format:
{
  "title": "Your viral title here",
  "description": "Your viral description here", 
  "hashtags": ["#tag1", "#tag2", "#tag3"]
}

Transcript:"""
        
        # Ã¯Â¿Â½Ã¯Â¿Â½ PROMPT ULTRA-VIRAL mais court pour modÃƒÂ¨les 4B
        self.fast_prompt = """ULTRA-VIRAL social media metadata.

Title: Ã¢â€°Â¤60 chars, start with Ã°Å¸â€Â¥Ã°Å¸â€™Â¡Ã°Å¸Å¡â‚¬Ã°Å¸ËœÂ±Ã°Å¸Â¤Â¯, be EXTREMELY catchy
Description: Ã¢â€°Â¤180 chars, include "Watch NOW", "You won't BELIEVE"
Hashtags: 10-15 mix trending + niche + viral

JSON: {"title": "Ã°Å¸â€Â¥ Title", "description": "Description", "hashtags": ["#tag"]}

Transcript:"""

    def generate_viral_metadata(self, transcript: str) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ¨re des mÃƒÂ©tadonnÃƒÂ©es virales en 2 appels sÃƒÂ©parÃƒÂ©s pour ÃƒÂ©viter les timeouts"""
        
        try:
            print(f"Ã°Å¸Â§Â  [LLM] SPLIT METADATA pour {self.model} (2 appels)")
            print(f"Ã°Å¸Â§Â  [LLM] GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es virales pour {len(transcript)} caractÃƒÂ¨res")
            print(f"Ã°Å¸Å½Â¯ ModÃƒÂ¨le: {self.model}")
            
            start_time = time.time()
            
            # Ã°Å¸Å¡â‚¬ APPEL 1: Titre + Description
            title_desc = self._generate_title_description(transcript)
            
            # Ã°Å¸Å¡â‚¬ APPEL 2: Hashtags
            hashtags = self._generate_hashtags(transcript)
            
            # Combiner les rÃƒÂ©sultats
            metadata = {
                "title": title_desc.get("title", ""),
                "description": title_desc.get("description", ""),
                "hashtags": hashtags
            }
            
            duration = time.time() - start_time
            print(f"Ã¢Å“â€¦ [MÃƒâ€°TADONNÃƒâ€°ES SPLIT] Titre: {metadata.get('title', 'N/A')[:50]}...")
            print(f"Ã°Å¸â€œâ€“ Description: {metadata.get('description', 'N/A')[:50]}...")
            print(f"#Ã¯Â¸ÂÃ¢Æ’Â£ Hashtags: {len(metadata.get('hashtags', []))} gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
            print(f"Ã¢ÂÂ±Ã¯Â¸Â Temps total: {duration:.1f}s")
            
            return metadata
            
        except Exception as e:
            print(f"Ã°Å¸â€â€ž [FALLBACK] Erreur split metadata: {str(e)}")
            return self._generate_fallback_metadata(transcript)
    
    def _generate_title_description(self, transcript: str) -> Dict[str, str]:
        """GÃƒÂ©nÃƒÂ¨re titre et description en un appel"""
        try:
            # Prompt optimisÃƒÂ© pour titre + description
            title_desc_prompt = """Generate VIRAL TikTok/Instagram title and description.

REQUIREMENTS:
- Title: Ã¢â€°Â¤60 chars, start with Ã°Å¸â€Â¥Ã°Å¸â€™Â¡Ã°Å¸Å¡â‚¬Ã°Å¸ËœÂ±Ã°Å¸Â¤Â¯, be EXTREMELY catchy
- Description: Ã¢â€°Â¤180 chars, include "Watch NOW", "You won't BELIEVE"

JSON format: {"title": "Ã°Å¸â€Â¥ Title", "description": "Description"}

Transcript:"""
            
            full_prompt = title_desc_prompt + transcript
            print(f"Ã°Å¸â€œÂ [APPEL 1] Titre+Description: {len(full_prompt)} chars")
            
            # Timeout rÃƒÂ©duit pour appel simple
            timeout = 75 if self.model in ["gemma3:4b", "qwen3:4b"] else 90
            
            response = self._call_llm(full_prompt, timeout)
            
            if response:
                parsed = self._parse_title_description(response)
                if parsed:
                    print(f"Ã¢Å“â€¦ [APPEL 1] Titre+Description gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
                    return parsed
                
                # Fallback
                return {
                    "title": "Ã°Å¸â€Â¥ Amazing Content That Will BLOW Your Mind!",
                    "description": "You won't BELIEVE what happens next! Watch NOW to discover the truth! Ã°Å¸â€Â¥"
                }
            
        except Exception as e:
            print(f"Ã¢Å¡Â Ã¯Â¸Â [APPEL 1] Erreur: {e}")
            return {
                "title": "Ã°Å¸â€Â¥ Amazing Content That Will BLOW Your Mind!",
                "description": "You won't BELIEVE what happens next! Watch NOW to discover the truth! Ã°Å¸â€Â¥"
            }
    
    def _generate_hashtags(self, transcript: str) -> List[str]:
        """GÃƒÂ©nÃƒÂ¨re hashtags en un appel sÃƒÂ©parÃƒÂ©"""
        try:
            # Prompt optimisÃƒÂ© pour hashtags
            hashtags_prompt = """Generate 10-15 VIRAL hashtags for TikTok/Instagram.

MIX:
- 3-4 TRENDING: #fyp #viral #trending #foryou
- 3-4 NICHE: specific to content topic
- 3-4 ENGAGEMENT: #fypage #explore #shorts #reels
- 2-3 COMMUNITY: #tiktok #content #video

JSON format: {"hashtags": ["#tag1", "#tag2", "#tag3"]}

Transcript:"""
            
            full_prompt = hashtags_prompt + transcript
            print(f"Ã°Å¸â€œÂ [APPEL 2] Hashtags: {len(full_prompt)} chars")
            
            # Timeout rÃƒÂ©duit pour appel simple
            timeout = 60 if self.model in ["gemma3:4b", "qwen3:4b"] else 75
            
            response = self._call_llm(full_prompt, timeout)
            
            if response:
                parsed = self._parse_hashtags(response)
                if parsed:
                    print(f"Ã¢Å“â€¦ [APPEL 2] {len(parsed)} hashtags gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
                    return parsed
                
                # Fallback
                return ["#fyp", "#viral", "#trending", "#foryou", "#explore", "#shorts", "#reels", "#tiktok", "#content", "#video", "#fypage"]
            
        except Exception as e:
            print(f"Ã¢Å¡Â Ã¯Â¸Â [APPEL 2] Erreur: {e}")
            return ["#fyp", "#viral", "#trending", "#foryou", "#explore", "#shorts", "#reels", "#tiktok", "#content", "#video", "#fypage"]
    
    def _call_llm(self, prompt: str, timeout: int) -> Optional[str]:
        """Appel LLM gÃƒÂ©nÃƒÂ©rique qui retourne la rÃƒÂ©ponse brute (str) ou None en cas d'erreur"""
        try:
            start_time = time.time()
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=timeout
            )
            duration = time.time() - start_time
            if response.status_code == 200:
                data = response.json() if response.headers.get('Content-Type','').startswith('application/json') else {}
                llm_response = (data.get('response') or response.text or '').strip()
                logger.info(f"Ã¢Å“â€¦ [LLM] RÃƒÂ©ponse reÃƒÂ§ue en {duration:.1f}s | {len(llm_response)} chars")
                return llm_response if llm_response else None
            else:
                logger.error(f"Ã¢ÂÅ’ [LLM] Erreur HTTP: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            logger.error(f"Ã¢ÂÂ±Ã¯Â¸Â [LLM] Timeout aprÃƒÂ¨s {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ [LLM] Erreur gÃƒÂ©nÃƒÂ©rale: {e}")
            return None
    
    def _clean_llm_response(self, response: str) -> str:
        """Nettoie la rÃƒÂ©ponse du LLM pour extraire le JSON"""
        if not response:
            return "{}"
        # Chercher le JSON dans la rÃƒÂ©ponse
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx + 1]
        # Si pas de JSON trouvÃƒÂ©, essayer de nettoyer
        cleaned = response.replace('```json', '').replace('```', '').strip()
        return cleaned if cleaned else "{}"
    
    def _validate_metadata(self, data: Dict[str, Any]) -> bool:
        """Valide la structure des mÃƒÂ©tadonnÃƒÂ©es"""
        
        required_fields = ['title', 'description', 'hashtags']
        
        # VÃƒÂ©rifier que tous les champs requis sont prÃƒÂ©sents
        for field in required_fields:
            if field not in data:
                print(f"Ã¢ÂÅ’ Champ manquant: {field}")
                return False
        
        # VÃƒÂ©rifier le titre
        title = data.get('title', '')
        if not title or len(title) > 60:
            print(f"Ã¢ÂÅ’ Titre invalide: {len(title)} caractÃƒÂ¨res (max 60)")
            return False
        
        # VÃƒÂ©rifier que le titre commence par un emoji viral
        viral_emojis = ['Ã°Å¸â€Â¥', 'Ã°Å¸â€™Â¡', 'Ã°Å¸Å¡â‚¬', 'Ã°Å¸ËœÂ±', 'Ã°Å¸Â¤Â¯', 'Ã°Å¸â€™Âª', 'Ã°Å¸Å½Â¯', 'Ã¢Å¡Â¡']
        if not any(title.startswith(emoji) for emoji in viral_emojis):
            print(f"Ã¢ÂÅ’ Titre doit commencer par un emoji viral: {viral_emojis}")
            return False
        
        # VÃƒÂ©rifier la description
        description = data.get('description', '')
        if not description or len(description) > 180:
            print(f"Ã¢ÂÅ’ Description invalide: {len(description)} caractÃƒÂ¨res (max 180)")
            return False
        
        # VÃƒÂ©rifier que la description contient un CTA viral
        viral_ctas = ['watch', 'now', 'believe', 'share', 'try', 'discover', 'shock']
        if not any(cta in description.lower() for cta in viral_ctas):
            print(f"Ã¢ÂÅ’ Description doit contenir un CTA viral: {viral_ctas}")
            return False
        
        # VÃƒÂ©rifier les hashtags
        hashtags = data.get('hashtags', [])
        if not isinstance(hashtags, list) or len(hashtags) < 10:
            print(f"Ã¢ÂÅ’ Hashtags invalides: {len(hashtags)} (min 10)")
            return False
        
        return True
    
    def _validate_hashtags(self, hashtags: List[str]) -> List[str]:
        """Valide et nettoie les hashtags"""
        
        validated = []
        
        for hashtag in hashtags:
            if isinstance(hashtag, str) and hashtag.strip():
                clean_hashtag = hashtag.strip()
                
                # S'assurer que ÃƒÂ§a commence par #
                if not clean_hashtag.startswith('#'):
                    clean_hashtag = '#' + clean_hashtag
                
                # Ãƒâ€°viter les hashtags trop longs
                if len(clean_hashtag) <= 30:
                    validated.append(clean_hashtag)
        
        # Garantir au moins 10 hashtags avec stratÃƒÂ©gie virale
        if len(validated) < 10:
            # Ajouter des hashtags viraux de fallback
            viral_hashtags = ['#fyp', '#viral', '#trending', '#foryou', '#explore', '#shorts', '#reels', '#tiktok', '#content', '#video', '#fypage', '#foryoupage']
            for i in range(10 - len(validated)):
                if viral_hashtags[i] not in validated:
                    validated.append(viral_hashtags[i])
        
        # Limiter ÃƒÂ  15 maximum
        return validated[:15]
    
    def _fallback_generation(self, transcript: str, error_reason: str) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ©ration de fallback intelligente"""
        
        print(f"Ã°Å¸â€â€ž [FALLBACK] GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es de fallback: {error_reason}")
        
        # Analyse simple du transcript
        words = transcript.lower().split()
        
        # Extraire des mots-clÃƒÂ©s pour le titre
        relevant_words = []
        for word in words:
            if len(word) > 3 and word.isalpha():
                common_words = ['the', 'and', 'that', 'this', 'with', 'from', 'they', 'have', 'been', 'will', 'would', 'could', 'should']
                if word not in common_words:
                    relevant_words.append(word)
        
        # Prendre les mots les plus frÃƒÂ©quents
        from collections import Counter
        word_counts = Counter(relevant_words)
        top_words = [word for word, _ in word_counts.most_common(5)]
        
        # GÃƒÂ©nÃƒÂ©rer un titre viral de fallback
        if top_words:
            main_topic = top_words[0].title()
            title = f"Ã°Å¸â€Â¥ {main_topic} Secrets That Will BLOW Your Mind!"
        else:
            title = "Ã°Å¸â€Â¥ Amazing Content - You Won't BELIEVE This!"
        
        # GÃƒÂ©nÃƒÂ©rer une description virale
        description = f"You won't BELIEVE what happens next! Watch NOW to discover the truth about {top_words[0] if top_words else 'success'}! Ã°Å¸â€Â¥"
        
        # Hashtags viraux de fallback
        hashtags = ['#fyp', '#viral', '#trending', '#foryou', '#explore', '#shorts', '#reels', '#tiktok', '#content', '#video', '#fypage']
        
        print(f"Ã°Å¸â€â€ž [FALLBACK] MÃƒÂ©tadonnÃƒÂ©es virales gÃƒÂ©nÃƒÂ©rÃƒÂ©es par fallback")
        
        return {
            'success': True,
            'title': title,
            'description': description,
            'hashtags': hashtags,
            'processing_time': 0.1,
            'model_used': 'fallback_system',
            'fallback_reason': error_reason
        }

    def _parse_title_description(self, response: str) -> Optional[Dict[str, str]]:
        """Parse la rÃƒÂ©ponse titre + description"""
        try:
            # Nettoyer la rÃƒÂ©ponse
            cleaned = self._clean_llm_response(response)
            parsed = json.loads(cleaned)
            
            if isinstance(parsed, dict) and 'title' in parsed and 'description' in parsed:
                return {
                    'title': parsed['title'].strip(),
                    'description': parsed['description'].strip()
                }
        except Exception as e:
            print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur parsing titre/description: {e}")
        return None
    
    def _parse_hashtags(self, response: str) -> Optional[List[str]]:
        """Parse la rÃƒÂ©ponse hashtags"""
        try:
            # Nettoyer la rÃƒÂ©ponse
            cleaned = self._clean_llm_response(response)
            parsed = json.loads(cleaned)
            
            if isinstance(parsed, dict) and 'hashtags' in parsed:
                hashtags = parsed['hashtags']
                if isinstance(hashtags, list):
                    # Nettoyer et valider les hashtags
                    clean_hashtags = []
                    for tag in hashtags:
                        if isinstance(tag, str):
                            tag = tag.strip()
                            if not tag.startswith('#'):
                                tag = '#' + tag
                            clean_hashtags.append(tag)
                    return clean_hashtags[:15]  # Limiter ÃƒÂ  15 max
        except Exception as e:
            print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur parsing hashtags: {e}")
        return None

def create_llm_metadata_generator(model: str = "gemma3:4b") -> LLMMetadataGenerator:
    """Factory pour crÃƒÂ©er un gÃƒÂ©nÃƒÂ©rateur de mÃƒÂ©tadonnÃƒÂ©es LLM"""
    return LLMMetadataGenerator(model=model) 

