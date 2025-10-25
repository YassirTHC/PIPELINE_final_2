# -*- coding: utf-8 -*-
# ðŸš€ GÃ‰NÃ‰RATEUR DE MÃ‰TADONNÃ‰ES VIRALES AVEC LLM DIRECT
# Titres, descriptions et hashtags TikTok/Instagram optimisÃ©s

import json
import logging
import time
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class LLMMetadataGenerator:
    """GÃ©nÃ©rateur de mÃ©tadonnÃ©es virales utilisant directement le LLM local"""
    
    def __init__(self, model: str = "gemma3:4b", timeout: int = 120):
        self.model = model
        self.timeout = timeout
        self.api_url = "http://localhost:11434/api/generate"
        
        # ðŸ§  PROMPT SYSTÃˆME OPTIMISÃ‰ pour viralitÃ© maximale
        self.system_prompt = """Create ULTRA-VIRAL TikTok/Instagram metadata that will explode on social media.

CRITICAL VIRAL REQUIREMENTS:
- Title: â‰¤60 chars, MUST start with ðŸ”¥ðŸ’¡ðŸš€ðŸ˜±ðŸ¤¯, be EXTREMELY catchy
- Description: â‰¤180 chars, include STRONG CTA like "Watch NOW", "You won't BELIEVE"
- Hashtags: 10-15 total, MIX trending + niche + viral + community

VIRAL TITLE PATTERNS:
- "ðŸ”¥ This [Topic] Will BLOW Your Mind"
- "ðŸ˜± [Number] Things About [Topic] You Never Knew"
- "ðŸš€ How I [Achievement] in [Timeframe]"
- "ðŸ’¡ [Topic] Secrets That Changed Everything"

VIRAL DESCRIPTION PATTERNS:
- "You won't BELIEVE what happens next! Watch NOW to discover the truth about [topic]"
- "This [topic] revelation will SHOCK you! Share if you agree! ðŸ”¥"
- "The [topic] hack that changed my life! Try it NOW and thank me later! ðŸ’ª"

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
        
        # ï¿½ï¿½ PROMPT ULTRA-VIRAL mais court pour modÃ¨les 4B
        self.fast_prompt = """ULTRA-VIRAL social media metadata.

Title: â‰¤60 chars, start with ðŸ”¥ðŸ’¡ðŸš€ðŸ˜±ðŸ¤¯, be EXTREMELY catchy
Description: â‰¤180 chars, include "Watch NOW", "You won't BELIEVE"
Hashtags: 10-15 mix trending + niche + viral

JSON: {"title": "ðŸ”¥ Title", "description": "Description", "hashtags": ["#tag"]}

Transcript:"""

    def generate_viral_metadata(self, transcript: str) -> Dict[str, Any]:
        """GÃ©nÃ¨re des mÃ©tadonnÃ©es virales en 2 appels sÃ©parÃ©s pour Ã©viter les timeouts"""
        
        try:
            print(f"ðŸ§  [LLM] SPLIT METADATA pour {self.model} (2 appels)")
            print(f"ðŸ§  [LLM] GÃ©nÃ©ration mÃ©tadonnÃ©es virales pour {len(transcript)} caractÃ¨res")
            print(f"ðŸŽ¯ ModÃ¨le: {self.model}")
            
            start_time = time.time()
            
            # ðŸš€ APPEL 1: Titre + Description
            title_desc = self._generate_title_description(transcript)
            
            # ðŸš€ APPEL 2: Hashtags
            hashtags = self._generate_hashtags(transcript)
            
            # Combiner les rÃ©sultats
            metadata = {
                "title": title_desc.get("title", ""),
                "description": title_desc.get("description", ""),
                "hashtags": hashtags
            }
            
            duration = time.time() - start_time
            print(f"âœ… [MÃ‰TADONNÃ‰ES SPLIT] Titre: {metadata.get('title', 'N/A')[:50]}...")
            print(f"ðŸ“– Description: {metadata.get('description', 'N/A')[:50]}...")
            print(f"#ï¸âƒ£ Hashtags: {len(metadata.get('hashtags', []))} gÃ©nÃ©rÃ©s")
            print(f"â±ï¸ Temps total: {duration:.1f}s")
            
            return metadata
            
        except Exception as e:
            print(f"ðŸ”„ [FALLBACK] Erreur split metadata: {str(e)}")
            return self._generate_fallback_metadata(transcript)
    
    def _generate_title_description(self, transcript: str) -> Dict[str, str]:
        """GÃ©nÃ¨re titre et description en un appel"""
        try:
            # Prompt optimisÃ© pour titre + description
            title_desc_prompt = """Generate VIRAL TikTok/Instagram title and description.

REQUIREMENTS:
- Title: â‰¤60 chars, start with ðŸ”¥ðŸ’¡ðŸš€ðŸ˜±ðŸ¤¯, be EXTREMELY catchy
- Description: â‰¤180 chars, include "Watch NOW", "You won't BELIEVE"

JSON format: {"title": "ðŸ”¥ Title", "description": "Description"}

Transcript:"""
            
            full_prompt = title_desc_prompt + transcript
            print(f"ðŸ“ [APPEL 1] Titre+Description: {len(full_prompt)} chars")
            
            # Timeout rÃ©duit pour appel simple
            timeout = 75 if self.model in ["gemma3:4b", "qwen3:4b"] else 90
            
            response = self._call_llm(full_prompt, timeout)
            
            if response:
                parsed = self._parse_title_description(response)
                if parsed:
                    print(f"âœ… [APPEL 1] Titre+Description gÃ©nÃ©rÃ©s")
                    return parsed
                
                # Fallback
                return {
                    "title": "ðŸ”¥ Amazing Content That Will BLOW Your Mind!",
                    "description": "You won't BELIEVE what happens next! Watch NOW to discover the truth! ðŸ”¥"
                }
            
        except Exception as e:
            print(f"âš ï¸ [APPEL 1] Erreur: {e}")
            return {
                "title": "ðŸ”¥ Amazing Content That Will BLOW Your Mind!",
                "description": "You won't BELIEVE what happens next! Watch NOW to discover the truth! ðŸ”¥"
            }
    
    def _generate_hashtags(self, transcript: str) -> List[str]:
        """GÃ©nÃ¨re hashtags en un appel sÃ©parÃ©"""
        try:
            # Prompt optimisÃ© pour hashtags
            hashtags_prompt = """Generate 10-15 VIRAL hashtags for TikTok/Instagram.

MIX:
- 3-4 TRENDING: #fyp #viral #trending #foryou
- 3-4 NICHE: specific to content topic
- 3-4 ENGAGEMENT: #fypage #explore #shorts #reels
- 2-3 COMMUNITY: #tiktok #content #video

JSON format: {"hashtags": ["#tag1", "#tag2", "#tag3"]}

Transcript:"""
            
            full_prompt = hashtags_prompt + transcript
            print(f"ðŸ“ [APPEL 2] Hashtags: {len(full_prompt)} chars")
            
            # Timeout rÃ©duit pour appel simple
            timeout = 60 if self.model in ["gemma3:4b", "qwen3:4b"] else 75
            
            response = self._call_llm(full_prompt, timeout)
            
            if response:
                parsed = self._parse_hashtags(response)
                if parsed:
                    print(f"âœ… [APPEL 2] {len(parsed)} hashtags gÃ©nÃ©rÃ©s")
                    return parsed
                
                # Fallback
                return ["#fyp", "#viral", "#trending", "#foryou", "#explore", "#shorts", "#reels", "#tiktok", "#content", "#video", "#fypage"]
            
        except Exception as e:
            print(f"âš ï¸ [APPEL 2] Erreur: {e}")
            return ["#fyp", "#viral", "#trending", "#foryou", "#explore", "#shorts", "#reels", "#tiktok", "#content", "#video", "#fypage"]
    
    def _call_llm(self, prompt: str, timeout: int) -> Optional[str]:
        """Appel LLM gÃ©nÃ©rique qui retourne la rÃ©ponse brute (str) ou None en cas d'erreur"""
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
                logger.info(f"âœ… [LLM] RÃ©ponse reÃ§ue en {duration:.1f}s | {len(llm_response)} chars")
                return llm_response if llm_response else None
            else:
                logger.error(f"âŒ [LLM] Erreur HTTP: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            logger.error(f"â±ï¸ [LLM] Timeout aprÃ¨s {timeout}s")
            return None
        except Exception as e:
            logger.error(f"âŒ [LLM] Erreur gÃ©nÃ©rale: {e}")
            return None
    
    def _clean_llm_response(self, response: str) -> str:
        """Nettoie la rÃ©ponse du LLM pour extraire le JSON"""
        if not response:
            return "{}"
        # Chercher le JSON dans la rÃ©ponse
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx + 1]
        # Si pas de JSON trouvÃ©, essayer de nettoyer
        cleaned = response.replace('```json', '').replace('```', '').strip()
        return cleaned if cleaned else "{}"
    
    def _validate_metadata(self, data: Dict[str, Any]) -> bool:
        """Valide la structure des mÃ©tadonnÃ©es"""
        
        required_fields = ['title', 'description', 'hashtags']
        
        # VÃ©rifier que tous les champs requis sont prÃ©sents
        for field in required_fields:
            if field not in data:
                print(f"âŒ Champ manquant: {field}")
                return False
        
        # VÃ©rifier le titre
        title = data.get('title', '')
        if not title or len(title) > 60:
            print(f"âŒ Titre invalide: {len(title)} caractÃ¨res (max 60)")
            return False
        
        # VÃ©rifier que le titre commence par un emoji viral
        viral_emojis = ['ðŸ”¥', 'ðŸ’¡', 'ðŸš€', 'ðŸ˜±', 'ðŸ¤¯', 'ðŸ’ª', 'ðŸŽ¯', 'âš¡']
        if not any(title.startswith(emoji) for emoji in viral_emojis):
            print(f"âŒ Titre doit commencer par un emoji viral: {viral_emojis}")
            return False
        
        # VÃ©rifier la description
        description = data.get('description', '')
        if not description or len(description) > 180:
            print(f"âŒ Description invalide: {len(description)} caractÃ¨res (max 180)")
            return False
        
        # VÃ©rifier que la description contient un CTA viral
        viral_ctas = ['watch', 'now', 'believe', 'share', 'try', 'discover', 'shock']
        if not any(cta in description.lower() for cta in viral_ctas):
            print(f"âŒ Description doit contenir un CTA viral: {viral_ctas}")
            return False
        
        # VÃ©rifier les hashtags
        hashtags = data.get('hashtags', [])
        if not isinstance(hashtags, list) or len(hashtags) < 10:
            print(f"âŒ Hashtags invalides: {len(hashtags)} (min 10)")
            return False
        
        return True
    
    def _validate_hashtags(self, hashtags: List[str]) -> List[str]:
        """Valide et nettoie les hashtags"""
        
        validated = []
        
        for hashtag in hashtags:
            if isinstance(hashtag, str) and hashtag.strip():
                clean_hashtag = hashtag.strip()
                
                # S'assurer que Ã§a commence par #
                if not clean_hashtag.startswith('#'):
                    clean_hashtag = '#' + clean_hashtag
                
                # Ã‰viter les hashtags trop longs
                if len(clean_hashtag) <= 30:
                    validated.append(clean_hashtag)
        
        # Garantir au moins 10 hashtags avec stratÃ©gie virale
        if len(validated) < 10:
            # Ajouter des hashtags viraux de fallback
            viral_hashtags = ['#fyp', '#viral', '#trending', '#foryou', '#explore', '#shorts', '#reels', '#tiktok', '#content', '#video', '#fypage', '#foryoupage']
            for i in range(10 - len(validated)):
                if viral_hashtags[i] not in validated:
                    validated.append(viral_hashtags[i])
        
        # Limiter Ã  15 maximum
        return validated[:15]
    
    def _fallback_generation(self, transcript: str, error_reason: str) -> Dict[str, Any]:
        """GÃ©nÃ©ration de fallback intelligente"""
        
        print(f"ðŸ”„ [FALLBACK] GÃ©nÃ©ration mÃ©tadonnÃ©es de fallback: {error_reason}")
        
        # Analyse simple du transcript
        words = transcript.lower().split()
        
        # Extraire des mots-clÃ©s pour le titre
        relevant_words = []
        for word in words:
            if len(word) > 3 and word.isalpha():
                common_words = ['the', 'and', 'that', 'this', 'with', 'from', 'they', 'have', 'been', 'will', 'would', 'could', 'should']
                if word not in common_words:
                    relevant_words.append(word)
        
        # Prendre les mots les plus frÃ©quents
        from collections import Counter
        word_counts = Counter(relevant_words)
        top_words = [word for word, _ in word_counts.most_common(5)]
        
        # GÃ©nÃ©rer un titre viral de fallback
        if top_words:
            main_topic = top_words[0].title()
            title = f"ðŸ”¥ {main_topic} Secrets That Will BLOW Your Mind!"
        else:
            title = "ðŸ”¥ Amazing Content - You Won't BELIEVE This!"
        
        # GÃ©nÃ©rer une description virale
        description = f"You won't BELIEVE what happens next! Watch NOW to discover the truth about {top_words[0] if top_words else 'success'}! ðŸ”¥"
        
        # Hashtags viraux de fallback
        hashtags = ['#fyp', '#viral', '#trending', '#foryou', '#explore', '#shorts', '#reels', '#tiktok', '#content', '#video', '#fypage']
        
        print(f"ðŸ”„ [FALLBACK] MÃ©tadonnÃ©es virales gÃ©nÃ©rÃ©es par fallback")
        
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
        """Parse la rÃ©ponse titre + description"""
        try:
            # Nettoyer la rÃ©ponse
            cleaned = self._clean_llm_response(response)
            parsed = json.loads(cleaned)
            
            if isinstance(parsed, dict) and 'title' in parsed and 'description' in parsed:
                return {
                    'title': parsed['title'].strip(),
                    'description': parsed['description'].strip()
                }
        except Exception as e:
            print(f"âš ï¸ Erreur parsing titre/description: {e}")
        return None
    
    def _parse_hashtags(self, response: str) -> Optional[List[str]]:
        """Parse la rÃ©ponse hashtags"""
        try:
            # Nettoyer la rÃ©ponse
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
                    return clean_hashtags[:15]  # Limiter Ã  15 max
        except Exception as e:
            print(f"âš ï¸ Erreur parsing hashtags: {e}")
        return None

def create_llm_metadata_generator(model: str = "gemma3:4b") -> LLMMetadataGenerator:
    """Factory pour crÃ©er un gÃ©nÃ©rateur de mÃ©tadonnÃ©es LLM"""
    return LLMMetadataGenerator(model=model) 
