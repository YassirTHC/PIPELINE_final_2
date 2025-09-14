# 🚀 GÉNÉRATEUR DE MÉTADONNÉES VIRALES AVEC LLM DIRECT
# Titres, descriptions et hashtags TikTok/Instagram optimisés

import json
import logging
import time
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class LLMMetadataGenerator:
    """Générateur de métadonnées virales utilisant directement le LLM local"""
    
    def __init__(self, model: str = "gemma3:4b", timeout: int = 120):
        self.model = model
        self.timeout = timeout
        self.api_url = "http://localhost:11434/api/generate"
        
        # 🧠 PROMPT SYSTÈME OPTIMISÉ pour Gemma3:4B avec détection domaine automatique
        self.system_prompt = """Analyze content and create domain-specific viral metadata.

STEP 1 - DOMAIN ANALYSIS:
Detect specific domain from content (medical_therapy, tech_innovation, cooking_techniques, fitness_training, education_science, business_strategy, creative_arts, etc.). Create new domains as needed.

STEP 2 - CONTEXT IDENTIFICATION:
Identify specific context/technique within domain.

STEP 3 - VIRAL CONTENT GENERATION:
- Title: ≤60 chars, viral emoji start 🔥💡🚀😱🤯
- Description: ≤180 chars, strong CTA "Watch NOW", "You won't BELIEVE"
- Hashtags: 12-15 mix (trending + domain + engagement)
- B-roll: 15-20 VISUAL search terms (specific actions/objects/settings)

JSON OUTPUT:
{
  "domain": "detected_domain",
  "context": "specific_context", 
  "title": "🔥 Viral Title",
  "description": "Engaging description with CTA",
  "hashtags": ["#domain", "#trending", "#fyp"],
  "broll_keywords": ["specific_visual1", "concrete_action2"],
  "search_queries": ["2-4 word search phrase"]
}

B-ROLL REQUIREMENTS:
Generate DOMAIN-SPECIFIC visual keywords:
✅ CONCRETE: "surgeon_operating_room", "chef_knife_skills"
✅ SEARCHABLE: "patient_consultation_closeup", "coding_screen_multiple"
✅ PROFESSIONAL: "therapist_notes_session", "trainer_exercise_demo"
❌ AVOID: Generic "person", "room", abstract "success", "growth"

Transcript:"""
        
        # �� PROMPT ULTRA-VIRAL mais court pour modèles 4B
        self.fast_prompt = """ULTRA-VIRAL social media metadata.

Title: ≤60 chars, start with 🔥💡🚀😱🤯, be EXTREMELY catchy
Description: ≤180 chars, include "Watch NOW", "You won't BELIEVE"
Hashtags: 10-15 mix trending + niche + viral

JSON: {"title": "🔥 Title", "description": "Description", "hashtags": ["#tag"]}

Transcript:"""

    def generate_viral_metadata(self, transcript: str) -> Dict[str, Any]:
        """Génère des métadonnées virales en 2 appels séparés pour éviter les timeouts"""
        
        try:
            print(f"🧠 [LLM] SPLIT METADATA pour {self.model} (2 appels)")
            print(f"🧠 [LLM] Génération métadonnées virales pour {len(transcript)} caractères")
            print(f"🎯 Modèle: {self.model}")
            
            start_time = time.time()
            
            # 🚀 APPEL 1: Titre + Description
            title_desc = self._generate_title_description(transcript)
            
            # 🚀 APPEL 2: Hashtags
            hashtags = self._generate_hashtags(transcript)
            
            # Combiner les résultats
            metadata = {
                "title": title_desc.get("title", ""),
                "description": title_desc.get("description", ""),
                "hashtags": hashtags
            }
            
            duration = time.time() - start_time
            print(f"✅ [MÉTADONNÉES SPLIT] Titre: {metadata.get('title', 'N/A')[:50]}...")
            print(f"📖 Description: {metadata.get('description', 'N/A')[:50]}...")
            print(f"#️⃣ Hashtags: {len(metadata.get('hashtags', []))} générés")
            print(f"⏱️ Temps total: {duration:.1f}s")
            
            return metadata
            
        except Exception as e:
            print(f"🔄 [FALLBACK] Erreur split metadata: {str(e)}")
            return self._generate_fallback_metadata(transcript)
    
    def _generate_title_description(self, transcript: str) -> Dict[str, str]:
        """Génère titre et description en un appel"""
        try:
            # Prompt optimisé pour titre + description
            title_desc_prompt = """Generate VIRAL TikTok/Instagram title and description.

REQUIREMENTS:
- Title: ≤60 chars, start with 🔥💡🚀😱🤯, be EXTREMELY catchy
- Description: ≤180 chars, include "Watch NOW", "You won't BELIEVE"

JSON format: {"title": "🔥 Title", "description": "Description"}

Transcript:"""
            
            full_prompt = title_desc_prompt + transcript
            print(f"📝 [APPEL 1] Titre+Description: {len(full_prompt)} chars")
            
            # Timeout réduit pour appel simple
            timeout = 75 if self.model in ["gemma3:4b", "qwen3:4b"] else 90
            
            response = self._call_llm(full_prompt, timeout)
            
            if response:
                parsed = self._parse_title_description(response)
                if parsed:
                    print(f"✅ [APPEL 1] Titre+Description générés")
                    return parsed
            
            # Fallback
            return {
                "title": "🔥 Amazing Content That Will BLOW Your Mind!",
                "description": "You won't BELIEVE what happens next! Watch NOW to discover the truth! 🔥"
            }
            
        except Exception as e:
            print(f"⚠️ [APPEL 1] Erreur: {e}")
            return {
                "title": "🔥 Amazing Content That Will BLOW Your Mind!",
                "description": "You won't BELIEVE what happens next! Watch NOW to discover the truth! 🔥"
            }
    
    def _generate_hashtags(self, transcript: str) -> List[str]:
        """Génère hashtags en un appel séparé"""
        try:
            # Prompt optimisé pour hashtags
            hashtags_prompt = """Generate 10-15 VIRAL hashtags for TikTok/Instagram.

MIX:
- 3-4 TRENDING: #fyp #viral #trending #foryou
- 3-4 NICHE: specific to content topic
- 3-4 ENGAGEMENT: #fypage #explore #shorts #reels
- 2-3 COMMUNITY: #tiktok #content #video

JSON format: {"hashtags": ["#tag1", "#tag2", "#tag3"]}

Transcript:"""
            
            full_prompt = hashtags_prompt + transcript
            print(f"📝 [APPEL 2] Hashtags: {len(full_prompt)} chars")
            
            # Timeout réduit pour appel simple
            timeout = 60 if self.model in ["gemma3:4b", "qwen3:4b"] else 75
            
            response = self._call_llm(full_prompt, timeout)
            
            if response:
                parsed = self._parse_hashtags(response)
                if parsed:
                    print(f"✅ [APPEL 2] {len(parsed)} hashtags générés")
                    return parsed
            
            # Fallback
            return ["#fyp", "#viral", "#trending", "#foryou", "#explore", "#shorts", "#reels", "#tiktok", "#content", "#video", "#fypage"]
            
        except Exception as e:
            print(f"⚠️ [APPEL 2] Erreur: {e}")
            return ["#fyp", "#viral", "#trending", "#foryou", "#explore", "#shorts", "#reels", "#tiktok", "#content", "#video", "#fypage"]
    
    def _call_llm(self, prompt: str, timeout: int) -> Optional[str]:
        """Appel LLM générique"""
        try:
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
                    # Nettoyer la réponse
                    cleaned_response = self._clean_llm_response(llm_response)
                    
                    # Parser le JSON
                    parsed_data = json.loads(cleaned_response)
                    
                    # Validation des champs
                    if self._validate_metadata(parsed_data):
                        title = parsed_data.get('title', '').strip()
                        description = parsed_data.get('description', '').strip()
                        hashtags = parsed_data.get('hashtags', [])
                        
                        # Nettoyer et valider les hashtags
                        validated_hashtags = self._validate_hashtags(hashtags)
                        
                        print(f"🎯 [LLM] Métadonnées virales générées")
                        print(f"📝 Titre: {title}")
                        print(f"📖 Description: {description[:50]}...")
                        print(f"#️⃣ Hashtags: {len(validated_hashtags)} générés")
                        
                        return {
                            'success': True,
                            'title': title,
                            'description': description,
                            'hashtags': validated_hashtags,
                            'processing_time': duration,
                            'model_used': self.model
                        }
                    else:
                        raise ValueError("Métadonnées invalides")
                        
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
    
    def _validate_metadata(self, data: Dict[str, Any]) -> bool:
        """Valide la structure des métadonnées"""
        
        required_fields = ['title', 'description', 'hashtags']
        
        # Vérifier que tous les champs requis sont présents
        for field in required_fields:
            if field not in data:
                print(f"❌ Champ manquant: {field}")
                return False
        
        # Vérifier le titre
        title = data.get('title', '')
        if not title or len(title) > 60:
            print(f"❌ Titre invalide: {len(title)} caractères (max 60)")
            return False
        
        # Vérifier que le titre commence par un emoji viral
        viral_emojis = ['🔥', '💡', '🚀', '😱', '🤯', '💪', '🎯', '⚡']
        if not any(title.startswith(emoji) for emoji in viral_emojis):
            print(f"❌ Titre doit commencer par un emoji viral: {viral_emojis}")
            return False
        
        # Vérifier la description
        description = data.get('description', '')
        if not description or len(description) > 180:
            print(f"❌ Description invalide: {len(description)} caractères (max 180)")
            return False
        
        # Vérifier que la description contient un CTA viral
        viral_ctas = ['watch', 'now', 'believe', 'share', 'try', 'discover', 'shock']
        if not any(cta in description.lower() for cta in viral_ctas):
            print(f"❌ Description doit contenir un CTA viral: {viral_ctas}")
            return False
        
        # Vérifier les hashtags
        hashtags = data.get('hashtags', [])
        if not isinstance(hashtags, list) or len(hashtags) < 10:
            print(f"❌ Hashtags invalides: {len(hashtags)} (min 10)")
            return False
        
        return True
    
    def _validate_hashtags(self, hashtags: List[str]) -> List[str]:
        """Valide et nettoie les hashtags"""
        
        validated = []
        
        for hashtag in hashtags:
            if isinstance(hashtag, str) and hashtag.strip():
                clean_hashtag = hashtag.strip()
                
                # S'assurer que ça commence par #
                if not clean_hashtag.startswith('#'):
                    clean_hashtag = '#' + clean_hashtag
                
                # Éviter les hashtags trop longs
                if len(clean_hashtag) <= 30:
                    validated.append(clean_hashtag)
        
        # Garantir au moins 10 hashtags avec stratégie virale
        if len(validated) < 10:
            # Ajouter des hashtags viraux de fallback
            viral_hashtags = ['#fyp', '#viral', '#trending', '#foryou', '#explore', '#shorts', '#reels', '#tiktok', '#content', '#video', '#fypage', '#foryoupage']
            for i in range(10 - len(validated)):
                if viral_hashtags[i] not in validated:
                    validated.append(viral_hashtags[i])
        
        # Limiter à 15 maximum
        return validated[:15]
    
    def _fallback_generation(self, transcript: str, error_reason: str) -> Dict[str, Any]:
        """Génération de fallback intelligente"""
        
        print(f"🔄 [FALLBACK] Génération métadonnées de fallback: {error_reason}")
        
        # Analyse simple du transcript
        words = transcript.lower().split()
        
        # Extraire des mots-clés pour le titre
        relevant_words = []
        for word in words:
            if len(word) > 3 and word.isalpha():
                common_words = ['the', 'and', 'that', 'this', 'with', 'from', 'they', 'have', 'been', 'will', 'would', 'could', 'should']
                if word not in common_words:
                    relevant_words.append(word)
        
        # Prendre les mots les plus fréquents
        from collections import Counter
        word_counts = Counter(relevant_words)
        top_words = [word for word, _ in word_counts.most_common(5)]
        
        # Générer un titre viral de fallback
        if top_words:
            main_topic = top_words[0].title()
            title = f"🔥 {main_topic} Secrets That Will BLOW Your Mind!"
        else:
            title = "🔥 Amazing Content - You Won't BELIEVE This!"
        
        # Générer une description virale
        description = f"You won't BELIEVE what happens next! Watch NOW to discover the truth about {top_words[0] if top_words else 'success'}! 🔥"
        
        # Hashtags viraux de fallback
        hashtags = ['#fyp', '#viral', '#trending', '#foryou', '#explore', '#shorts', '#reels', '#tiktok', '#content', '#video', '#fypage']
        
        print(f"🔄 [FALLBACK] Métadonnées virales générées par fallback")
        
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
        """Parse la réponse titre + description"""
        try:
            # Nettoyer la réponse
            cleaned = self._clean_llm_response(response)
            parsed = json.loads(cleaned)
            
            if isinstance(parsed, dict) and 'title' in parsed and 'description' in parsed:
                return {
                    'title': parsed['title'].strip(),
                    'description': parsed['description'].strip()
                }
        except Exception as e:
            print(f"⚠️ Erreur parsing titre/description: {e}")
        return None
    
    def _parse_hashtags(self, response: str) -> Optional[List[str]]:
        """Parse la réponse hashtags"""
        try:
            # Nettoyer la réponse
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
                    return clean_hashtags[:15]  # Limiter à 15 max
        except Exception as e:
            print(f"⚠️ Erreur parsing hashtags: {e}")
        return None

def create_llm_metadata_generator(model: str = "gemma3:4b") -> LLMMetadataGenerator:
    """Factory pour créer un générateur de métadonnées LLM"""
    return LLMMetadataGenerator(model=model) 