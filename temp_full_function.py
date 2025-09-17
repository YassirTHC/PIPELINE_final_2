#!/usr/bin/env python3
"""
Version corrigée de la fonction LLM pour éviter les erreurs d'indentation
"""

import time
import json as _json
import requests
from typing import Optional, Dict

def _llm_generate_caption_hashtags_fixed(transcript_text: str) -> Optional[Dict[str, object]]:
    """Use local LLM to generate viral-ready title, description, hashtags and B-roll keywords as JSON. Returns dict or None on failure."""
    
    # Configuration LLM centralisée avec fallback
    try:
        import yaml
        with open("config/llm_config.yaml", "r", encoding="utf-8") as f:
            llm_config = yaml.safe_load(f)
        json_retry_attempts = llm_config["llm"]["json_retry_attempts"]
        enforce_json_output = llm_config["llm"]["enforce_json_output"]
        log_prompt = llm_config["llm"]["log_prompt"]
        log_response_time = llm_config["llm"]["log_response_time"]
        log_response_size = llm_config["llm"]["log_response_size"]
    except Exception as e:
        print(f"    ⚠️ Erreur chargement config LLM: {e}")
        json_retry_attempts = 3
        enforce_json_output = True
        log_prompt = True
        log_response_time = True
        log_response_size = True
    
    # Configuration Ollama
    provider = "ollama"
    base = "http://localhost:11434"
    model = "qwen3:8b"
    
    # Limit transcript for prompt brevity
    text = (transcript_text or "").strip()
    if len(text) > 2000:
        text = text[:2000]
    
    # 🚀 NOUVEAU: Prompt enrichi et optimisé pour inclure les mots-clés B-roll
    prompt = (
        "You are a social media strategist and B-roll content expert for TikTok and Instagram.\n"
        "From this transcript, produce elements optimized for virality AND B-roll video content selection.\n"
        "Write in English only, generate keywords that can be directly matched to stock video footage.\n\n"
        "Your goal is to maximize visual relevance and search engine performance across Pexels, Pixabay, and Archive.org,\n"
        "while ensuring maximum engagement potential on social media platforms.\n\n"
        "REQUIRED OUTPUT:\n"
        "1. title: short, catchy (<= 60 characters), TikTok/Instagram Reels style\n"
        "2. description: 1-2 punchy sentences with an implicit call-to-action\n"
        "3. hashtags: 10-14 varied hashtags, exact format #keyword (no spaces), mix of niche + trending\n"
        "4. broll_keywords: 25-35 base keywords, each with 2-3 synonyms, distributed across:\n\n"
        "   VISUAL ACTIONS (8-12): Specific, filmable actions with variations\n"
        "   PEOPLE & ROLES (8-10): Specific person types with alternatives\n"
        "   ENVIRONMENTS & PLACES (8-10): Concrete locations with options\n"
        "   OBJECTS & PROPS (6-8): Tangible items with variations\n"
        "   EMOTIONAL/CONTEXTUAL (6-8): Visual concepts with synonyms\n\n"
        "OPTIMIZATION RULES:\n"
        "- Use domain-relevant terminology (healthcare, finance, sports, education, business) based on transcript context\n"
        "- Include 2-3 synonym variations per keyword for broader stock matching\n"
        "- If transcript is vague, generate universal but dynamic visuals (people interacting, bustling city, active nature)\n"
        "- Avoid static/generic clips (plain landscapes, empty streets)\n"
        "- Favor emotionally resonant, trend-adjacent terms for viral engagement\n"
        "- Use hierarchical format: {\"base\": \"keyword\", \"synonyms\": [\"syn1\", \"syn2\", \"syn3\"]}\n\n"
        "Output must be valid JSON. Do not include explanations, text, or formatting outside of the JSON.\n\n"
        f"Transcript:\n{text}\n\n"
        "JSON:"
    )
    
    # 🚀 NOUVEAU: Monitoring du prompt
    if log_prompt:
        print(f"    📝 [LLM] Prompt envoyé ({len(prompt)} caractères)")
        print(f"    🎯 [LLM] Modèle cible: {model}")
    
    # 🚀 NOUVEAU: Tentatives multiples avec validation JSON
    for attempt in range(json_retry_attempts):
        try:
            url = base.rstrip("/") + "/api/generate"
            payload = {"model": model, "prompt": prompt, "temperature": 0.7, "stream": False}
            
            # 🚀 NOUVEAU: Timeout plus long pour Ollama (génération complexe)
            print(f"    🤖 [LLM] Génération avec Ollama ({model}) - Tentative {attempt + 1}/{json_retry_attempts}...")
            
            start_time = time.time()
            r = requests.post(url, json=payload, timeout=120)  # 2 minutes au lieu de 60s
            end_time = time.time()
            r.raise_for_status()
            
            data = r.json() if r.headers.get("content-type","" ).startswith("application/json") else {"response": r.text}
            raw = data.get("response", "") if isinstance(data, dict) else ""
            
            if not raw:
                print(f"    ⚠️ [LLM] Réponse vide d'Ollama")
                if attempt < json_retry_attempts - 1:
                    print(f"    🔄 Nouvelle tentative dans 2s...")
                    time.sleep(2)
                    continue
                return None
            
            # 🚀 NOUVEAU: Monitoring de la réponse
            response_time = end_time - start_time
            if log_response_time:
                print(f"    ⏱️ [LLM] Temps de réponse: {response_time:.1f}s")
            if log_response_size:
                print(f"    📊 [LLM] Taille réponse: {len(raw)} caractères")
            
            # 🚀 NOUVEAU: Validation JSON stricte
            if enforce_json_output:
                try:
                    # Nettoyer la réponse pour extraire le JSON
                    raw_str = raw.strip()
                    sidx = raw_str.find("{")
                    eidx = raw_str.rfind("}")
                    if sidx != -1 and eidx != -1 and eidx > sidx:
                        raw_str = raw_str[sidx:eidx+1]
                    
                    # Tester si c'est du JSON valide
                    test_obj = _json.loads(raw_str)
                    if not isinstance(test_obj, dict):
                        raise ValueError("La réponse n'est pas un objet JSON valide")
                    
                    print(f"    ✅ [LLM] JSON valide détecté à la tentative {attempt + 1}")
                    break  # Sortir de la boucle si JSON valide
                    
                except (_json.JSONDecodeError, ValueError) as json_error:
                    print(f"    ❌ [LLM] JSON invalide à la tentative {attempt + 1}: {json_error}")
                    if attempt < json_retry_attempts - 1:
                        print(f"    🔄 Nouvelle tentative avec prompt renforcé...")
                        # Renforcer le prompt pour la prochaine tentative
                        prompt = prompt + "\n\nIMPORTANT: Output ONLY valid JSON. No explanations, no text outside JSON."
                        time.sleep(2)
                        continue
                    else:
                        print(f"    🚨 [LLM] Échec de validation JSON après {json_retry_attempts} tentatives")
                        return None
            else:
                print(f"    ✅ [LLM] Réponse reçue ({len(raw)} caractères)")
                break
                
        except Exception as e:
            print(f"    ❌ [LLM] Erreur à la tentative {attempt + 1}: {e}")
            if attempt < json_retry_attempts - 1:
                print(f"    🔄 Nouvelle tentative dans 2s...")
                time.sleep(2)
                continue
            else:
                print(f"    🚨 [LLM] Échec après {json_retry_attempts} tentatives")
                return None
    
    # Try parse JSON from raw (may include surrounding text)
    try:
        raw_str = raw.strip()
        # Find first and last braces to robustly extract JSON
        sidx = raw_str.find("{")
        eidx = raw_str.rfind("}")
        if sidx != -1 and eidx != -1 and eidx > sidx:
            raw_str = raw_str[sidx:eidx+1]
        obj = _json.loads(raw_str)
        
        title = (obj.get("title") or "").strip()
        description = (obj.get("description") or "").strip()
        tags = obj.get("hashtags") or []
        tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
        
        # 🚀 NOUVEAU: Extraction des mots-clés B-roll
        broll_keywords = obj.get("broll_keywords") or []
        broll_keywords = [kw.strip().lower() for kw in broll_keywords if isinstance(kw, str) and kw.strip()]
        
        # Validation et fallback pour les mots-clés B-roll
        if not broll_keywords:
            # Fallback: extraire des mots-clés basiques du titre et de la description
            fallback_text = f"{title} {description}".lower()
            fallback_keywords = [word for word in fallback_text.split() if len(word) > 3 and word.isalpha()]
            broll_keywords = list(set(fallback_keywords))[:10]  # Limiter à 10 mots-clés de fallback
        
        if not (title or description or tags):
            return None
        
        # 🚀 NOUVEAU: Retourner aussi les mots-clés B-roll
        return {
            "title": title, 
            "description": description, 
            "hashtags": tags,
            "broll_keywords": broll_keywords  # Nouveau champ
        }
        
    except Exception as e:
        print(f"⚠️ Erreur génération LLM: {e}")
        return None

if __name__ == "__main__":
    # Test de la fonction
    test_transcript = "EMDR movement sensation reprocessing lateralized movements people doing clinic"
    result = _llm_generate_caption_hashtags_fixed(test_transcript)
    if result:
        print("✅ Test réussi!")
        print(f"Title: {result.get('title')}")
        print(f"Description: {result.get('description')}")
        print(f"Hashtags: {result.get('hashtags')}")
        print(f"B-roll keywords: {result.get('broll_keywords')}")
    else:
        print("❌ Test échoué") 