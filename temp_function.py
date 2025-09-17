#!/usr/bin/env python3
import time
import json as _json
import requests
from typing import Optional, Dict
import os

def _llm_generate_caption_hashtags_fixed(transcript_text: str) -> Optional[Dict[str, object]]:
    try:
        import yaml
        with open("config/llm_config.yaml", "r", encoding="utf-8") as f:
            llm_config = yaml.safe_load(f) or {}
        llm_section = llm_config.get("llm", {}) or {}
        model = llm_section.get("model", "gemma3:4b")
        request_timeout = int(llm_section.get("request_timeout", 120))
    except:
        model = "gemma3:4b"
        request_timeout = 120
    
    base = "http://127.0.0.1:11434"
    text = (transcript_text or "").strip()[:1500]
    
    prompt = (
        "Analyze the transcript and generate comprehensive content metadata. Output ONLY valid JSON:\n\n"
        "{\n"
        '  "domain": "[auto-detected domain/category]",\n'
        '  "context": "[specific context within domain]",\n'
        '  "title": "[viral emoji] [catchy title 60 chars max]",\n'
        '  "description": "[engaging description with emojis + CTA]",\n'
        '  "hashtags": ["#domain", "#context", "#keyword1", ...],\n'
        '  "broll_keywords": ["specific_visual_term1", "concrete_action2", ...],\n'
        '  "search_queries": ["2-4 word search phrase1", "visual term phrase2", ...]\n'
        "}\n\n"
        "DOMAIN DETECTION: Analyze the content and identify the specific domain/field (e.g., medical_therapy, business_strategy, technology_ai, fitness_training, cooking_tutorial, science_education, etc.). Be specific and create new domains as needed.\n\n"
        "CONTEXT ANALYSIS: Within the domain, identify the specific context or sub-topic being discussed.\n\n"
        "B-ROLL REQUIREMENTS:\n"
        "- Generate 15-25 VISUALLY-SPECIFIC keywords that represent concrete, searchable visual elements\n"
        "- Focus on: specific actions, recognizable objects, professional settings, identifiable people types\n"
        "- Examples: 'doctor_examining_patient' not 'medical_care', 'chef_chopping_vegetables' not 'cooking'\n"
        "- Avoid abstract concepts, emotions, or generic terms\n\n"
        "SEARCH QUERIES: Create 8-12 ready-to-use search phrases (2-4 words each) optimized for stock footage APIs.\n\n"
        "LANGUAGE: All output in English for optimal understanding.\n\n"
        "Transcript: " + text + "\n\nJSON:"
    )
    
    print(f"     [LLM] Prompt envoyé ({len(prompt)} caractères)")
    print(f"     [LLM] Modèle cible: {model}")
    
    try:
        payload = {"model": model, "prompt": prompt, "temperature": 0.1, "stream": False, "format": "json"}
        print(f"     [LLM] Génération avec Ollama ({model})...")
        
        start_time = time.time()
        r = requests.post(f"{base}/api/generate", json=payload, timeout=request_timeout)
        end_time = time.time()
        r.raise_for_status()
        
        data = r.json()
        raw = data.get("response", "")
        
        print(f"     [LLM] Temps de réponse: {end_time - start_time:.1f}s")
        print(f"     [LLM] Taille réponse: {len(raw)} caractères")
        
        # Extract JSON
        raw_str = raw.strip()
        sidx = raw_str.find("{")
        eidx = raw_str.rfind("}")
        if sidx != -1 and eidx != -1 and eidx > sidx:
            raw_str = raw_str[sidx:eidx+1]
        
        obj = _json.loads(raw_str)
        
        # Extraction des nouveaux champs
        domain = (obj.get("domain") or "").strip()
        context = (obj.get("context") or "").strip()
        title = (obj.get("title") or "").strip()
        description = (obj.get("description") or "").strip()
        tags = [t.strip() for t in (obj.get("hashtags") or []) if isinstance(t, str) and t.strip()]
        broll_keywords = [kw.strip().lower() for kw in (obj.get("broll_keywords") or []) if isinstance(kw, str) and kw.strip()]
        search_queries = [q.strip() for q in (obj.get("search_queries") or []) if isinstance(q, str) and q.strip()]
        
        if not (title or description or tags):
            return None
        
        print(f"     [LLM] JSON valide - Domaine: {domain}, Contexte: {context}")
        print(f"     [LLM] B-roll: {len(broll_keywords)} mots-clés, {len(search_queries)} requêtes")
        
        return {
            "domain": domain,
            "context": context,
            "title": title, 
            "description": description, 
            "hashtags": tags,
            "broll_keywords": broll_keywords,
            "search_queries": search_queries
        }
        
    except Exception as e:
        print(f"     [LLM] Erreur: {e}")
        return None

if __name__ == "__main__":
    test_transcript = "EMDR movement sensation reprocessing lateralized movements people doing clinic"
    result = _llm_generate_caption_hashtags_fixed(test_transcript)
    if result:
        print(" Test réussi!")
        print(f"Title: {result.get('title')}")
        print(f"Description: {result.get('description')}")
        print(f"Hashtags: {result.get('hashtags')}")
        print(f"B-roll keywords: {result.get('broll_keywords')}")
