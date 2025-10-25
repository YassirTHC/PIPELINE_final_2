ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ PIPELINE HYBRIDE ROBUSTE (Qwen3 + Llama2 + Retry + Validation)

import requests
import json
import time
import logging
import re
from typing import Dict, Any, Optional
from prompts_hybrides_etapes import get_prompt_etape_1, get_prompt_etape_2
from schema_validation_hybride import validate_etape_1, validate_etape_2, combine_etapes

# ========================================
# CONFIGURATION
# ========================================
TIMEOUT_ETAPE_1 = 300  # 5 minutes pour Qwen3:8B (rapide)
TIMEOUT_ETAPE_2 = 900  # 15 minutes pour Llama2:13B (lourd)
MAX_RETRIES = 3
BACKOFF_DELAY = 5  # DÃƒÂ©lai initial entre retries

# ========================================
# LOGGING
# ========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================================
# FONCTIONS D'EXTRACTION JSON INTELLIGENTE
# ========================================
def extract_json_from_response(response_text: str) -> Optional[str]:
    """
    Extrait le JSON valide d'une rÃƒÂ©ponse LLM mÃƒÂªme s'il contient du texte explicatif
    """
    if not response_text:
        return None
    
    # Nettoyage du texte
    cleaned = response_text.strip()
    
    # Tentative 1: JSON pur
    try:
        json.loads(cleaned)
        logger.info("Ã¢Å“â€¦ JSON pur dÃƒÂ©tectÃƒÂ©")
        return cleaned
    except:
        pass
    
    # Tentative 2: Recherche de JSON entre accolades
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, cleaned, re.DOTALL)
    
    if matches:
        # Prendre le dernier match (probablement le plus complet)
        last_match = matches[-1]
        try:
            json.loads(last_match)
            logger.info(f"Ã¢Å“â€¦ JSON extrait du texte (longueur: {len(last_match)} caractÃƒÂ¨res)")
            return last_match
        except:
            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Match trouvÃƒÂ© mais JSON invalide: {last_match[:100]}...")
    
    # Tentative 3: Recherche aprÃƒÂ¨s "JSON:" ou "Output:"
    for marker in ["JSON:", "Output:", "Response:", "Result:"]:
        if marker in cleaned:
            parts = cleaned.split(marker, 1)
            if len(parts) > 1:
                json_part = parts[1].strip()
                try:
                    json.loads(json_part)
                    logger.info(f"Ã¢Å“â€¦ JSON extrait aprÃƒÂ¨s '{marker}' (longueur: {len(json_part)} caractÃƒÂ¨res)")
                    return json_part
                except:
                    pass
    
    # Tentative 4: Recherche de la derniÃƒÂ¨re accolade ouvrante
    last_open = cleaned.rfind('{')
    if last_open != -1:
        try:
            json_part = cleaned[last_open:]
            json.loads(json_part)
            logger.info(f"Ã¢Å“â€¦ JSON extrait depuis la derniÃƒÂ¨re accolade (longueur: {len(json_part)} caractÃƒÂ¨res)")
            return json_part
        except:
            pass
    
    logger.error("Ã¢ÂÅ’ Impossible d'extraire du JSON valide")
    return None

# ========================================
# FONCTIONS LLM AVEC RETRY ET FALLBACK
# ========================================
def call_llm_with_fallback(model_primary: str, model_fallback: str, prompt: str, timeout: int) -> Optional[str]:
    """
    Appelle le LLM avec retry et fallback automatique
    """
    models_to_try = [model_primary, model_fallback]
    
    for model in models_to_try:
        logger.info(f"Ã°Å¸Â¤â€“ Tentative avec {model} (timeout: {timeout}s)")
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"   Ã°Å¸â€œÂ Tentative {attempt + 1}/{MAX_RETRIES}")
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "response" in result:
                        llm_response = result["response"]
                        logger.info(f"Ã¢Å“â€¦ SuccÃƒÂ¨s avec {model} en {timeout}s")
                        logger.info(f"Ã°Å¸â€œÂ RÃƒÂ©ponse brute: {len(llm_response)} caractÃƒÂ¨res")
                        
                        # Extraction intelligente du JSON
                        json_extracted = extract_json_from_response(llm_response)
                        if json_extracted:
                            logger.info(f"Ã°Å¸Å½Â¯ JSON extrait: {len(json_extracted)} caractÃƒÂ¨res")
                            return json_extracted
                        else:
                            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Impossible d'extraire du JSON de {model}")
                    else:
                        logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â RÃƒÂ©ponse invalide de {model}: {result}")
                else:
                    logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Erreur HTTP {response.status_code} avec {model}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Ã¢ÂÂ±Ã¯Â¸Â Timeout avec {model} (tentative {attempt + 1})")
                if attempt < MAX_RETRIES - 1:
                    delay = BACKOFF_DELAY * (2 ** attempt)
                    logger.info(f"   Ã¢ÂÂ³ Attente de {delay}s avant retry...")
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Ã¢ÂÅ’ Erreur avec {model}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_DELAY)
        
        logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Ãƒâ€°chec de tous les retries avec {model}")
    
    logger.error("Ã¢ÂÅ’ Ãƒâ€°chec de tous les modÃƒÂ¨les")
    return None

# ========================================
# PIPELINE HYBRIDE EN 2 Ãƒâ€°TAPES
# ========================================
def generate_etape_1(text: str) -> Dict[str, Any]:
    """
    Ãƒâ€°tape 1 : GÃƒÂ©nÃƒÂ©ration rapide avec Qwen3:8B (titres + hashtags)
    """
    logger.info("Ã°Å¸Å¡â‚¬ DÃƒâ€°BUT Ãƒâ€°TAPE 1: Titres + Hashtags")
    
    prompt = get_prompt_etape_1(text)
    logger.info(f"Ã°Å¸â€œÂ Prompt ÃƒÂ©tape 1: {len(prompt)} caractÃƒÂ¨res")
    
    # Tentative avec Qwen3:8B (rapide)
    result = call_llm_with_fallback(
        model_primary="qwen3:8b",
        model_fallback="qwen3:8b",
        prompt=prompt,
        timeout=TIMEOUT_ETAPE_1
    )
    
    if not result:
        return {"success": False, "error": "Aucun modÃƒÂ¨le n'a rÃƒÂ©ussi ÃƒÂ  gÃƒÂ©nÃƒÂ©rer une rÃƒÂ©ponse"}
    
    # Validation JSON
    validation = validate_etape_1(result)
    if not validation["success"]:
        return {"success": False, "error": f"JSON invalide: {validation['errors']}"}
    
    logger.info("Ã¢Å“â€¦ Ãƒâ€°TAPE 1 RÃƒâ€°USSIE")
    return {"success": True, "data": validation["data"]}

def generate_etape_2(text: str) -> Dict[str, Any]:
    """
    Ãƒâ€°tape 2 : GÃƒÂ©nÃƒÂ©ration lourde avec Llama2:13B (descriptions + B-roll keywords)
    """
    logger.info("Ã°Å¸Å¡â‚¬ DÃƒâ€°BUT Ãƒâ€°TAPE 2: Descriptions + B-roll Keywords")
    
    prompt = get_prompt_etape_2(text)
    logger.info(f"Ã°Å¸â€œÂ Prompt ÃƒÂ©tape 2: {len(prompt)} caractÃƒÂ¨res")
    
    # Tentative avec Llama2:13B (complet)
    result = call_llm_with_fallback(
        model_primary="qwen3:8b",
        model_fallback="qwen3:8b",
        prompt=prompt,
        timeout=TIMEOUT_ETAPE_2
    )
    
    if not result:
        return {"success": False, "error": "Aucun modÃƒÂ¨le n'a rÃƒÂ©ussi ÃƒÂ  gÃƒÂ©nÃƒÂ©rer une rÃƒÂ©ponse"}
    
    # Validation JSON
    validation = validate_etape_2(result)
    if not validation["success"]:
        return {"success": False, "error": f"JSON invalide: {validation['errors']}"}
    
    logger.info("Ã¢Å“â€¦ Ãƒâ€°TAPE 2 RÃƒâ€°USSIE")
    return {"success": True, "data": validation["data"]}

def pipeline_hybride_complet(text: str) -> Dict[str, Any]:
    """
    Pipeline hybride complet en 2 ÃƒÂ©tapes
    """
    logger.info("Ã°Å¸Å¡â‚¬ DÃƒâ€°BUT PIPELINE HYBRIDE COMPLET")
    logger.info(f"Ã°Å¸â€œÂ Transcript: {len(text)} caractÃƒÂ¨res")
    
    # Ãƒâ€°tape 1 : Titres + Hashtags (rapide)
    etape1_result = generate_etape_1(text)
    if not etape1_result["success"]:
        return {"success": False, "error": f"Ãƒâ€°tape 1 ÃƒÂ©chouÃƒÂ©e: {etape1_result['error']}"}
    
    # Ãƒâ€°tape 2 : Descriptions + B-roll Keywords (lourd)
    etape2_result = generate_etape_2(text)
    if not etape2_result["success"]:
        return {"success": False, "error": f"Ãƒâ€°tape 2 ÃƒÂ©chouÃƒÂ©e: {etape2_result['error']}"}
    
    # Combinaison et validation finale
    logger.info("Ã°Å¸â€â€” Combinaison des deux ÃƒÂ©tapes...")
    final_result = combine_etapes(etape1_result["data"], etape2_result["data"])
    
    if not final_result["success"]:
        return {"success": False, "error": f"Combinaison ÃƒÂ©chouÃƒÂ©e: {final_result['errors']}"}
    
    logger.info("Ã°Å¸Å½â€° PIPELINE HYBRIDE RÃƒâ€°USSI !")
    return {"success": True, "data": final_result["data"]}

# ========================================
# FONCTIONS UTILITAIRES
# ========================================
def get_pipeline_stats() -> Dict[str, Any]:
    """Retourne les statistiques du pipeline"""
    return {
        "timeouts": {
            "etape_1": f"{TIMEOUT_ETAPE_1}s (Qwen3:8B rapide)",
            "etape_2": f"{TIMEOUT_ETAPE_2}s (Llama2:13B lourd)"
        },
        "retry_config": {
            "max_retries": MAX_RETRIES,
            "backoff_delay": f"{BACKOFF_DELAY}s",
            "total_attempts": MAX_RETRIES * 2  # 2 modÃƒÂ¨les
        },
        "fallback_strategy": {
            "etape_1": "Qwen3:8B Ã¢â€ â€™ Llama2:13B",
            "etape_2": "Llama2:13B Ã¢â€ â€™ Qwen3:8B"
        },
        "json_extraction": "Intelligente (texte + JSON, JSON pur, patterns)"
    }

# ========================================
# TEST DU PIPELINE
# ========================================
if __name__ == "__main__":
    # Test avec un transcript court
    test_transcript = "EMDR movement sensation reprocessing lateralized movements people doing clinic got goofy looking things"
    
    print("Ã°Å¸Å¡â‚¬ TEST DU PIPELINE HYBRIDE ROBUSTE AVEC EXTRACTION JSON INTELLIGENTE")
    print("=" * 70)
    
    stats = get_pipeline_stats()
    print("Ã°Å¸â€œÅ  Configuration:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()
    
    # Test du pipeline
    result = pipeline_hybride_complet(test_transcript)
    
    if result["success"]:
        print("Ã¢Å“â€¦ SUCCÃƒË†S DU PIPELINE !")
        print("Ã°Å¸â€œâ€¹ RÃƒÂ©sultat final:")
        data = result["data"]
        print(f"   Titres: {len(data['title'])}")
        print(f"   Descriptions: {len(data['description'])}")
        print(f"   Hashtags: {len(data['hashtags'])}")
        print(f"   B-roll keywords: {len(data['broll_keywords'])}")
    else:
        print(f"Ã¢ÂÅ’ Ãƒâ€°CHEC DU PIPELINE: {result['error']}") 

