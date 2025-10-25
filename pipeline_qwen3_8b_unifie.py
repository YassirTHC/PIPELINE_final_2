ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ PIPELINE UNIFIÃƒâ€° QWEN3:8B + AUTO-CORRECTION

import requests
import json
import time
import logging
import re
from typing import Dict, Any, Optional, List
from prompts_hybrides_etapes import get_prompt_etape_1, get_prompt_etape_2
from schema_validation_hybride import validate_etape_1, validate_etape_2, combine_etapes

# ========================================
# CONFIGURATION UNIFIÃƒâ€°E QWEN3:8B
# ========================================
TIMEOUT_QWEN3_8B = 300  # 5 minutes (Qwen3:8B est rapide)
MAX_RETRIES = 3
BACKOFF_DELAY = 2  # DÃƒÂ©lai court entre retries

# ========================================
# LOGGING
# ========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================================
# FONCTIONS DE NORMALISATION JSON AUTOMATIQUE
# ========================================
def normalize_json_format_etape1(raw_json: str) -> str:
    """
    Normalise le format JSON de l'ÃƒÂ©tape 1 (titres + hashtags)
    Convertit le format Qwen3:8B vers le format attendu par Pydantic
    """
    try:
        data = json.loads(raw_json)
        logger.info("Ã°Å¸â€Â§ Normalisation JSON ÃƒÂ©tape 1...")
        
        # Normaliser les titres
        if "title" in data:
            if isinstance(data["title"], str):
                # Si c'est un string, le convertir en liste
                data["title"] = [data["title"]]
            elif not isinstance(data["title"], list):
                data["title"] = []
        
        # Normaliser les hashtags
        if "hashtags" in data:
            if isinstance(data["hashtags"], str):
                # Si c'est un string, essayer de l'extraire
                hashtags = re.findall(r'#\w+', data["hashtags"])
                data["hashtags"] = hashtags if hashtags else []
            elif not isinstance(data["hashtags"], list):
                data["hashtags"] = []
        
        # S'assurer qu'on a au moins les clÃƒÂ©s minimales
        if "title" not in data:
            data["title"] = []
        if "hashtags" not in data:
            data["hashtags"] = []
        
        normalized = json.dumps(data, ensure_ascii=False)
        logger.info(f"Ã¢Å“â€¦ JSON normalisÃƒÂ© ÃƒÂ©tape 1: {len(normalized)} caractÃƒÂ¨res")
        return normalized
        
    except Exception as e:
        logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Erreur normalisation ÃƒÂ©tape 1: {e}")
        return raw_json

def normalize_json_format_etape2(raw_json: str) -> str:
    """
    Normalise le format JSON de l'ÃƒÂ©tape 2 (descriptions + B-roll keywords)
    Convertit le format Qwen3:8B vers le format attendu par Pydantic
    """
    try:
        data = json.loads(raw_json)
        logger.info("Ã°Å¸â€Â§ Normalisation JSON ÃƒÂ©tape 2...")
        
        # Normaliser les descriptions
        if "description" in data:
            if isinstance(data["description"], str):
                # Si c'est un string, le convertir en liste
                data["description"] = [data["description"]]
            elif not isinstance(data["description"], list):
                data["description"] = []
        
        # Normaliser les B-roll keywords
        if "broll_keywords" in data:
            if isinstance(data["broll_keywords"], dict):
                # Si c'est un dict, le convertir en liste de catÃƒÂ©gories
                normalized_keywords = []
                for category, keywords in data["broll_keywords"].items():
                    if isinstance(keywords, list):
                        for keyword in keywords:
                            if isinstance(keyword, str):
                                # CrÃƒÂ©er un objet structurÃƒÂ©
                                normalized_keywords.append({
                                    "category": category,
                                    "base": keyword,
                                    "synonyms": [f"{keyword}_syn1", f"{keyword}_syn2", f"{keyword}_syn3", f"{keyword}_syn4"]
                                })
                            elif isinstance(keyword, dict) and "base" in keyword:
                                # DÃƒÂ©jÃƒÂ  structurÃƒÂ©, ajouter la catÃƒÂ©gorie si manquante
                                if "category" not in keyword:
                                    keyword["category"] = category
                                normalized_keywords.append(keyword)
                
                data["broll_keywords"] = normalized_keywords
            elif not isinstance(data["broll_keywords"], list):
                data["broll_keywords"] = []
        
        # S'assurer qu'on a au moins les clÃƒÂ©s minimales
        if "description" not in data:
            data["description"] = []
        if "broll_keywords" not in data:
            data["broll_keywords"] = []
        
        normalized = json.dumps(data, ensure_ascii=False)
        logger.info(f"Ã¢Å“â€¦ JSON normalisÃƒÂ© ÃƒÂ©tape 2: {len(normalized)} caractÃƒÂ¨res")
        return normalized
        
    except Exception as e:
        logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Erreur normalisation ÃƒÂ©tape 2: {e}")
        return raw_json

# ========================================
# FONCTIONS D'EXTRACTION JSON INTELLIGENTE
# ========================================
def extract_json_from_response(response_text: str) -> Optional[str]:
    """Extrait le JSON valide d'une rÃƒÂ©ponse LLM mÃƒÂªme s'il contient du texte explicatif"""
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
# FONCTIONS LLM QWEN3:8B UNIFIÃƒâ€°ES
# ========================================
def call_qwen3_8b(prompt: str, timeout: int = TIMEOUT_QWEN3_8B) -> Optional[str]:
    """Appelle Qwen3:8B avec retry automatique"""
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Ã°Å¸Â¤â€“ Qwen3:8B - Tentative {attempt + 1}/{MAX_RETRIES}")
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3:8b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if "response" in result:
                    llm_response = result["response"]
                    logger.info(f"Ã¢Å“â€¦ Qwen3:8B rÃƒÂ©pond en {timeout}s")
                    logger.info(f"Ã°Å¸â€œÂ RÃƒÂ©ponse brute: {len(llm_response)} caractÃƒÂ¨res")
                    
                    # Extraction intelligente du JSON
                    json_extracted = extract_json_from_response(llm_response)
                    if json_extracted:
                        logger.info(f"Ã°Å¸Å½Â¯ JSON extrait: {len(json_extracted)} caractÃƒÂ¨res")
                        return json_extracted
                    else:
                        logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Impossible d'extraire du JSON de Qwen3:8B")
                else:
                    logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â RÃƒÂ©ponse invalide de Qwen3:8B: {result}")
            else:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Erreur HTTP {response.status_code} avec Qwen3:8B")
                
        except requests.exceptions.Timeout:
            logger.warning(f"Ã¢ÂÂ±Ã¯Â¸Â Timeout avec Qwen3:8B (tentative {attempt + 1})")
            if attempt < MAX_RETRIES - 1:
                delay = BACKOFF_DELAY * (2 ** attempt)
                logger.info(f"   Ã¢ÂÂ³ Attente de {delay}s avant retry...")
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur avec Qwen3:8B: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_DELAY)
    
    logger.error("Ã¢ÂÅ’ Ãƒâ€°chec de tous les retries avec Qwen3:8B")
    return None

# ========================================
# PROMPTS DE COMPLÃƒâ€°TION AUTO-CORRECTIVE
# ========================================
def get_completion_prompt_etape1(missing_titles: int, missing_hashtags: int) -> str:
    """Prompt de complÃƒÂ©tion pour l'ÃƒÂ©tape 1"""
    return f"""Tu dois complÃƒÂ©ter le JSON existant avec les ÃƒÂ©lÃƒÂ©ments manquants.

Ãƒâ€°LÃƒâ€°MENTS MANQUANTS Ãƒâ‚¬ AJOUTER:
- Titres: {missing_titles} de plus
- Hashtags: {missing_hashtags} de plus

GÃƒÂ©nÃƒÂ¨re UNIQUEMENT le JSON complet avec tous les ÃƒÂ©lÃƒÂ©ments.
Format: {{"title": ["titre1", "titre2", ...], "hashtags": ["#tag1", "#tag2", ...]}}

JSON:"""

def get_completion_prompt_etape2(missing_keywords: int, category: str) -> str:
    """Prompt de complÃƒÂ©tion pour l'ÃƒÂ©tape 2"""
    return f"""Tu dois complÃƒÂ©ter le JSON existant avec des mots-clÃƒÂ©s B-roll manquants.

CATÃƒâ€°GORIE: {category}
MOTS-CLÃƒâ€°S MANQUANTS: {missing_keywords} de plus

Format pour chaque mot-clÃƒÂ©:
{{"category": "{category}", "base": "mot_clÃƒÂ©", "synonyms": ["syn1", "syn2", "syn3", "syn4"]}}

GÃƒÂ©nÃƒÂ¨re UNIQUEMENT le JSON avec les mots-clÃƒÂ©s manquants pour cette catÃƒÂ©gorie.

JSON:"""

# ========================================
# PIPELINE UNIFIÃƒâ€° QWEN3:8B AVEC AUTO-CORRECTION
# ========================================
def generate_etape_1_with_correction(text: str) -> Dict[str, Any]:
    """Ãƒâ€°tape 1 avec auto-correction automatique"""
    logger.info("Ã°Å¸Å¡â‚¬ DÃƒâ€°BUT Ãƒâ€°TAPE 1: Titres + Hashtags (Qwen3:8B)")
    
    prompt = get_prompt_etape_1(text)
    logger.info(f"Ã°Å¸â€œÂ Prompt ÃƒÂ©tape 1: {len(prompt)} caractÃƒÂ¨res")
    
    # GÃƒÂ©nÃƒÂ©ration initiale
    result = call_qwen3_8b(prompt)
    if not result:
        return {"success": False, "error": "Qwen3:8B n'a pas rÃƒÂ©ussi ÃƒÂ  gÃƒÂ©nÃƒÂ©rer une rÃƒÂ©ponse"}
    
    # Normalisation JSON automatique
    result = normalize_json_format_etape1(result)
    
    # Validation et auto-correction
    validation = validate_etape_1(result)
    if validation["success"]:
        logger.info("Ã¢Å“â€¦ Ãƒâ€°TAPE 1 RÃƒâ€°USSIE du premier coup")
        return {"success": True, "data": validation["data"]}
    
    # Auto-correction nÃƒÂ©cessaire
    logger.info("Ã°Å¸â€â€ž Auto-correction nÃƒÂ©cessaire pour l'ÃƒÂ©tape 1")
    
    try:
        # Parser le JSON existant pour voir ce qui manque
        existing_data = json.loads(result)
        current_titles = len(existing_data.get("title", []))
        current_hashtags = len(existing_data.get("hashtags", []))
        
        missing_titles = max(0, 3 - current_titles)  # Minimum 3 titres
        missing_hashtags = max(0, 10 - current_hashtags)  # Minimum 10 hashtags
        
        if missing_titles > 0 or missing_hashtags > 0:
            logger.info(f"Ã°Å¸â€Â§ ComplÃƒÂ©tion: {missing_titles} titres, {missing_hashtags} hashtags")
            
            completion_prompt = get_completion_prompt_etape1(missing_titles, missing_hashtags)
            completion_result = call_qwen3_8b(completion_prompt)
            
            if completion_result:
                # Fusionner les rÃƒÂ©sultats
                try:
                    completion_data = json.loads(completion_result)
                    # Logique de fusion (simplifiÃƒÂ©e pour l'exemple)
                    final_data = {
                        "title": existing_data.get("title", []) + completion_data.get("title", []),
                        "hashtags": existing_data.get("hashtags", []) + completion_data.get("hashtags", [])
                    }
                    
                    # Validation finale
                    final_validation = validate_etape_1(json.dumps(final_data))
                    if final_validation["success"]:
                        logger.info("Ã¢Å“â€¦ Ãƒâ€°TAPE 1 RÃƒâ€°USSIE aprÃƒÂ¨s auto-correction")
                        return {"success": True, "data": final_validation["data"]}
                except:
                    pass
    
    except:
        pass
    
    return {"success": False, "error": f"Auto-correction ÃƒÂ©chouÃƒÂ©e: {validation['errors']}"}

def generate_etape_2_with_correction(text: str) -> Dict[str, Any]:
    """Ãƒâ€°tape 2 avec auto-correction automatique"""
    logger.info("Ã°Å¸Å¡â‚¬ DÃƒâ€°BUT Ãƒâ€°TAPE 2: Descriptions + B-roll Keywords (Qwen3:8B)")
    
    prompt = get_prompt_etape_2(text)
    logger.info(f"Ã°Å¸â€œÂ Prompt ÃƒÂ©tape 2: {len(prompt)} caractÃƒÂ¨res")
    
    # GÃƒÂ©nÃƒÂ©ration initiale
    result = call_qwen3_8b(prompt)
    if not result:
        return {"success": False, "error": "Qwen3:8B n'a pas rÃƒÂ©ussi ÃƒÂ  gÃƒÂ©nÃƒÂ©rer une rÃƒÂ©ponse"}
    
    # Normalisation JSON automatique
    result = normalize_json_format_etape2(result)
    
    # Validation et auto-correction
    validation = validate_etape_2(result)
    if validation["success"]:
        logger.info("Ã¢Å“â€¦ Ãƒâ€°TAPE 2 RÃƒâ€°USSIE du premier coup")
        return {"success": True, "data": validation["data"]}
    
    # Auto-correction nÃƒÂ©cessaire
    logger.info("Ã°Å¸â€â€ž Auto-correction nÃƒÂ©cessaire pour l'ÃƒÂ©tape 2")
    
    try:
        # Parser le JSON existant pour voir ce qui manque
        existing_data = json.loads(result)
        current_keywords = len(existing_data.get("broll_keywords", []))
        
        if current_keywords < 25:
            missing_keywords = 25 - current_keywords
            logger.info(f"Ã°Å¸â€Â§ ComplÃƒÂ©tion: {missing_keywords} mots-clÃƒÂ©s B-roll manquants")
            
            # ComplÃƒÂ©ter par catÃƒÂ©gorie
            categories = ["VISUAL ACTIONS", "PEOPLE & ROLES", "ENVIRONMENTS & PLACES", "OBJECTS & PROPS", "EMOTIONAL/CONTEXTUAL"]
            
            for category in categories:
                category_count = len([k for k in existing_data.get("broll_keywords", []) if k.get("category") == category])
                if category_count < 5:
                    missing_in_category = 5 - category_count
                    logger.info(f"Ã°Å¸â€Â§ ComplÃƒÂ©tion catÃƒÂ©gorie {category}: {missing_in_category} mots-clÃƒÂ©s")
                    
                    completion_prompt = get_completion_prompt_etape2(missing_in_category, category)
                    completion_result = call_qwen3_8b(completion_prompt)
                    
                    if completion_result:
                        try:
                            completion_data = json.loads(completion_result)
                            existing_data["broll_keywords"].extend(completion_data)
                        except:
                            pass
            
            # Validation finale
            final_validation = validate_etape_2(json.dumps(existing_data))
            if final_validation["success"]:
                logger.info("Ã¢Å“â€¦ Ãƒâ€°TAPE 2 RÃƒâ€°USSIE aprÃƒÂ¨s auto-correction")
                return {"success": True, "data": final_validation["data"]}
    
    except:
        pass
    
    return {"success": False, "error": f"Auto-correction ÃƒÂ©chouÃƒÂ©e: {validation['errors']}"}

def pipeline_qwen3_8b_unifie(text: str) -> Dict[str, Any]:
    """Pipeline unifiÃƒÂ© Qwen3:8B avec auto-correction"""
    logger.info("Ã°Å¸Å¡â‚¬ DÃƒâ€°BUT PIPELINE UNIFIÃƒâ€° QWEN3:8B + AUTO-CORRECTION")
    logger.info(f"Ã°Å¸â€œÂ Transcript: {len(text)} caractÃƒÂ¨res")
    
    # Ãƒâ€°tape 1 : Titres + Hashtags avec auto-correction
    etape1_result = generate_etape_1_with_correction(text)
    if not etape1_result["success"]:
        return {"success": False, "error": f"Ãƒâ€°tape 1 ÃƒÂ©chouÃƒÂ©e: {etape1_result['error']}"}
    
    # Ãƒâ€°tape 2 : Descriptions + B-roll Keywords avec auto-correction
    etape2_result = generate_etape_2_with_correction(text)
    if not etape2_result["success"]:
        return {"success": False, "error": f"Ãƒâ€°tape 2 ÃƒÂ©chouÃƒÂ©e: {etape2_result['error']}"}
    
    # Combinaison et validation finale
    logger.info("Ã°Å¸â€â€” Combinaison des deux ÃƒÂ©tapes...")
    final_result = combine_etapes(etape1_result["data"], etape2_result["data"])
    
    if not final_result["success"]:
        return {"success": False, "error": f"Combinaison ÃƒÂ©chouÃƒÂ©e: {final_result['errors']}"}
    
    logger.info("Ã°Å¸Å½â€° PIPELINE UNIFIÃƒâ€° QWEN3:8B RÃƒâ€°USSI !")
    return {"success": True, "data": final_result["data"]}

# ========================================
# FONCTIONS UTILITAIRES
# ========================================
def get_pipeline_stats() -> Dict[str, Any]:
    """Retourne les statistiques du pipeline"""
    return {
        "modele_unique": "qwen3:8b",
        "timeout": f"{TIMEOUT_QWEN3_8B}s (Qwen3:8B rapide)",
        "retry_config": {
            "max_retries": MAX_RETRIES,
            "backoff_delay": f"{BACKOFF_DELAY}s",
            "total_attempts": MAX_RETRIES
        },
        "auto_correction": "ActivÃƒÂ©e pour toutes les ÃƒÂ©tapes",
        "validation": "Pydantic strict + complÃƒÂ©tion automatique",
        "normalisation": "JSON automatique Qwen3:8B Ã¢â€ â€™ Format attendu"
    }

# ========================================
# TEST DU PIPELINE
# ========================================
if __name__ == "__main__":
    # Test avec un transcript court
    test_transcript = "EMDR movement sensation reprocessing lateralized movements people doing clinic got goofy looking things"
    
    print("Ã°Å¸Å¡â‚¬ TEST PIPELINE UNIFIÃƒâ€° QWEN3:8B + AUTO-CORRECTION + NORMALISATION JSON")
    print("=" * 75)
    
    stats = get_pipeline_stats()
    print("Ã°Å¸â€œÅ  Configuration:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()
    
    # Test du pipeline
    result = pipeline_qwen3_8b_unifie(test_transcript)
    
    if result["success"]:
        print("Ã¢Å“â€¦ SUCCÃƒË†S DU PIPELINE UNIFIÃƒâ€° !")
        print("Ã°Å¸â€œâ€¹ RÃƒÂ©sultat final:")
        data = result["data"]
        print(f"   Titres: {len(data['title'])}")
        print(f"   Descriptions: {len(data['description'])}")
        print(f"   Hashtags: {len(data['hashtags'])}")
        print(f"   B-roll keywords: {len(data['broll_keywords'])}")
    else:
        print(f"Ã¢ÂÅ’ Ãƒâ€°CHEC DU PIPELINE: {result['error']}") 

