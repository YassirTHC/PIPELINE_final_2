ï»¿#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ã°Å¸Â§Â¹ NETTOYEUR JSON AUTOMATIQUE
Extrait et nettoie le JSON des rÃƒÂ©ponses LLM
"""

import json
import re
import logging
from typing import Optional, Dict, Any

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONCleaner:
    """Classe pour nettoyer et valider les rÃƒÂ©ponses JSON des LLM"""
    
    def __init__(self):
        # Patterns pour extraire le JSON des blocs markdown
        self.json_patterns = [
            r'```json\s*(\{.*?\})\s*```',      # ```json {...} ```
            r'```\s*(\{.*?\})\s*```',          # ``` {...} ```
            r'`(\{.*?\})`',                    # `{...}`
            r'(\{.*?\})',                      # {...} (fallback)
        ]
        
        # Patterns pour nettoyer le JSON
        self.cleanup_patterns = [
            (r'\n\s*\n', ' '),                 # Supprimer les sauts de ligne multiples
            (r'\s+', ' '),                     # Normaliser les espaces
            (r'^\s+|\s+$', ''),                # Supprimer espaces dÃƒÂ©but/fin
        ]
    
    def clean_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Nettoie et parse la rÃƒÂ©ponse LLM pour extraire le JSON valide
        
        Args:
            response_text: RÃƒÂ©ponse brute du LLM
            
        Returns:
            Dict JSON parsÃƒÂ© ou None si ÃƒÂ©chec
        """
        
        if not response_text or not response_text.strip():
            logger.warning("RÃƒÂ©ponse LLM vide")
            return None
        
        logger.info(f"Nettoyage de la rÃƒÂ©ponse LLM: {len(response_text)} caractÃƒÂ¨res")
        
        # 1. Tentative de parsing JSON direct
        try:
            parsed_json = json.loads(response_text)
            logger.info("Ã¢Å“â€¦ JSON direct valide dÃƒÂ©tectÃƒÂ©")
            return parsed_json
        except json.JSONDecodeError:
            logger.info("Ã¢Å¡Â Ã¯Â¸Â JSON direct invalide, tentative de nettoyage...")
        
        # 2. Extraction du JSON du markdown
        extracted_json = self._extract_json_from_markdown(response_text)
        if extracted_json:
            try:
                parsed_json = json.loads(extracted_json)
                logger.info("Ã¢Å“â€¦ JSON extrait du markdown et validÃƒÂ©")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"Ã¢ÂÅ’ JSON extrait invalide: {e}")
        
        # 3. Tentative de rÃƒÂ©paration JSON
        repaired_json = self._repair_json(response_text)
        if repaired_json:
            try:
                parsed_json = json.loads(repaired_json)
                logger.info("Ã¢Å“â€¦ JSON rÃƒÂ©parÃƒÂ© et validÃƒÂ©")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"Ã¢ÂÅ’ JSON rÃƒÂ©parÃƒÂ© invalide: {e}")
        
        logger.error("Ã¢ÂÅ’ Impossible de nettoyer et valider le JSON")
        return None
    
    def _extract_json_from_markdown(self, text: str) -> Optional[str]:
        """Extrait le JSON des blocs markdown"""
        
        for pattern in self.json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1)
                logger.info(f"JSON extrait avec pattern: {pattern[:20]}...")
                return json_str
        
        return None
    
    def _repair_json(self, text: str) -> Optional[str]:
        """Tente de rÃƒÂ©parer le JSON corrompu"""
        
        # Recherche de structures JSON partielles
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = text[json_start:json_end + 1]
            
            # Nettoyage des caractÃƒÂ¨res problÃƒÂ©matiques
            for pattern, replacement in self.cleanup_patterns:
                json_str = re.sub(pattern, replacement, json_str)
            
            logger.info("Tentative de rÃƒÂ©paration JSON")
            return json_str
        
        return None
    
    def validate_keywords_response(self, parsed_json: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Valide une rÃƒÂ©ponse de mots-clÃƒÂ©s
        
        Args:
            parsed_json: JSON parsÃƒÂ©
            
        Returns:
            (valid, keywords_list)
        """
        
        if not isinstance(parsed_json, dict):
            logger.error("RÃƒÂ©ponse n'est pas un dictionnaire")
            return False, []
        
        if 'keywords' not in parsed_json:
            logger.error("ClÃƒÂ© 'keywords' manquante")
            return False, []
        
        keywords = parsed_json['keywords']
        if not isinstance(keywords, list):
            logger.error("'keywords' n'est pas une liste")
            return False, []
        
        if len(keywords) < 3:
            logger.warning(f"Nombre de mots-clÃƒÂ©s insuffisant: {len(keywords)}")
            return False, []
        
        # Validation des mots-clÃƒÂ©s individuels
        valid_keywords = []
        for i, keyword in enumerate(keywords):
            if isinstance(keyword, str) and keyword.strip():
                valid_keywords.append(keyword.strip())
            else:
                logger.warning(f"Mots-clÃƒÂ©s {i} invalide: {keyword}")
        
        if len(valid_keywords) < 3:
            logger.error("Pas assez de mots-clÃƒÂ©s valides")
            return False, []
        
        logger.info(f"Ã¢Å“â€¦ {len(valid_keywords)} mots-clÃƒÂ©s valides trouvÃƒÂ©s")
        return True, valid_keywords
    
    def clean_and_validate(self, response_text: str) -> tuple[bool, list[str]]:
        """
        MÃƒÂ©thode principale : nettoie et valide la rÃƒÂ©ponse LLM
        
        Returns:
            (success, keywords_list)
        """
        
        parsed_json = self.clean_llm_response(response_text)
        if not parsed_json:
            return False, []
        
        return self.validate_keywords_response(parsed_json)

# Instance globale pour utilisation facile
json_cleaner = JSONCleaner()

def clean_llm_json(response_text: str) -> Optional[Dict[str, Any]]:
    """Fonction utilitaire pour nettoyer le JSON LLM"""
    return json_cleaner.clean_llm_response(response_text)

def validate_keywords(response_text: str) -> tuple[bool, list[str]]:
    """Fonction utilitaire pour valider les mots-clÃƒÂ©s"""
    return json_cleaner.clean_and_validate(response_text) 

