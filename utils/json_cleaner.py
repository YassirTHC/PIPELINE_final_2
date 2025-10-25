#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ðŸ§¹ NETTOYEUR JSON AUTOMATIQUE
Extrait et nettoie le JSON des rÃ©ponses LLM
"""

import json
import re
import logging
from typing import Optional, Dict, Any

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONCleaner:
    """Classe pour nettoyer et valider les rÃ©ponses JSON des LLM"""
    
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
            (r'^\s+|\s+$', ''),                # Supprimer espaces dÃ©but/fin
        ]
    
    def clean_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Nettoie et parse la rÃ©ponse LLM pour extraire le JSON valide
        
        Args:
            response_text: RÃ©ponse brute du LLM
            
        Returns:
            Dict JSON parsÃ© ou None si Ã©chec
        """
        
        if not response_text or not response_text.strip():
            logger.warning("RÃ©ponse LLM vide")
            return None
        
        logger.info(f"Nettoyage de la rÃ©ponse LLM: {len(response_text)} caractÃ¨res")
        
        # 1. Tentative de parsing JSON direct
        try:
            parsed_json = json.loads(response_text)
            logger.info("âœ… JSON direct valide dÃ©tectÃ©")
            return parsed_json
        except json.JSONDecodeError:
            logger.info("âš ï¸ JSON direct invalide, tentative de nettoyage...")
        
        # 2. Extraction du JSON du markdown
        extracted_json = self._extract_json_from_markdown(response_text)
        if extracted_json:
            try:
                parsed_json = json.loads(extracted_json)
                logger.info("âœ… JSON extrait du markdown et validÃ©")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"âŒ JSON extrait invalide: {e}")
        
        # 3. Tentative de rÃ©paration JSON
        repaired_json = self._repair_json(response_text)
        if repaired_json:
            try:
                parsed_json = json.loads(repaired_json)
                logger.info("âœ… JSON rÃ©parÃ© et validÃ©")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"âŒ JSON rÃ©parÃ© invalide: {e}")
        
        logger.error("âŒ Impossible de nettoyer et valider le JSON")
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
        """Tente de rÃ©parer le JSON corrompu"""
        
        # Recherche de structures JSON partielles
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = text[json_start:json_end + 1]
            
            # Nettoyage des caractÃ¨res problÃ©matiques
            for pattern, replacement in self.cleanup_patterns:
                json_str = re.sub(pattern, replacement, json_str)
            
            logger.info("Tentative de rÃ©paration JSON")
            return json_str
        
        return None
    
    def validate_keywords_response(self, parsed_json: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Valide une rÃ©ponse de mots-clÃ©s
        
        Args:
            parsed_json: JSON parsÃ©
            
        Returns:
            (valid, keywords_list)
        """
        
        if not isinstance(parsed_json, dict):
            logger.error("RÃ©ponse n'est pas un dictionnaire")
            return False, []
        
        if 'keywords' not in parsed_json:
            logger.error("ClÃ© 'keywords' manquante")
            return False, []
        
        keywords = parsed_json['keywords']
        if not isinstance(keywords, list):
            logger.error("'keywords' n'est pas une liste")
            return False, []
        
        if len(keywords) < 3:
            logger.warning(f"Nombre de mots-clÃ©s insuffisant: {len(keywords)}")
            return False, []
        
        # Validation des mots-clÃ©s individuels
        valid_keywords = []
        for i, keyword in enumerate(keywords):
            if isinstance(keyword, str) and keyword.strip():
                valid_keywords.append(keyword.strip())
            else:
                logger.warning(f"Mots-clÃ©s {i} invalide: {keyword}")
        
        if len(valid_keywords) < 3:
            logger.error("Pas assez de mots-clÃ©s valides")
            return False, []
        
        logger.info(f"âœ… {len(valid_keywords)} mots-clÃ©s valides trouvÃ©s")
        return True, valid_keywords
    
    def clean_and_validate(self, response_text: str) -> tuple[bool, list[str]]:
        """
        MÃ©thode principale : nettoie et valide la rÃ©ponse LLM
        
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
    """Fonction utilitaire pour valider les mots-clÃ©s"""
    return json_cleaner.clean_and_validate(response_text) 
