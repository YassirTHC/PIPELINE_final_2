#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 NETTOYEUR JSON AUTOMATIQUE
Extrait et nettoie le JSON des réponses LLM
"""

import json
import re
import logging
from typing import Optional, Dict, Any

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONCleaner:
    """Classe pour nettoyer et valider les réponses JSON des LLM"""
    
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
            (r'^\s+|\s+$', ''),                # Supprimer espaces début/fin
        ]
    
    def clean_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Nettoie et parse la réponse LLM pour extraire le JSON valide
        
        Args:
            response_text: Réponse brute du LLM
            
        Returns:
            Dict JSON parsé ou None si échec
        """
        
        if not response_text or not response_text.strip():
            logger.warning("Réponse LLM vide")
            return None
        
        logger.info(f"Nettoyage de la réponse LLM: {len(response_text)} caractères")
        
        # 1. Tentative de parsing JSON direct
        try:
            parsed_json = json.loads(response_text)
            logger.info("✅ JSON direct valide détecté")
            return parsed_json
        except json.JSONDecodeError:
            logger.info("⚠️ JSON direct invalide, tentative de nettoyage...")
        
        # 2. Extraction du JSON du markdown
        extracted_json = self._extract_json_from_markdown(response_text)
        if extracted_json:
            try:
                parsed_json = json.loads(extracted_json)
                logger.info("✅ JSON extrait du markdown et validé")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON extrait invalide: {e}")
        
        # 3. Tentative de réparation JSON
        repaired_json = self._repair_json(response_text)
        if repaired_json:
            try:
                parsed_json = json.loads(repaired_json)
                logger.info("✅ JSON réparé et validé")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON réparé invalide: {e}")
        
        logger.error("❌ Impossible de nettoyer et valider le JSON")
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
        """Tente de réparer le JSON corrompu"""
        
        # Recherche de structures JSON partielles
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = text[json_start:json_end + 1]
            
            # Nettoyage des caractères problématiques
            for pattern, replacement in self.cleanup_patterns:
                json_str = re.sub(pattern, replacement, json_str)
            
            logger.info("Tentative de réparation JSON")
            return json_str
        
        return None
    
    def validate_keywords_response(self, parsed_json: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Valide une réponse de mots-clés
        
        Args:
            parsed_json: JSON parsé
            
        Returns:
            (valid, keywords_list)
        """
        
        if not isinstance(parsed_json, dict):
            logger.error("Réponse n'est pas un dictionnaire")
            return False, []
        
        if 'keywords' not in parsed_json:
            logger.error("Clé 'keywords' manquante")
            return False, []
        
        keywords = parsed_json['keywords']
        if not isinstance(keywords, list):
            logger.error("'keywords' n'est pas une liste")
            return False, []
        
        if len(keywords) < 3:
            logger.warning(f"Nombre de mots-clés insuffisant: {len(keywords)}")
            return False, []
        
        # Validation des mots-clés individuels
        valid_keywords = []
        for i, keyword in enumerate(keywords):
            if isinstance(keyword, str) and keyword.strip():
                valid_keywords.append(keyword.strip())
            else:
                logger.warning(f"Mots-clés {i} invalide: {keyword}")
        
        if len(valid_keywords) < 3:
            logger.error("Pas assez de mots-clés valides")
            return False, []
        
        logger.info(f"✅ {len(valid_keywords)} mots-clés valides trouvés")
        return True, valid_keywords
    
    def clean_and_validate(self, response_text: str) -> tuple[bool, list[str]]:
        """
        Méthode principale : nettoie et valide la réponse LLM
        
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
    """Fonction utilitaire pour valider les mots-clés"""
    return json_cleaner.clean_and_validate(response_text) 