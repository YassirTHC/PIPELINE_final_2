ï»¿#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ã°Å¸Å¡â‚¬ SYSTÃƒË†ME LLM MINIMALISTE - PROMPTS GÃƒâ€°NÃƒâ€°RIQUES + SPÃƒâ€°CIALISATION PIPELINE
BasÃƒÂ© sur l'analyse brillante de l'utilisateur : prompts simples + spÃƒÂ©cialisation intelligente
"""

import os
import requests
import json
import time
import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple, Any
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_int_env(name: str, default: int, *, minimum: int = 0) -> int:
    value = os.getenv(name)
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _parse_float_env(
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
) -> float:
    value = os.getenv(name)
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    if maximum is not None:
        parsed = min(maximum, parsed)
    return max(minimum, parsed)


def _parse_stop_tokens_env(name: str, default: Sequence[str]) -> List[str]:
    value = os.getenv(name)
    if not value:
        return list(default)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, (list, tuple)):
            return [str(token) for token in parsed if str(token)]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    tokens = [token.strip() for token in value.split("|") if token.strip()]
    if tokens:
        return tokens
    return list(default)


_DEFAULT_STOP_TOKENS: Tuple[str, ...] = ("```", "\n\n\n", "END_OF_CONTEXT", "</json>")

class OptimizedLLM:
    """SystÃƒÂ¨me LLM avec prompts minimalistes et spÃƒÂ©cialisation via pipeline"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:4b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = 60  # Timeout plus court pour dÃƒÂ©tecter rapidement les blocages
        self.num_predict = _parse_int_env("PIPELINE_LLM_NUM_PREDICT", 256, minimum=1)
        self.temperature = _parse_float_env("PIPELINE_LLM_TEMP", 0.1, minimum=0.0)
        self.top_p = _parse_float_env("PIPELINE_LLM_TOP_P", 0.9, minimum=0.0, maximum=1.0)
        self.repeat_penalty = _parse_float_env("PIPELINE_LLM_REPEAT_PENALTY", 1.1, minimum=0.0)
        self.stop: List[str] = _parse_stop_tokens_env("PIPELINE_LLM_STOP_TOKENS", _DEFAULT_STOP_TOKENS)

    def configure_generation(
        self,
        *,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> None:
        if num_predict is not None:
            try:
                self.num_predict = max(1, int(num_predict))
            except (TypeError, ValueError):
                pass
        if temperature is not None:
            try:
                self.temperature = float(temperature)
            except (TypeError, ValueError):
                pass
        if top_p is not None:
            try:
                self.top_p = float(top_p)
            except (TypeError, ValueError):
                pass
        if repeat_penalty is not None:
            try:
                self.repeat_penalty = float(repeat_penalty)
            except (TypeError, ValueError):
                pass
        if stop is not None:
            try:
                self.stop = [str(token) for token in stop if str(token)]
            except Exception:
                pass

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 100,
        *,
        timeout: Optional[int] = None,
        json_mode: bool = False,
        stream: Optional[bool] = None,
        non_stream: Optional[bool] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        try:
            options = {
                "num_predict": max_tokens if max_tokens is not None else self.num_predict,
                "temperature": temperature if temperature is not None else self.temperature,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "stop": self.stop,
            }
            options = {key: value for key, value in options.items() if value is not None}

            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "options": options,
            }
            if non_stream is not None:
                payload["stream"] = not bool(non_stream)
            elif stream is not None:
                payload["stream"] = bool(stream)
            else:
                payload["stream"] = False
            if json_mode:
                payload["format"] = "json"
            elif "format" in payload:
                payload.pop("format", None)

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout or self.timeout,
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                duration = end_time - start_time

                if len(response_text) < 2:
                    logger.warning("[LLM] empty response payload")
                    return False, '', 'empty'

                logger.info(f"[LLM] success in {duration:.1f}s - {len(response_text)} chars")
                return True, response_text, None
            else:
                logger.error(f"[LLM] HTTP error: {response.status_code}")
                return False, "", "http_error"

        except requests.exceptions.Timeout:
            effective_timeout = timeout or self.timeout
            logger.error(f"[LLM] timeout after {effective_timeout}s")
            return False, "", "timeout"
        except Exception as e:
            logger.error(f"[LLM] exception: {str(e)}")
            return False, "", "exception"

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Robust JSON extraction from LLM responses."""
        if not text:
            return None

        try:
            cleaned = text.strip()
            cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S | re.I)
            cleaned = re.sub(r"```json\s*(.*?)\s*```", r"\1", cleaned, flags=re.S | re.I)
            cleaned = re.sub(r"```\s*(.*?)\s*```", r"\1", cleaned, flags=re.S | re.I)

            match = re.search(r"\{.*\}", cleaned, flags=re.S)
            if not match:
                logger.warning("[LLM] no JSON object found in response")
                return None

            json_str = match.group(0)
            return json.loads(json_str)

        except json.JSONDecodeError as exc:
            logger.error(f"[LLM] JSON parsing error: {exc}")
            return None
        except Exception as exc:
            logger.error(f"[LLM] unexpected error decoding JSON: {exc}")
            return None

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout: Optional[int] = None,
    ) -> str:
        success, response, err = self._call_llm(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if not success:
            raise RuntimeError(f"LLM completion failed: {err or 'unknown'}")
        return response

    def complete_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        response = self.complete(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        payload = self._extract_json(response)
        if payload is None:
            raise ValueError('LLM response did not contain JSON payload')
        return payload


    def generate_keywords(self, transcript: str, max_keywords: int = 15) -> Tuple[bool, List[str]]:
        """GÃƒÂ©nÃƒÂ©ration de mots-clÃƒÂ©s avec prompt minimaliste gÃƒÂ©nÃƒÂ©rique"""
        
        # Ã°Å¸Å½Â¯ PROMPT MINIMALISTE (votre approche parfaite)
        prompt = f"""Extract {max_keywords} to {max_keywords + 5} relevant single-word keywords from the transcript.
Do not invent unrelated terms.
Output JSON only: {{"keywords":["word1","word2", "..."]}}

Transcript: {transcript}

JSON:"""
        
        logger.info(f"Ã°Å¸Å½Â¯ GÃƒÂ©nÃƒÂ©ration mots-clÃƒÂ©s avec prompt minimaliste ({len(prompt)} caractÃƒÂ¨res)")

        success, response, _ = self._call_llm(prompt, json_mode=True, non_stream=True)
        if not success:
            return False, []
        
        # Extraction et validation
        json_data = self._extract_json(response)
        if not json_data:
            return False, []
        
        keywords = json_data.get("keywords", [])
        if not keywords or not isinstance(keywords, list):
            logger.warning("Ã¢Å¡Â Ã¯Â¸Â Aucun mot-clÃƒÂ© valide trouvÃƒÂ©")
            return False, []
        
        # Nettoyage et validation
        clean_keywords = []
        seen = set()
        generic = {"what","over","your","look","when","learn","about","thing","things","people","really","going","want","need","make","take","time","back","good","best","more","most","very","that","this","those","these"}
        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                clean_kw = kw.strip().lower()
                if len(clean_kw) <= 2:
                    continue
                if clean_kw in generic:
                    continue
                if not clean_kw.isalpha():
                    continue
                if clean_kw in seen:
                    continue
                seen.add(clean_kw)
                clean_keywords.append(clean_kw)
        
        # Prioritize specificity by length and uniqueness
        clean_keywords.sort(key=lambda k: (-len(k), k))
        
        logger.info(f"Ã¢Å“â€¦ {len(clean_keywords)} mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©rÃƒÂ©s avec succÃƒÂ¨s")
        return True, clean_keywords[:max_keywords]
    
    def generate_title_hashtags(self, transcript: str) -> Tuple[bool, Dict[str, Any]]:
        """GÃƒÂ©nÃƒÂ©ration titre + hashtags avec prompt minimaliste"""
        
        # Ã°Å¸Å½Â¯ PROMPT MINIMALISTE pour titre + hashtags
        prompt = f"""Generate a title and hashtags from this transcript.
Output JSON only: {{"title": "Title here", "hashtags": ["#tag1", "#tag2", "..."]}}

Transcript: {transcript}

JSON:"""
        
        logger.info(f"Ã°Å¸Å½Â¯ GÃƒÂ©nÃƒÂ©ration titre + hashtags avec prompt minimaliste ({len(prompt)} caractÃƒÂ¨res)")
        
        success, response, _ = self._call_llm(prompt, json_mode=True, non_stream=True)
        if not success:
            return False, {}
        
        # Extraction et validation
        json_data = self._extract_json(response)
        if not json_data:
            return False, {}
        
        title = json_data.get("title", "").strip()
        hashtags = json_data.get("hashtags", [])
        
        if not title:
            logger.warning("Ã¢Å¡Â Ã¯Â¸Â Aucun titre valide trouvÃƒÂ©")
            return False, {}
        
        # Nettoyage des hashtags
        clean_hashtags = []
        for tag in hashtags:
            if isinstance(tag, str) and tag.strip():
                clean_tag = tag.strip()
                if not clean_tag.startswith("#"):
                    clean_tag = f"#{clean_tag}"
                clean_hashtags.append(clean_tag)
        
        result = {
            "title": title,
            "hashtags": clean_hashtags
        }
        
        logger.info(f"Ã¢Å“â€¦ Titre et {len(clean_hashtags)} hashtags gÃƒÂ©nÃƒÂ©rÃƒÂ©s avec succÃƒÂ¨s")
        return True, result
    
    def generate_complete_metadata(self, transcript: str) -> Tuple[bool, Dict[str, Any]]:
        """GÃƒÂ©nÃƒÂ©ration complÃƒÂ¨te : titre, description, hashtags, mots-clÃƒÂ©s"""
        
        # Ã°Å¸Å½Â¯ PROMPT MINIMALISTE pour mÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes
        prompt = f"""Generate title, description, hashtags, and keywords from this transcript.
Output JSON only: {{"title": "Title", "description": "Description", "hashtags": ["#tag1"], "keywords": ["word1"]}}

Transcript: {transcript}

JSON:"""
        
        logger.info(f"Ã°Å¸Å½Â¯ GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes avec prompt minimaliste ({len(prompt)} caractÃƒÂ¨res)")
        
        success, response, _ = self._call_llm(prompt, json_mode=True, non_stream=True)
        if not success:
            return False, {}
        
        # Extraction et validation
        json_data = self._extract_json(response)
        if not json_data:
            return False, {}
        
        # Extraction des champs
        title = json_data.get("title", "").strip()
        description = json_data.get("description", "").strip()
        hashtags = json_data.get("hashtags", [])
        keywords = json_data.get("keywords", [])
        
        # Validation des champs obligatoires
        if not title:
            logger.warning("Ã¢Å¡Â Ã¯Â¸Â Aucun titre valide trouvÃƒÂ©")
            return False, {}
        
        # Nettoyage des hashtags
        clean_hashtags = []
        for tag in hashtags:
            if isinstance(tag, str) and tag.strip():
                clean_tag = tag.strip()
                if not clean_tag.startswith("#"):
                    clean_tag = f"#{clean_tag}"
                clean_hashtags.append(clean_tag)
        
        # Nettoyage des mots-clÃƒÂ©s
        clean_keywords = []
        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                clean_kw = kw.strip().lower()
                if len(clean_kw) > 2:
                    clean_keywords.append(clean_kw)
        
        result = {
            "title": title,
            "description": description,
            "hashtags": clean_hashtags,
            "keywords": clean_keywords
        }
        
        logger.info(f"Ã¢Å“â€¦ MÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes gÃƒÂ©nÃƒÂ©rÃƒÂ©es : titre, description, {len(clean_hashtags)} hashtags, {len(clean_keywords)} mots-clÃƒÂ©s")
        return True, result
    
    def generate_broll_keywords_and_queries(self, transcript: str, max_keywords: int = 15) -> Tuple[bool, Dict[str, Any]]:
        """
        Ã°Å¸Å½Â¯ NOUVEAU: GÃƒÂ©nÃƒÂ©ration spÃƒÂ©cialisÃƒÂ©e pour B-roll
        Produit explicitement broll_keywords + search_queries
        """
        
        # Ã°Å¸Å½Â¯ PROMPT OPTIMISÃƒâ€° pour B-roll hybride (actions + concepts)
        trimmed = transcript[:1500]
        prompt = f"""Tu es planificatrice B-roll pour un format vertical (TikTok/Shorts, 9:16). Ãƒâ‚¬ partir du transcript ci-dessous, produis des idÃƒÂ©es de vidÃƒÂ©os libres de droits.

Exigences :
- Analyse le thÃƒÂ¨me, lÃ¢â‚¬â„¢ÃƒÂ©motion et le rythme : pense en fenÃƒÂªtres de 3 ÃƒÂ  6 secondes.
- Garde uniquement des idÃƒÂ©es filmables (actions humaines prÃƒÂ©cises, dÃƒÂ©tails dÃ¢â‚¬â„¢objet, dÃƒÂ©cors identifiables).
- Ãƒâ€°vite les termes creux : people, thing, nice, background, start, generic.
- 60 %% dÃ¢â‚¬â„¢actions humaines (sujet_action_contexte avec underscores) / 40 %% de concepts visuels directs (ex. "brain_scan_monitor").
- Donne pour chaque idÃƒÂ©e une requÃƒÂªte courte (2 ÃƒÂ  4 mots) optimisÃƒÂ©e pour les APIs vidÃƒÂ©o.
- Produis aussi un mapping segmentaire facultatif pour faciliter la synchro.

RÃƒÂ©ponds uniquement en JSON :
{{
  "detected_domain": "...",
  "context": "rÃƒÂ©sumÃƒÂ© en 12 mots max",
  "broll_keywords": ["..."],
  "search_queries": ["..."],
  "segment_briefs": [
    {{"segment_index": 0, "suggested_window_s": 4, "keywords": ["action_prÃƒÂ©cise", "dÃƒÂ©tail_visuel"]}}
  ]
}}

Transcript (tronquÃƒÂ©) : {trimmed}
JSON:"""

        logger.info(f"Ã°Å¸Å½Â¯ GÃƒÂ©nÃƒÂ©ration B-roll avec prompt minimaliste ({len(prompt)} caractÃƒÂ¨res)")

        success, response, error_kind = self._call_llm(prompt, max_tokens=350, json_mode=True, non_stream=True)
        if not success and error_kind in {"timeout", "empty"}:
            if error_kind == "timeout":
                shorter = trimmed[:600]
                retry_prompt = prompt.replace(trimmed, shorter)
                logger.info("Ã¢ÂÂ±Ã¯Â¸Â Retentative LLM B-roll avec transcript raccourci")
                success, response, error_kind = self._call_llm(retry_prompt, max_tokens=200, timeout=40, json_mode=True, non_stream=True)
            else:
                logger.info("[LLM] Retentative B-roll aprÃƒÂ¨s rÃƒÂ©ponse vide")
                success, response, error_kind = self._call_llm(prompt, max_tokens=250, json_mode=True, non_stream=True)
        if not success:
            return False, {}
        
        # Extraction et validation
        json_data = self._extract_json(response)
        if not json_data:
            return False, {}
        
        # Extraction des champs enrichis
        domain = json_data.get("domain", "").strip()
        context = json_data.get("context", "").strip() 
        broll_keywords = json_data.get("broll_keywords", [])
        search_queries = json_data.get("search_queries", [])
        
        # Validation des champs
        if not broll_keywords or not search_queries:
            logger.warning("Ã¢Å¡Â Ã¯Â¸Â Champs B-roll manquants dans la rÃƒÂ©ponse")
            return False, {}
        
        # Nettoyage des mots-clÃƒÂ©s B-roll
        clean_broll_keywords = []
        for kw in broll_keywords:
            if isinstance(kw, str) and kw.strip():
                clean_kw = kw.strip().lower()
                if len(clean_kw) > 2:
                    clean_broll_keywords.append(clean_kw)
        
        # Nettoyage des requÃƒÂªtes de recherche
        clean_search_queries = []
        for query in search_queries:
            if isinstance(query, str) and query.strip():
                clean_query = query.strip()
                if len(clean_query) <= 30:  # AugmentÃƒÂ© pour phrases plus descriptives
                    clean_search_queries.append(clean_query)
        
        result = {
            "domain": domain,
            "context": context,
            "broll_keywords": clean_broll_keywords[:max_keywords],
            "search_queries": clean_search_queries[:max_keywords]
        }
        
        logger.info(f"Ã¢Å“â€¦ B-roll gÃƒÂ©nÃƒÂ©rÃƒÂ© : {len(clean_broll_keywords)} mots-clÃƒÂ©s, {len(clean_search_queries)} requÃƒÂªtes")
        return True, result
    
    def generate_metadata_with_broll(self, transcript: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Ã°Å¸Å½Â¯ NOUVEAU: GÃƒÂ©nÃƒÂ©ration complÃƒÂ¨te avec mÃƒÂ©tadonnÃƒÂ©es + B-roll
        Combine toutes les informations nÃƒÂ©cessaires
        """
        
        # Ã°Å¸Å½Â¯ PROMPT VIRAL pour mÃƒÂ©tadonnÃƒÂ©es + B-roll
        prompt = f"""Tu es copywriter growth pour vidÃƒÂ©os verticales (TikTok/Shorts).

Objectif : gÃƒÂ©nÃƒÂ©rer un TITRE + DESCRIPTION qui stoppent le scroll et maximisent la rÃƒÂ©tention.

Contraintes :
- Titre : 60 ÃƒÂ  70 caractÃƒÂ¨res, commence par un hook (verbe dÃ¢â‚¬â„¢action, question ou chiffre) et annonce le bÃƒÂ©nÃƒÂ©fice principal.
- Description : 3 phrases max. Phrase 1 = bÃƒÂ©nÃƒÂ©fice concret; Phrase 2 = preuve/tip actionnable; Phrase 3 = CTA soft (ex. "Sauvegarde ce clip"). Total Ã¢â€°Â¤ 220 caractÃƒÂ¨res.
- Ajoute 4 ÃƒÂ  6 hashtags pertinents (mix niche + large, sans doublon).
- Fournis 6 mots-clÃƒÂ©s SEO en snake_case et 3 requÃƒÂªtes B-roll optimisÃƒÂ©es pour des banques vidÃƒÂ©o.
- Ton positif, pas de clickbait vide, pas de MAJUSCULES abusives.

RÃƒÂ©ponds uniquement en JSON :
{{
    "title": "...",
    "description": "...",
    "hashtags": ["#..."],
    "keywords": ["mot_clef"],
    "broll_keywords": ["visual_word"],
    "search_queries": ["requÃƒÂªte vidÃƒÂ©o"]
}}

Transcript : {transcript}
JSON:"""
        
        logger.info(f"Ã°Å¸Å½Â¯ GÃƒÂ©nÃƒÂ©ration complÃƒÂ¨te avec B-roll ({len(prompt)} caractÃƒÂ¨res)")
        
        success, response, _ = self._call_llm(prompt, json_mode=True, non_stream=True)
        if not success:
            return False, {}
        
        # Extraction et validation
        json_data = self._extract_json(response)
        if not json_data:
            return False, {}
        
        # Extraction de tous les champs
        title = json_data.get("title", "").strip()
        description = json_data.get("description", "").strip()
        hashtags = json_data.get("hashtags", [])
        keywords = json_data.get("keywords", [])
        broll_keywords = json_data.get("broll_keywords", [])
        search_queries = json_data.get("search_queries", [])
        
        # Validation des champs obligatoires
        if not title:
            logger.warning("Ã¢Å¡Â Ã¯Â¸Â Aucun titre valide trouvÃƒÂ©")
            return False, {}
        
        # Nettoyage des hashtags
        clean_hashtags = []
        for tag in hashtags:
            if isinstance(tag, str) and tag.strip():
                clean_tag = tag.strip()
                if not clean_tag.startswith("#"):
                    clean_tag = f"#{clean_tag}"
                clean_hashtags.append(clean_tag)
        
        # Nettoyage des mots-clÃƒÂ©s
        clean_keywords = []
        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                clean_kw = kw.strip().lower()
                if len(clean_kw) > 2:
                    clean_keywords.append(clean_kw)
        
        # Nettoyage des mots-clÃƒÂ©s B-roll
        clean_broll_keywords = []
        for kw in broll_keywords:
            if isinstance(kw, str) and kw.strip():
                clean_kw = kw.strip().lower()
                if len(clean_kw) > 2:
                    clean_broll_keywords.append(clean_kw)
        
        # Nettoyage des requÃƒÂªtes de recherche
        clean_search_queries = []
        for query in search_queries:
            if isinstance(query, str) and query.strip():
                clean_query = query.strip()
                if len(clean_query) <= 25:
                    clean_search_queries.append(clean_query)
        
        result = {
            "title": title,
            "description": description,
            "hashtags": clean_hashtags,
            "keywords": clean_keywords,
            "broll_keywords": clean_broll_keywords,
            "search_queries": clean_search_queries
        }
        
        logger.info(f"Ã¢Å“â€¦ MÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes avec B-roll : titre, description, {len(clean_hashtags)} hashtags, {len(clean_keywords)} mots-clÃƒÂ©s, {len(clean_broll_keywords)} B-roll, {len(clean_search_queries)} requÃƒÂªtes")
        return True, result

# === FONCTIONS UTILITAIRES POUR L'INTÃƒâ€°GRATION ===

def create_optimized_llm(base_url: str = None, model: str = None) -> OptimizedLLM:
    """Factory pour crÃƒÂ©er une instance LLM optimisÃƒÂ©e"""
    
    # DÃƒÂ©tection automatique de l'URL et du modÃƒÂ¨le
    if not base_url:
        # Essayer Ollama en premier
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                base_url = "http://localhost:11434"
                logger.info("Ã¢Å“â€¦ Ollama dÃƒÂ©tectÃƒÂ© sur localhost:11434")
            else:
                base_url = "http://localhost:1234"  # LM Studio par dÃƒÂ©faut
                logger.info("Ã¢Å¡Â Ã¯Â¸Â Ollama non disponible, utilisation LM Studio par dÃƒÂ©faut")
        except:
            base_url = "http://localhost:1234"
            logger.info("Ã¢Å¡Â Ã¯Â¸Â Aucun LLM local dÃƒÂ©tectÃƒÂ©, utilisation LM Studio par dÃƒÂ©faut")
    
    if not model:
        # ModÃƒÂ¨le par dÃƒÂ©faut selon la disponibilitÃƒÂ©
        if "11434" in base_url:  # Ollama
            model = "gemma3:4b"  # ModÃƒÂ¨le recommandÃƒÂ©
        else:  # LM Studio
            model = "default"
    
    return OptimizedLLM(base_url, model)

def generate_keywords_for_pipeline(transcript: str, max_keywords: int = 15) -> Tuple[bool, List[str]]:
    """Fonction utilitaire pour intÃƒÂ©gration directe dans le pipeline"""
    llm = create_optimized_llm()
    return llm.generate_keywords(transcript, max_keywords)

def generate_metadata_for_pipeline(transcript: str) -> Tuple[bool, Dict[str, Any]]:
    """Fonction utilitaire pour intÃƒÂ©gration directe dans le pipeline"""
    llm = create_optimized_llm()
    return llm.generate_complete_metadata(transcript)

def generate_broll_for_pipeline(transcript: str, max_keywords: int = 15) -> Tuple[bool, Dict[str, Any]]:
    """Ã°Å¸Å½Â¯ NOUVEAU: Fonction utilitaire pour B-roll"""
    llm = create_optimized_llm()
    return llm.generate_broll_keywords_and_queries(transcript, max_keywords)

def generate_complete_with_broll(transcript: str) -> Tuple[bool, Dict[str, Any]]:
    """Ã°Å¸Å½Â¯ NOUVEAU: Fonction utilitaire pour mÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes avec B-roll"""
    llm = create_optimized_llm()
    return llm.generate_metadata_with_broll(transcript)

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("Ã°Å¸Â§Â  Test du systÃƒÂ¨me LLM optimisÃƒÂ©...")
    
    # Test avec un transcript simple
    test_transcript = "EMDR therapy utilizes bilateral stimulation to process traumatic memories. The therapist guides the patient through eye movements while recalling distressing events."
    
    llm = create_optimized_llm()
    
    # Test mots-clÃƒÂ©s
    print("\nÃ°Å¸Å½Â¯ Test gÃƒÂ©nÃƒÂ©ration mots-clÃƒÂ©s...")
    success, keywords = llm.generate_keywords(test_transcript, 10)
    if success:
        print(f"Ã¢Å“â€¦ Mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©rÃƒÂ©s: {keywords}")
    else:
        print("Ã¢ÂÅ’ Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration mots-clÃƒÂ©s")
    
    # Test mÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes
    print("\nÃ°Å¸Å½Â¯ Test gÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes...")
    success, metadata = llm.generate_complete_metadata(test_transcript)
    if success:
        print(f"Ã¢Å“â€¦ MÃƒÂ©tadonnÃƒÂ©es gÃƒÂ©nÃƒÂ©rÃƒÂ©es:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    else:
        print("Ã¢ÂÅ’ Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es")
    
    # Ã°Å¸Å½Â¯ NOUVEAU: Test B-roll
    print("\nÃ°Å¸Å½Â¯ Test gÃƒÂ©nÃƒÂ©ration B-roll...")
    success, broll_data = llm.generate_broll_keywords_and_queries(test_transcript, 8)
    if success:
        print(f"Ã¢Å“â€¦ B-roll gÃƒÂ©nÃƒÂ©rÃƒÂ©:")
        print(f"   Mots-clÃƒÂ©s: {broll_data['broll_keywords']}")
        print(f"   RequÃƒÂªtes: {broll_data['search_queries']}")
    else:
        print("Ã¢ÂÅ’ Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration B-roll")
    
    # Ã°Å¸Å½Â¯ NOUVEAU: Test complet avec B-roll
    print("\nÃ°Å¸Å½Â¯ Test gÃƒÂ©nÃƒÂ©ration complÃƒÂ¨te avec B-roll...")
    success, complete_data = llm.generate_metadata_with_broll(test_transcript)
    if success:
        print(f"Ã¢Å“â€¦ DonnÃƒÂ©es complÃƒÂ¨tes gÃƒÂ©nÃƒÂ©rÃƒÂ©es:")
        for key, value in complete_data.items():
            print(f"   {key}: {value}")
    else:
        print("Ã¢ÂÅ’ Ãƒâ€°chec gÃƒÂ©nÃƒÂ©ration complÃƒÂ¨te") 


