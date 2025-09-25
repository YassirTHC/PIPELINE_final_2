#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SYSTÈME LLM MINIMALISTE - PROMPTS GÉNÉRIQUES + SPÉCIALISATION PIPELINE
Basé sur l'analyse brillante de l'utilisateur : prompts simples + spécialisation intelligente
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedLLM:
    """Système LLM avec prompts minimalistes et spécialisation via pipeline"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:4b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = 60  # Timeout plus court pour détecter rapidement les blocages

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 100,
        *,
        timeout: Optional[int] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Appel LLM simple avec gestion d'erreur"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
                "max_tokens": max_tokens,
            }

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

                logger.info(f"✅ LLM réussi en {duration:.1f}s - {len(response_text)} caractères")
                return True, response_text, None
            else:
                logger.error(f"❌ Erreur HTTP: {response.status_code}")
                return False, "", "http_error"

        except requests.exceptions.Timeout:
            effective_timeout = timeout or self.timeout
            logger.error(f"⏱️ Timeout après {effective_timeout}s")
            return False, "", "timeout"
        except Exception as e:
            logger.error(f"❌ Erreur LLM: {str(e)}")
            return False, "", "exception"
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extraction robuste du JSON depuis la réponse LLM"""
        try:
            # Nettoyer et extraire le JSON
            text = text.strip()
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx + 1]
                return json.loads(json_str)
            else:
                logger.warning("⚠️ Aucun JSON trouvé dans la réponse")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erreur parsing JSON: {e}")
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
        """Génération de mots-clés avec prompt minimaliste générique"""
        
        # 🎯 PROMPT MINIMALISTE (votre approche parfaite)
        prompt = f"""Extract {max_keywords} to {max_keywords + 5} relevant single-word keywords from the transcript.
Do not invent unrelated terms.
Output JSON only: {{"keywords":["word1","word2", "..."]}}

Transcript: {transcript}

JSON:"""
        
        logger.info(f"🎯 Génération mots-clés avec prompt minimaliste ({len(prompt)} caractères)")

        success, response, _ = self._call_llm(prompt)
        if not success:
            return False, []
        
        # Extraction et validation
        json_data = self._extract_json(response)
        if not json_data:
            return False, []
        
        keywords = json_data.get("keywords", [])
        if not keywords or not isinstance(keywords, list):
            logger.warning("⚠️ Aucun mot-clé valide trouvé")
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
        
        logger.info(f"✅ {len(clean_keywords)} mots-clés générés avec succès")
        return True, clean_keywords[:max_keywords]
    
    def generate_title_hashtags(self, transcript: str) -> Tuple[bool, Dict[str, Any]]:
        """Génération titre + hashtags avec prompt minimaliste"""
        
        # 🎯 PROMPT MINIMALISTE pour titre + hashtags
        prompt = f"""Generate a title and hashtags from this transcript.
Output JSON only: {{"title": "Title here", "hashtags": ["#tag1", "#tag2", "..."]}}

Transcript: {transcript}

JSON:"""
        
        logger.info(f"🎯 Génération titre + hashtags avec prompt minimaliste ({len(prompt)} caractères)")
        
        success, response, _ = self._call_llm(prompt)
        if not success:
            return False, {}
        
        # Extraction et validation
        json_data = self._extract_json(response)
        if not json_data:
            return False, {}
        
        title = json_data.get("title", "").strip()
        hashtags = json_data.get("hashtags", [])
        
        if not title:
            logger.warning("⚠️ Aucun titre valide trouvé")
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
        
        logger.info(f"✅ Titre et {len(clean_hashtags)} hashtags générés avec succès")
        return True, result
    
    def generate_complete_metadata(self, transcript: str) -> Tuple[bool, Dict[str, Any]]:
        """Génération complète : titre, description, hashtags, mots-clés"""
        
        # 🎯 PROMPT MINIMALISTE pour métadonnées complètes
        prompt = f"""Generate title, description, hashtags, and keywords from this transcript.
Output JSON only: {{"title": "Title", "description": "Description", "hashtags": ["#tag1"], "keywords": ["word1"]}}

Transcript: {transcript}

JSON:"""
        
        logger.info(f"🎯 Génération métadonnées complètes avec prompt minimaliste ({len(prompt)} caractères)")
        
        success, response, _ = self._call_llm(prompt)
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
            logger.warning("⚠️ Aucun titre valide trouvé")
            return False, {}
        
        # Nettoyage des hashtags
        clean_hashtags = []
        for tag in hashtags:
            if isinstance(tag, str) and tag.strip():
                clean_tag = tag.strip()
                if not clean_tag.startswith("#"):
                    clean_tag = f"#{clean_tag}"
                clean_hashtags.append(clean_tag)
        
        # Nettoyage des mots-clés
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
        
        logger.info(f"✅ Métadonnées complètes générées : titre, description, {len(clean_hashtags)} hashtags, {len(clean_keywords)} mots-clés")
        return True, result
    
    def generate_broll_keywords_and_queries(self, transcript: str, max_keywords: int = 15) -> Tuple[bool, Dict[str, Any]]:
        """
        🎯 NOUVEAU: Génération spécialisée pour B-roll
        Produit explicitement broll_keywords + search_queries
        """
        
        # 🎯 PROMPT OPTIMISÉ pour B-roll hybride (actions + concepts)
        trimmed = transcript[:1500]
        prompt = f"""Tu es planificatrice B-roll pour un format vertical (TikTok/Shorts, 9:16). À partir du transcript ci-dessous, produis des idées de vidéos libres de droits.

Exigences :
- Analyse le thème, l’émotion et le rythme : pense en fenêtres de 3 à 6 secondes.
- Garde uniquement des idées filmables (actions humaines précises, détails d’objet, décors identifiables).
- Évite les termes creux : people, thing, nice, background, start, generic.
- 60 %% d’actions humaines (sujet_action_contexte avec underscores) / 40 %% de concepts visuels directs (ex. "brain_scan_monitor").
- Donne pour chaque idée une requête courte (2 à 4 mots) optimisée pour les APIs vidéo.
- Produis aussi un mapping segmentaire facultatif pour faciliter la synchro.

Réponds uniquement en JSON :
{{
  "detected_domain": "...",
  "context": "résumé en 12 mots max",
  "broll_keywords": ["..."],
  "search_queries": ["..."],
  "segment_briefs": [
    {{"segment_index": 0, "suggested_window_s": 4, "keywords": ["action_précise", "détail_visuel"]}}
  ]
}}

Transcript (tronqué) : {trimmed}
JSON:"""

        logger.info(f"🎯 Génération B-roll avec prompt minimaliste ({len(prompt)} caractères)")

        success, response, error_kind = self._call_llm(prompt, max_tokens=350)
        if not success and error_kind == "timeout":
            shorter = trimmed[:600]
            retry_prompt = prompt.replace(trimmed, shorter)
            logger.info("⏱️ Retentative LLM B-roll avec transcript raccourci")
            success, response, error_kind = self._call_llm(retry_prompt, max_tokens=200, timeout=40)
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
            logger.warning("⚠️ Champs B-roll manquants dans la réponse")
            return False, {}
        
        # Nettoyage des mots-clés B-roll
        clean_broll_keywords = []
        for kw in broll_keywords:
            if isinstance(kw, str) and kw.strip():
                clean_kw = kw.strip().lower()
                if len(clean_kw) > 2:
                    clean_broll_keywords.append(clean_kw)
        
        # Nettoyage des requêtes de recherche
        clean_search_queries = []
        for query in search_queries:
            if isinstance(query, str) and query.strip():
                clean_query = query.strip()
                if len(clean_query) <= 30:  # Augmenté pour phrases plus descriptives
                    clean_search_queries.append(clean_query)
        
        result = {
            "domain": domain,
            "context": context,
            "broll_keywords": clean_broll_keywords[:max_keywords],
            "search_queries": clean_search_queries[:max_keywords]
        }
        
        logger.info(f"✅ B-roll généré : {len(clean_broll_keywords)} mots-clés, {len(clean_search_queries)} requêtes")
        return True, result
    
    def generate_metadata_with_broll(self, transcript: str) -> Tuple[bool, Dict[str, Any]]:
        """
        🎯 NOUVEAU: Génération complète avec métadonnées + B-roll
        Combine toutes les informations nécessaires
        """
        
        # 🎯 PROMPT VIRAL pour métadonnées + B-roll
        prompt = f"""Tu es copywriter growth pour vidéos verticales (TikTok/Shorts).

Objectif : générer un TITRE + DESCRIPTION qui stoppent le scroll et maximisent la rétention.

Contraintes :
- Titre : 60 à 70 caractères, commence par un hook (verbe d’action, question ou chiffre) et annonce le bénéfice principal.
- Description : 3 phrases max. Phrase 1 = bénéfice concret; Phrase 2 = preuve/tip actionnable; Phrase 3 = CTA soft (ex. "Sauvegarde ce clip"). Total ≤ 220 caractères.
- Ajoute 4 à 6 hashtags pertinents (mix niche + large, sans doublon).
- Fournis 6 mots-clés SEO en snake_case et 3 requêtes B-roll optimisées pour des banques vidéo.
- Ton positif, pas de clickbait vide, pas de MAJUSCULES abusives.

Réponds uniquement en JSON :
{{
    "title": "...",
    "description": "...",
    "hashtags": ["#..."],
    "keywords": ["mot_clef"],
    "broll_keywords": ["visual_word"],
    "search_queries": ["requête vidéo"]
}}

Transcript : {transcript}
JSON:"""
        
        logger.info(f"🎯 Génération complète avec B-roll ({len(prompt)} caractères)")
        
        success, response, _ = self._call_llm(prompt)
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
            logger.warning("⚠️ Aucun titre valide trouvé")
            return False, {}
        
        # Nettoyage des hashtags
        clean_hashtags = []
        for tag in hashtags:
            if isinstance(tag, str) and tag.strip():
                clean_tag = tag.strip()
                if not clean_tag.startswith("#"):
                    clean_tag = f"#{clean_tag}"
                clean_hashtags.append(clean_tag)
        
        # Nettoyage des mots-clés
        clean_keywords = []
        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                clean_kw = kw.strip().lower()
                if len(clean_kw) > 2:
                    clean_keywords.append(clean_kw)
        
        # Nettoyage des mots-clés B-roll
        clean_broll_keywords = []
        for kw in broll_keywords:
            if isinstance(kw, str) and kw.strip():
                clean_kw = kw.strip().lower()
                if len(clean_kw) > 2:
                    clean_broll_keywords.append(clean_kw)
        
        # Nettoyage des requêtes de recherche
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
        
        logger.info(f"✅ Métadonnées complètes avec B-roll : titre, description, {len(clean_hashtags)} hashtags, {len(clean_keywords)} mots-clés, {len(clean_broll_keywords)} B-roll, {len(clean_search_queries)} requêtes")
        return True, result

# === FONCTIONS UTILITAIRES POUR L'INTÉGRATION ===

def create_optimized_llm(base_url: str = None, model: str = None) -> OptimizedLLM:
    """Factory pour créer une instance LLM optimisée"""
    
    # Détection automatique de l'URL et du modèle
    if not base_url:
        # Essayer Ollama en premier
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                base_url = "http://localhost:11434"
                logger.info("✅ Ollama détecté sur localhost:11434")
            else:
                base_url = "http://localhost:1234"  # LM Studio par défaut
                logger.info("⚠️ Ollama non disponible, utilisation LM Studio par défaut")
        except:
            base_url = "http://localhost:1234"
            logger.info("⚠️ Aucun LLM local détecté, utilisation LM Studio par défaut")
    
    if not model:
        # Modèle par défaut selon la disponibilité
        if "11434" in base_url:  # Ollama
            model = "gemma3:4b"  # Modèle recommandé
        else:  # LM Studio
            model = "default"
    
    return OptimizedLLM(base_url, model)

def generate_keywords_for_pipeline(transcript: str, max_keywords: int = 15) -> Tuple[bool, List[str]]:
    """Fonction utilitaire pour intégration directe dans le pipeline"""
    llm = create_optimized_llm()
    return llm.generate_keywords(transcript, max_keywords)

def generate_metadata_for_pipeline(transcript: str) -> Tuple[bool, Dict[str, Any]]:
    """Fonction utilitaire pour intégration directe dans le pipeline"""
    llm = create_optimized_llm()
    return llm.generate_complete_metadata(transcript)

def generate_broll_for_pipeline(transcript: str, max_keywords: int = 15) -> Tuple[bool, Dict[str, Any]]:
    """🎯 NOUVEAU: Fonction utilitaire pour B-roll"""
    llm = create_optimized_llm()
    return llm.generate_broll_keywords_and_queries(transcript, max_keywords)

def generate_complete_with_broll(transcript: str) -> Tuple[bool, Dict[str, Any]]:
    """🎯 NOUVEAU: Fonction utilitaire pour métadonnées complètes avec B-roll"""
    llm = create_optimized_llm()
    return llm.generate_metadata_with_broll(transcript)

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("🧠 Test du système LLM optimisé...")
    
    # Test avec un transcript simple
    test_transcript = "EMDR therapy utilizes bilateral stimulation to process traumatic memories. The therapist guides the patient through eye movements while recalling distressing events."
    
    llm = create_optimized_llm()
    
    # Test mots-clés
    print("\n🎯 Test génération mots-clés...")
    success, keywords = llm.generate_keywords(test_transcript, 10)
    if success:
        print(f"✅ Mots-clés générés: {keywords}")
    else:
        print("❌ Échec génération mots-clés")
    
    # Test métadonnées complètes
    print("\n🎯 Test génération métadonnées complètes...")
    success, metadata = llm.generate_complete_metadata(test_transcript)
    if success:
        print(f"✅ Métadonnées générées:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    else:
        print("❌ Échec génération métadonnées")
    
    # 🎯 NOUVEAU: Test B-roll
    print("\n🎯 Test génération B-roll...")
    success, broll_data = llm.generate_broll_keywords_and_queries(test_transcript, 8)
    if success:
        print(f"✅ B-roll généré:")
        print(f"   Mots-clés: {broll_data['broll_keywords']}")
        print(f"   Requêtes: {broll_data['search_queries']}")
    else:
        print("❌ Échec génération B-roll")
    
    # 🎯 NOUVEAU: Test complet avec B-roll
    print("\n🎯 Test génération complète avec B-roll...")
    success, complete_data = llm.generate_metadata_with_broll(test_transcript)
    if success:
        print(f"✅ Données complètes générées:")
        for key, value in complete_data.items():
            print(f"   {key}: {value}")
    else:
        print("❌ Échec génération complète") 
