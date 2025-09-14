#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 PROMPTS OPTIMISÉS POUR GEMMA3:4B
Prompts concis et directifs adaptés au modèle léger
"""

class OptimizedPrompts:
    """Prompts optimisés pour gemma3:4b"""
    
    @staticmethod
    def generate_keywords_prompt(transcript: str, max_keywords: int = 15) -> str:
        """
        Prompt optimisé pour la génération de mots-clés
        
        Args:
            transcript: Transcription du texte
            max_keywords: Nombre maximum de mots-clés
            
        Returns:
            Prompt optimisé
        """
        
        # Prompt ultra-concis et directif
        prompt = f"""Generate {max_keywords} filmable keywords from this transcript.
Output ONLY valid JSON: {{"keywords":["k1","k2","k3"]}}

Transcript: {transcript[:500]}  # Limité à 500 caractères

JSON:"""
        
        return prompt
    
    @staticmethod
    def generate_title_hashtags_prompt(transcript: str) -> str:
        """
        Prompt optimisé pour titre + hashtags
        
        Args:
            transcript: Transcription du texte
            
        Returns:
            Prompt optimisé
        """
        
        prompt = f"""Generate title and hashtags from transcript.
Output ONLY valid JSON: {{"title":"Title","hashtags":["#tag1","#tag2"]}}

Transcript: {transcript[:300]}  # Limité à 300 caractères

JSON:"""
        
        return prompt
    
    @staticmethod
    def generate_content_summary_prompt(transcript: str) -> str:
        """
        Prompt optimisé pour résumé de contenu
        
        Args:
            transcript: Transcription du texte
            
        Returns:
            Prompt optimisé
        """
        
        prompt = f"""Summarize this content in 2-3 sentences.
Output ONLY valid JSON: {{"summary":"text"}}

Content: {transcript[:400]}  # Limité à 400 caractères

JSON:"""
        
        return prompt
    
    @staticmethod
    def get_model_parameters() -> dict:
        """
        Paramètres optimisés pour gemma3:4b
        
        Returns:
            Dict des paramètres
        """
        
        return {
            "temperature": 0.3,        # Modéré pour la cohérence
            "max_tokens": 2000,        # Limité pour éviter le markdown
            "top_p": 0.9,             # Diversité contrôlée
            "top_k": 40,              # Limite les choix
            "repeat_penalty": 1.1,    # Évite la répétition
            "stream": False            # Pas de streaming
        }
    
    @staticmethod
    def get_fallback_prompt() -> str:
        """
        Prompt de fallback ultra-simple
        
        Returns:
            Prompt de fallback
        """
        
        return """Generate 5 simple keywords.
JSON: {"keywords":["k1","k2","k3","k4","k5"]}"""

# Instance globale
optimized_prompts = OptimizedPrompts() 