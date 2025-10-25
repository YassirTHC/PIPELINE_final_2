ï»¿#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ã°Å¸Å½Â¯ PROMPTS OPTIMISÃƒâ€°S POUR GEMMA3:4B
Prompts concis et directifs adaptÃƒÂ©s au modÃƒÂ¨le lÃƒÂ©ger
"""

class OptimizedPrompts:
    """Prompts optimisÃƒÂ©s pour gemma3:4b"""
    
    @staticmethod
    def generate_keywords_prompt(transcript: str, max_keywords: int = 15) -> str:
        """
        Prompt optimisÃƒÂ© pour la gÃƒÂ©nÃƒÂ©ration de mots-clÃƒÂ©s
        
        Args:
            transcript: Transcription du texte
            max_keywords: Nombre maximum de mots-clÃƒÂ©s
            
        Returns:
            Prompt optimisÃƒÂ©
        """
        
        # Prompt ultra-concis et directif
        prompt = f"""Generate {max_keywords} filmable keywords from this transcript.
Output ONLY valid JSON: {{"keywords":["k1","k2","k3"]}}

Transcript: {transcript[:500]}  # LimitÃƒÂ© ÃƒÂ  500 caractÃƒÂ¨res

JSON:"""
        
        return prompt
    
    @staticmethod
    def generate_title_hashtags_prompt(transcript: str) -> str:
        """
        Prompt optimisÃƒÂ© pour titre + hashtags
        
        Args:
            transcript: Transcription du texte
            
        Returns:
            Prompt optimisÃƒÂ©
        """
        
        prompt = f"""Generate title and hashtags from transcript.
Output ONLY valid JSON: {{"title":"Title","hashtags":["#tag1","#tag2"]}}

Transcript: {transcript[:300]}  # LimitÃƒÂ© ÃƒÂ  300 caractÃƒÂ¨res

JSON:"""
        
        return prompt
    
    @staticmethod
    def generate_content_summary_prompt(transcript: str) -> str:
        """
        Prompt optimisÃƒÂ© pour rÃƒÂ©sumÃƒÂ© de contenu
        
        Args:
            transcript: Transcription du texte
            
        Returns:
            Prompt optimisÃƒÂ©
        """
        
        prompt = f"""Summarize this content in 2-3 sentences.
Output ONLY valid JSON: {{"summary":"text"}}

Content: {transcript[:400]}  # LimitÃƒÂ© ÃƒÂ  400 caractÃƒÂ¨res

JSON:"""
        
        return prompt
    
    @staticmethod
    def get_model_parameters() -> dict:
        """
        ParamÃƒÂ¨tres optimisÃƒÂ©s pour gemma3:4b
        
        Returns:
            Dict des paramÃƒÂ¨tres
        """
        
        return {
            "temperature": 0.3,        # ModÃƒÂ©rÃƒÂ© pour la cohÃƒÂ©rence
            "max_tokens": 2000,        # LimitÃƒÂ© pour ÃƒÂ©viter le markdown
            "top_p": 0.9,             # DiversitÃƒÂ© contrÃƒÂ´lÃƒÂ©e
            "top_k": 40,              # Limite les choix
            "repeat_penalty": 1.1,    # Ãƒâ€°vite la rÃƒÂ©pÃƒÂ©tition
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

