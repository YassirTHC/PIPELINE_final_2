ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Script de nettoyage des fonctions d'animations
Supprime toutes les fonctions d'animations problÃƒÂ©matiques
"""

import re
from pathlib import Path

def clean_animations():
    """Nettoie le fichier video_processor.py des fonctions d'animations"""
    
    file_path = Path('video_processor.py')
    
    print("Ã°Å¸Â§Â¹ NETTOYAGE DES FONCTIONS D'ANIMATIONS")
    print("=" * 50)
    
    try:
        # Lire le fichier
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_lines = len(content.split('\n'))
        print(f"Ã°Å¸â€œâ€ž Fichier original: {original_lines} lignes")
        
        # Supprimer les fonctions d'animations en utilisant des patterns plus simples
        functions_to_remove = [
            'def add_contextual_animations_and_emojis(',
            'def analyze_content_for_animations(',
            'def create_contextual_animations(',
            'def create_contextual_emojis(',
            'def load_emoji_png(',
            'def create_animation_timing(',
            'def create_emoji_timing('
        ]
        
        lines = content.split('\n')
        cleaned_lines = []
        skip_function = False
        current_function = None
        
        for line in lines:
            # VÃƒÂ©rifier si on commence une fonction ÃƒÂ  supprimer
            should_skip = False
            for func_start in functions_to_remove:
                if line.strip().startswith(func_start):
                    skip_function = True
                    current_function = func_start
                    print(f"Ã°Å¸â€”â€˜Ã¯Â¸Â Suppression de la fonction: {func_start}")
                    break
            
            # Si on est dans une fonction ÃƒÂ  supprimer, continuer ÃƒÂ  sauter
            if skip_function:
                # VÃƒÂ©rifier si on a atteint la fin de la fonction (ligne vide ou nouvelle fonction)
                if (line.strip() == '' or 
                    (line.strip().startswith('def ') and not line.strip().startswith(current_function))):
                    skip_function = False
                    current_function = None
                continue
            
            # Garder la ligne si elle n'est pas dans une fonction ÃƒÂ  supprimer
            cleaned_lines.append(line)
        
        # Reconstituer le contenu
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Nettoyer les lignes vides multiples
        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
        
        # Ãƒâ€°crire le fichier nettoyÃƒÂ©
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        new_lines = len(cleaned_content.split('\n'))
        print(f"Ã°Å¸â€œâ€ž Fichier nettoyÃƒÂ©: {new_lines} lignes")
        print(f"Ã°Å¸â€”â€˜Ã¯Â¸Â Lignes supprimÃƒÂ©es: {original_lines - new_lines}")
        
        print("Ã¢Å“â€¦ Nettoyage terminÃƒÂ© avec succÃƒÂ¨s!")
        return True
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur lors du nettoyage: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    clean_animations() 

