ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Script de nettoyage COMPLET des fonctions d'animations
Supprime TOUTES les fonctions d'animations problÃƒÂ©matiques
"""

from pathlib import Path

def clean_animations_complet():
    """Nettoie COMPLÃƒË†TEMENT le fichier video_processor.py des fonctions d'animations"""
    
    file_path = Path('video_processor.py')
    
    print("Ã°Å¸Â§Â¹ NETTOYAGE COMPLET DES FONCTIONS D'ANIMATIONS")
    print("=" * 60)
    
    try:
        # Lire le fichier
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        original_lines = len(lines)
        print(f"Ã°Å¸â€œâ€ž Fichier original: {original_lines} lignes")
        
        # Identifier les lignes ÃƒÂ  supprimer
        lines_to_keep = []
        in_function_to_remove = False
        current_function = None
        
        for i, line in enumerate(lines):
            # VÃƒÂ©rifier si on commence une fonction ÃƒÂ  supprimer
            if any(line.strip().startswith(start) for start in [
                'def add_contextual_animations_and_emojis(',
                'def analyze_content_for_animations(',
                'def create_contextual_animations(',
                'def create_contextual_emojis(',
                'def load_emoji_png(',
                'def create_animation_timing(',
                'def create_emoji_timing('
            ]):
                in_function_to_remove = True
                current_function = line.strip()
                print(f"Ã°Å¸â€”â€˜Ã¯Â¸Â Suppression de: {current_function}")
                continue
            
            # Si on est dans une fonction ÃƒÂ  supprimer
            if in_function_to_remove:
                # VÃƒÂ©rifier si on a atteint la fin de la fonction
                if (line.strip() == '' or 
                    (line.strip().startswith('def ') and not line.strip().startswith(current_function)) or
                    (line.strip().startswith('class ') and not line.strip().startswith(current_function))):
                    in_function_to_remove = False
                    current_function = None
                continue
            
            # Garder la ligne si elle n'est pas dans une fonction ÃƒÂ  supprimer
            lines_to_keep.append(line)
        
        # Ãƒâ€°crire le fichier nettoyÃƒÂ©
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines_to_keep)
        
        new_lines = len(lines_to_keep)
        print(f"Ã°Å¸â€œâ€ž Fichier nettoyÃƒÂ©: {new_lines} lignes")
        print(f"Ã°Å¸â€”â€˜Ã¯Â¸Â Lignes supprimÃƒÂ©es: {original_lines - new_lines}")
        
        print("Ã¢Å“â€¦ Nettoyage COMPLET terminÃƒÂ© avec succÃƒÂ¨s!")
        return True
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur lors du nettoyage: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    clean_animations_complet() 

