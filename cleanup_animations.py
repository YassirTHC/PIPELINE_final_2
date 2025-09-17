#!/usr/bin/env python3
"""
Script de nettoyage des fonctions d'animations
Supprime toutes les fonctions d'animations problématiques
"""

import re
from pathlib import Path

def clean_animations():
    """Nettoie le fichier video_processor.py des fonctions d'animations"""
    
    file_path = Path('video_processor.py')
    
    print("🧹 NETTOYAGE DES FONCTIONS D'ANIMATIONS")
    print("=" * 50)
    
    try:
        # Lire le fichier
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_lines = len(content.split('\n'))
        print(f"📄 Fichier original: {original_lines} lignes")
        
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
            # Vérifier si on commence une fonction à supprimer
            should_skip = False
            for func_start in functions_to_remove:
                if line.strip().startswith(func_start):
                    skip_function = True
                    current_function = func_start
                    print(f"🗑️ Suppression de la fonction: {func_start}")
                    break
            
            # Si on est dans une fonction à supprimer, continuer à sauter
            if skip_function:
                # Vérifier si on a atteint la fin de la fonction (ligne vide ou nouvelle fonction)
                if (line.strip() == '' or 
                    (line.strip().startswith('def ') and not line.strip().startswith(current_function))):
                    skip_function = False
                    current_function = None
                continue
            
            # Garder la ligne si elle n'est pas dans une fonction à supprimer
            cleaned_lines.append(line)
        
        # Reconstituer le contenu
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Nettoyer les lignes vides multiples
        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
        
        # Écrire le fichier nettoyé
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        new_lines = len(cleaned_content.split('\n'))
        print(f"📄 Fichier nettoyé: {new_lines} lignes")
        print(f"🗑️ Lignes supprimées: {original_lines - new_lines}")
        
        print("✅ Nettoyage terminé avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    clean_animations() 