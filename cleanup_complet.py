#!/usr/bin/env python3
"""
Script de nettoyage COMPLET des fonctions d'animations
Supprime TOUTES les fonctions d'animations problématiques
"""

from pathlib import Path

def clean_animations_complet():
    """Nettoie COMPLÈTEMENT le fichier video_processor.py des fonctions d'animations"""
    
    file_path = Path('video_processor.py')
    
    print("🧹 NETTOYAGE COMPLET DES FONCTIONS D'ANIMATIONS")
    print("=" * 60)
    
    try:
        # Lire le fichier
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        original_lines = len(lines)
        print(f"📄 Fichier original: {original_lines} lignes")
        
        # Identifier les lignes à supprimer
        lines_to_keep = []
        in_function_to_remove = False
        current_function = None
        
        for i, line in enumerate(lines):
            # Vérifier si on commence une fonction à supprimer
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
                print(f"🗑️ Suppression de: {current_function}")
                continue
            
            # Si on est dans une fonction à supprimer
            if in_function_to_remove:
                # Vérifier si on a atteint la fin de la fonction
                if (line.strip() == '' or 
                    (line.strip().startswith('def ') and not line.strip().startswith(current_function)) or
                    (line.strip().startswith('class ') and not line.strip().startswith(current_function))):
                    in_function_to_remove = False
                    current_function = None
                continue
            
            # Garder la ligne si elle n'est pas dans une fonction à supprimer
            lines_to_keep.append(line)
        
        # Écrire le fichier nettoyé
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines_to_keep)
        
        new_lines = len(lines_to_keep)
        print(f"📄 Fichier nettoyé: {new_lines} lignes")
        print(f"🗑️ Lignes supprimées: {original_lines - new_lines}")
        
        print("✅ Nettoyage COMPLET terminé avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    clean_animations_complet() 