#!/usr/bin/env python3
"""
Script de nettoyage MANUEL des morceaux de code orphelins
Supprime tous les fragments de code d'animations qui restent
"""

import re
from pathlib import Path

def clean_orphaned_code():
    """Nettoie les morceaux de code orphelins"""
    
    file_path = Path('video_processor.py')
    
    print("🧹 NETTOYAGE MANUEL DES MORCEAUX ORPHELINS")
    print("=" * 60)
    
    try:
        # Lire le fichier
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_length = len(content)
        print(f"📄 Fichier original: {original_length} caractères")
        
        # Supprimer les morceaux de code orphelins
        patterns_to_remove = [
            # Morceaux de docstring orphelins
            r'"""\s*\n\s*Ajoute automatiquement des animations.*?Path vers la vidéo avec animations et emojis\s*"""',
            # Morceaux de code orphelins
            r'try:\s*\n\s*print\(f"🎬 Ajout d\'animations.*?return video_path',
            # Morceaux de thèmes
            r'# Détecter le thème principal\s*\n\s*themes = \{.*?primary_theme = max\(theme_scores, key=theme_scores\.get\) if theme_scores else \'general\'',
            # Morceaux d'émotions
            r'# Détecter l\'émotion\s*\n\s*emotions = \{.*?primary_emotion = max\(emotion_scores, key=emotion_scores\.get\) if emotion_scores else \'neutral\'',
            # Morceaux de retour
            r'return \{.*?\'emotion_scores\': emotion_scores\s*\}\s*\n\s*except Exception as e:.*?return \{\'theme\': \'general\', \'emotion\': \'neutral\', \'text\': \'\'\}',
            # Morceaux d'animations
            r'# Créer des overlays d\'animations avec timing intelligent\s*\n\s*for i, anim_name in enumerate\(available_animations\[:2\]\):.*?Max 2 animations'
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.DOTALL)
        
        # Nettoyer les lignes vides multiples
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Écrire le fichier nettoyé
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        new_length = len(content)
        print(f"📄 Fichier nettoyé: {new_length} caractères")
        print(f"🗑️ Caractères supprimés: {original_length - new_length}")
        
        print("✅ Nettoyage manuel terminé avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    clean_orphaned_code() 