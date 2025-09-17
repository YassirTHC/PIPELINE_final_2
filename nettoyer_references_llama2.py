#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 NETTOYAGE RÉFÉRENCES LLAMA2:13B
Remplace toutes les références à qwen3:8b par qwen3:8b dans le code
"""

import os
import re
from pathlib import Path

def nettoyer_fichier(file_path):
    """Nettoie un fichier en remplaçant qwen3:8b par qwen3:8b"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Compter les occurrences avant
        count_before = content.count('qwen3:8b')
        if count_before == 0:
            return 0, 0
        
        # Remplacer qwen3:8b par qwen3:8b
        content_new = content.replace('qwen3:8b', 'qwen3:8b')
        
        # Compter les occurrences après
        count_after = content_new.count('qwen3:8b')
        
        # Écrire le fichier modifié
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_new)
        
        return count_before, count_after
        
    except Exception as e:
        print(f"❌ Erreur avec {file_path}: {e}")
        return 0, 0

def nettoyer_repertoire():
    """Nettoie tous les fichiers du répertoire"""
    
    print("🧹 NETTOYAGE RÉFÉRENCES LLAMA2:13B")
    print("=" * 50)
    
    # Extensions de fichiers à traiter
    extensions = ['.py', '.yaml', '.yml', '.md', '.txt', '.bat', '.sh']
    
    # Fichiers à ignorer
    ignore_files = {
        'test_llama2_13b_prompt_complet.py',
        'test_llama2_13b_prompt_final.py',
        'diagnostic_llama2_13b_json.py',
        'capture_json_llama2_13b.py',
        'golden_sample_json_llama2_13b.py'
    }
    
    total_files = 0
    total_replacements = 0
    
    # Parcourir tous les fichiers
    for root, dirs, files in os.walk('.'):
        # Ignorer les dossiers venv et .git
        if 'venv' in root or '.git' in root:
            continue
            
        for file in files:
            if file in ignore_files:
                continue
                
            file_path = Path(root) / file
            
            # Vérifier l'extension
            if file_path.suffix.lower() in extensions:
                count_before, count_after = nettoyer_fichier(file_path)
                if count_before > 0:
                    print(f"✅ {file_path}: {count_before} → {count_after} références")
                    total_files += 1
                    total_replacements += count_before
    
    print(f"\n🎉 NETTOYAGE TERMINÉ!")
    print(f"📁 Fichiers traités: {total_files}")
    print(f"🔄 Références remplacées: {total_replacements}")
    
    return total_files, total_replacements

if __name__ == "__main__":
    total_files, total_replacements = nettoyer_repertoire()
    
    if total_replacements > 0:
        print(f"\n✅ {total_replacements} références à qwen3:8b ont été remplacées par qwen3:8b")
        print("L'interface devrait maintenant afficher le bon modèle!")
    else:
        print("\nℹ️ Aucune référence à qwen3:8b trouvée")
    
    input("\nAppuyez sur Entrée pour continuer...") 