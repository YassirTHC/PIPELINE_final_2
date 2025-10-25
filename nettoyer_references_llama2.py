#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ðŸ§¹ NETTOYAGE RÃ‰FÃ‰RENCES LLAMA2:13B
Remplace toutes les rÃ©fÃ©rences Ã  qwen3:8b par qwen3:8b dans le code
"""

import os
import re
from pathlib import Path

def nettoyer_fichier(file_path):
    """Nettoie un fichier en remplaÃ§ant qwen3:8b par qwen3:8b"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Compter les occurrences avant
        count_before = content.count('qwen3:8b')
        if count_before == 0:
            return 0, 0
        
        # Remplacer qwen3:8b par qwen3:8b
        content_new = content.replace('qwen3:8b', 'qwen3:8b')
        
        # Compter les occurrences aprÃ¨s
        count_after = content_new.count('qwen3:8b')
        
        # Ã‰crire le fichier modifiÃ©
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_new)
        
        return count_before, count_after
        
    except Exception as e:
        print(f"âŒ Erreur avec {file_path}: {e}")
        return 0, 0

def nettoyer_repertoire():
    """Nettoie tous les fichiers du rÃ©pertoire"""
    
    print("ðŸ§¹ NETTOYAGE RÃ‰FÃ‰RENCES LLAMA2:13B")
    print("=" * 50)
    
    # Extensions de fichiers Ã  traiter
    extensions = ['.py', '.yaml', '.yml', '.md', '.txt', '.bat', '.sh']
    
    # Fichiers Ã  ignorer
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
            
            # VÃ©rifier l'extension
            if file_path.suffix.lower() in extensions:
                count_before, count_after = nettoyer_fichier(file_path)
                if count_before > 0:
                    print(f"âœ… {file_path}: {count_before} â†’ {count_after} rÃ©fÃ©rences")
                    total_files += 1
                    total_replacements += count_before
    
    print(f"\nðŸŽ‰ NETTOYAGE TERMINÃ‰!")
    print(f"ðŸ“ Fichiers traitÃ©s: {total_files}")
    print(f"ðŸ”„ RÃ©fÃ©rences remplacÃ©es: {total_replacements}")
    
    return total_files, total_replacements

if __name__ == "__main__":
    total_files, total_replacements = nettoyer_repertoire()
    
    if total_replacements > 0:
        print(f"\nâœ… {total_replacements} rÃ©fÃ©rences Ã  qwen3:8b ont Ã©tÃ© remplacÃ©es par qwen3:8b")
        print("L'interface devrait maintenant afficher le bon modÃ¨le!")
    else:
        print("\nâ„¹ï¸ Aucune rÃ©fÃ©rence Ã  qwen3:8b trouvÃ©e")
    
    input("\nAppuyez sur EntrÃ©e pour continuer...") 
