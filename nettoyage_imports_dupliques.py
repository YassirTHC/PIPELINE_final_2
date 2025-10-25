ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Ã°Å¸Â§Â¹ NETTOYAGE IMPORTS DUPLIQUÃƒâ€°S
Supprime tous les imports dupliquÃƒÂ©s de scoring import *
"""

import re
from pathlib import Path

def nettoyage_imports_dupliques():
    """Nettoie tous les imports dupliquÃƒÂ©s"""
    print("Ã°Å¸Â§Â¹ NETTOYAGE IMPORTS DUPLIQUÃƒâ€°S")
    print("=" * 50)
    
    # Sauvegarde
    backup_path = "video_processor.py.backup_nettoyage_imports"
    if not Path(backup_path).exists():
        import shutil
        shutil.copy2("video_processor.py", backup_path)
        print(f"Ã¢Å“â€¦ Sauvegarde crÃƒÂ©ÃƒÂ©e: {backup_path}")
    
    # Lire le fichier
    with open("video_processor.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    original_content = content
    imports_removed = 0
    
    print("Ã°Å¸â€Â Analyse des imports dupliquÃƒÂ©s...")
    
    # Compter les imports scoring
    scoring_imports = re.findall(r'from scoring import \*', content)
    print(f"Ã°Å¸â€œÅ  Imports 'from scoring import *' trouvÃƒÂ©s: {len(scoring_imports)}")
    
    if len(scoring_imports) > 1:
        print("Ã°Å¸Å¡Â¨ Trop d'imports dupliquÃƒÂ©s dÃƒÂ©tectÃƒÂ©s !")
        
        # Garder seulement le premier import et supprimer les autres
        lines = content.split('\n')
        new_lines = []
        first_scoring_import_found = False
        
        for line in lines:
            if line.strip() == 'from scoring import *':
                if not first_scoring_import_found:
                    new_lines.append(line)
                    first_scoring_import_found = True
                    print("Ã¢Å“â€¦ Premier import scoring conservÃƒÂ©")
                else:
                    print(f"Ã°Å¸â€”â€˜Ã¯Â¸Â Import dupliquÃƒÂ© supprimÃƒÂ©: {line.strip()}")
                    imports_removed += 1
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        print(f"Ã¢Å“â€¦ {imports_removed} imports dupliquÃƒÂ©s supprimÃƒÂ©s")
    else:
        print("Ã¢Å“â€¦ Aucun import dupliquÃƒÂ© dÃƒÂ©tectÃƒÂ©")
    
    # VÃƒÂ©rifier les autres imports dupliquÃƒÂ©s
    print("\nÃ°Å¸â€Â VÃƒÂ©rification autres imports dupliquÃƒÂ©s...")
    
    # Chercher les imports rÃƒÂ©pÃƒÂ©tÃƒÂ©s
    import_patterns = [
        r'from scoring import \*',
        r'import re',
        r'from datetime import datetime',
        r'import numpy as np'
    ]
    
    for pattern in import_patterns:
        matches = re.findall(pattern, content)
        if len(matches) > 1:
            print(f"Ã¢Å¡Â Ã¯Â¸Â {pattern}: {len(matches)} occurrences")
        else:
            print(f"Ã¢Å“â€¦ {pattern}: OK")
    
    # VÃƒÂ©rifier les modifications
    if content != original_content:
        print(f"\nÃ°Å¸â€Â§ {imports_removed} imports dupliquÃƒÂ©s supprimÃƒÂ©s")
        
        # Sauvegarder le fichier nettoyÃƒÂ©
        with open("video_processor.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"Ã¢Å“â€¦ Fichier nettoyÃƒÂ© sauvegardÃƒÂ©")
        
        # CrÃƒÂ©er un rapport
        report_path = "RAPPORT_NETTOYAGE_IMPORTS.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Ã°Å¸Â§Â¹ RAPPORT DE NETTOYAGE DES IMPORTS DUPLIQUÃƒâ€°S\n\n")
            f.write(f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Ã¢Å“â€¦ Imports NettoyÃƒÂ©s\n\n")
            f.write(f"- **Imports scoring supprimÃƒÂ©s:** {imports_removed}\n")
            f.write(f"- **Fichier sauvegardÃƒÂ©:** {backup_path}\n")
            f.write(f"- **Fichier nettoyÃƒÂ©:** video_processor.py\n")
            f.write(f"- **Rapport:** {report_path}\n")
        
        print(f"Ã°Å¸â€œâ€¹ Rapport de nettoyage crÃƒÂ©ÃƒÂ©: {report_path}")
        
        return True
    else:
        print("\nÃ¢Å“â€¦ Aucun nettoyage nÃƒÂ©cessaire")
        return False

def verification_post_nettoyage():
    """VÃƒÂ©rification aprÃƒÂ¨s nettoyage"""
    print("\nÃ°Å¸â€Â VÃƒâ€°RIFICATION POST-NETTOYAGE")
    print("=" * 40)
    
    with open("video_processor.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # VÃƒÂ©rifier les imports scoring
    scoring_imports = re.findall(r'from scoring import \*', content)
    print(f"Ã°Å¸â€œÅ  Imports scoring restants: {len(scoring_imports)}")
    
    if len(scoring_imports) == 1:
        print("Ã¢Å“â€¦ Un seul import scoring (correct)")
    else:
        print(f"Ã¢Å¡Â Ã¯Â¸Â {len(scoring_imports)} imports scoring (problÃƒÂ©matique)")
    
    # VÃƒÂ©rifier la syntaxe
    print("\nÃ°Å¸â€Â VÃƒÂ©rification syntaxe...")
    
    try:
        # Essayer de compiler le fichier
        compile(content, 'video_processor.py', 'exec')
        print("Ã¢Å“â€¦ Syntaxe Python correcte")
    except SyntaxError as e:
        print(f"Ã¢ÂÅ’ Erreur de syntaxe: {e}")
        return False
    
    print("\nÃ°Å¸Å½Â¯ VÃƒÂ©rification terminÃƒÂ©e")
    return True

if __name__ == "__main__":
    print("Ã°Å¸Å¡â‚¬ DÃƒâ€°MARRAGE NETTOYAGE IMPORTS DUPLIQUÃƒâ€°S")
    print("=" * 60)
    
    try:
        success = nettoyage_imports_dupliques()
        if success:
            verification_post_nettoyage()
            print("\nÃ°Å¸Å½â€° NETTOYAGE TERMINÃƒâ€° AVEC SUCCÃƒË†S!")
        else:
            print("\nÃ¢Å“â€¦ Aucun nettoyage nÃƒÂ©cessaire")
            
    except Exception as e:
        print(f"\nÃ¢ÂÅ’ Erreur lors du nettoyage: {e}")
        import traceback
        traceback.print_exc() 

