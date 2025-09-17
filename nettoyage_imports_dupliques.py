#!/usr/bin/env python3
"""
🧹 NETTOYAGE IMPORTS DUPLIQUÉS
Supprime tous les imports dupliqués de scoring import *
"""

import re
from pathlib import Path

def nettoyage_imports_dupliques():
    """Nettoie tous les imports dupliqués"""
    print("🧹 NETTOYAGE IMPORTS DUPLIQUÉS")
    print("=" * 50)
    
    # Sauvegarde
    backup_path = "video_processor.py.backup_nettoyage_imports"
    if not Path(backup_path).exists():
        import shutil
        shutil.copy2("video_processor.py", backup_path)
        print(f"✅ Sauvegarde créée: {backup_path}")
    
    # Lire le fichier
    with open("video_processor.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    original_content = content
    imports_removed = 0
    
    print("🔍 Analyse des imports dupliqués...")
    
    # Compter les imports scoring
    scoring_imports = re.findall(r'from scoring import \*', content)
    print(f"📊 Imports 'from scoring import *' trouvés: {len(scoring_imports)}")
    
    if len(scoring_imports) > 1:
        print("🚨 Trop d'imports dupliqués détectés !")
        
        # Garder seulement le premier import et supprimer les autres
        lines = content.split('\n')
        new_lines = []
        first_scoring_import_found = False
        
        for line in lines:
            if line.strip() == 'from scoring import *':
                if not first_scoring_import_found:
                    new_lines.append(line)
                    first_scoring_import_found = True
                    print("✅ Premier import scoring conservé")
                else:
                    print(f"🗑️ Import dupliqué supprimé: {line.strip()}")
                    imports_removed += 1
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        print(f"✅ {imports_removed} imports dupliqués supprimés")
    else:
        print("✅ Aucun import dupliqué détecté")
    
    # Vérifier les autres imports dupliqués
    print("\n🔍 Vérification autres imports dupliqués...")
    
    # Chercher les imports répétés
    import_patterns = [
        r'from scoring import \*',
        r'import re',
        r'from datetime import datetime',
        r'import numpy as np'
    ]
    
    for pattern in import_patterns:
        matches = re.findall(pattern, content)
        if len(matches) > 1:
            print(f"⚠️ {pattern}: {len(matches)} occurrences")
        else:
            print(f"✅ {pattern}: OK")
    
    # Vérifier les modifications
    if content != original_content:
        print(f"\n🔧 {imports_removed} imports dupliqués supprimés")
        
        # Sauvegarder le fichier nettoyé
        with open("video_processor.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✅ Fichier nettoyé sauvegardé")
        
        # Créer un rapport
        report_path = "RAPPORT_NETTOYAGE_IMPORTS.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 🧹 RAPPORT DE NETTOYAGE DES IMPORTS DUPLIQUÉS\n\n")
            f.write(f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## ✅ Imports Nettoyés\n\n")
            f.write(f"- **Imports scoring supprimés:** {imports_removed}\n")
            f.write(f"- **Fichier sauvegardé:** {backup_path}\n")
            f.write(f"- **Fichier nettoyé:** video_processor.py\n")
            f.write(f"- **Rapport:** {report_path}\n")
        
        print(f"📋 Rapport de nettoyage créé: {report_path}")
        
        return True
    else:
        print("\n✅ Aucun nettoyage nécessaire")
        return False

def verification_post_nettoyage():
    """Vérification après nettoyage"""
    print("\n🔍 VÉRIFICATION POST-NETTOYAGE")
    print("=" * 40)
    
    with open("video_processor.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Vérifier les imports scoring
    scoring_imports = re.findall(r'from scoring import \*', content)
    print(f"📊 Imports scoring restants: {len(scoring_imports)}")
    
    if len(scoring_imports) == 1:
        print("✅ Un seul import scoring (correct)")
    else:
        print(f"⚠️ {len(scoring_imports)} imports scoring (problématique)")
    
    # Vérifier la syntaxe
    print("\n🔍 Vérification syntaxe...")
    
    try:
        # Essayer de compiler le fichier
        compile(content, 'video_processor.py', 'exec')
        print("✅ Syntaxe Python correcte")
    except SyntaxError as e:
        print(f"❌ Erreur de syntaxe: {e}")
        return False
    
    print("\n🎯 Vérification terminée")
    return True

if __name__ == "__main__":
    print("🚀 DÉMARRAGE NETTOYAGE IMPORTS DUPLIQUÉS")
    print("=" * 60)
    
    try:
        success = nettoyage_imports_dupliques()
        if success:
            verification_post_nettoyage()
            print("\n🎉 NETTOYAGE TERMINÉ AVEC SUCCÈS!")
        else:
            print("\n✅ Aucun nettoyage nécessaire")
            
    except Exception as e:
        print(f"\n❌ Erreur lors du nettoyage: {e}")
        import traceback
        traceback.print_exc() 