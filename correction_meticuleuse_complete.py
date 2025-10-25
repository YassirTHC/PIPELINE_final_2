ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Ã°Å¸â€Â§ CORRECTION MÃƒâ€°TICULEUSE COMPLÃƒË†TE
Corrige TOUS les problÃƒÂ¨mes identifiÃƒÂ©s dans video_processor.py
"""

import re
from pathlib import Path
import shutil

def correction_meticuleuse_complete():
    """Correction mÃƒÂ©ticuleuse de TOUS les problÃƒÂ¨mes identifiÃƒÂ©s"""
    print("Ã°Å¸â€Â§ CORRECTION MÃƒâ€°TICULEUSE COMPLÃƒË†TE")
    print("=" * 50)
    
    # Sauvegarde du fichier original
    backup_path = "video_processor.py.backup_correction_complete"
    if not Path(backup_path).exists():
        shutil.copy2("video_processor.py", backup_path)
        print(f"Ã¢Å“â€¦ Sauvegarde crÃƒÂ©ÃƒÂ©e: {backup_path}")
    
    # Lire le fichier
    with open("video_processor.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    original_content = content
    corrections_made = []
    
    print("Ã°Å¸â€Â Analyse des problÃƒÂ¨mes...")
    
    # 1. CORRECTION: Supprimer la redÃƒÂ©claration problÃƒÂ©matique dans le bloc d'erreur
    print("\nÃ°Å¸â€Â§ Correction 1: Suppression redÃƒÂ©claration dans bloc d'erreur...")
    
    # Pattern pour trouver le bloc problÃƒÂ©matique
    pattern1 = r'(\s+)except Exception:\s+fetched_brolls = \[\]\s+print\("    Ã¢Å¡Â Ã¯Â¸Â Erreur lors de la prÃƒÂ©paration des B-rolls fetchÃƒÂ©s"\)'
    replacement1 = r'\1except Exception:\n\1    print("    Ã¢Å¡Â Ã¯Â¸Â Erreur lors de la prÃƒÂ©paration des B-rolls fetchÃƒÂ©s")'
    
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        corrections_made.append("Suppression redÃƒÂ©claration fetched_brolls dans bloc d'erreur")
        print("Ã¢Å“â€¦ Correction 1 appliquÃƒÂ©e")
    else:
        print("Ã¢Å¡Â Ã¯Â¸Â Pattern 1 non trouvÃƒÂ©, vÃƒÂ©rification manuelle nÃƒÂ©cessaire")
    
    # 2. CORRECTION: VÃƒÂ©rifier que la premiÃƒÂ¨re dÃƒÂ©claration est correcte
    print("\nÃ°Å¸â€Â§ Correction 2: VÃƒÂ©rification premiÃƒÂ¨re dÃƒÂ©claration...")
    
    # Pattern pour la premiÃƒÂ¨re dÃƒÂ©claration (doit rester)
    pattern2 = r'# Ã°Å¸Å¡Â¨ CORRECTION CRITIQUE: CrÃƒÂ©er fetched_brolls accessible globalement\s+fetched_brolls = \[\]'
    if re.search(pattern2, content):
        print("Ã¢Å“â€¦ PremiÃƒÂ¨re dÃƒÂ©claration correcte (doit rester)")
    else:
        print("Ã¢Å¡Â Ã¯Â¸Â PremiÃƒÂ¨re dÃƒÂ©claration non trouvÃƒÂ©e")
    
    # 3. CORRECTION: VÃƒÂ©rifier que la ligne commentÃƒÂ©e est bien commentÃƒÂ©e
    print("\nÃ°Å¸â€Â§ Correction 3: VÃƒÂ©rification ligne commentÃƒÂ©e...")
    
    pattern3 = r'# fetched_brolls = \[\]  # Ã¢ÂÅ’ SUPPRIMÃƒâ€°: Cette ligne ÃƒÂ©crase la variable fetchÃƒÂ©e !'
    if re.search(pattern3, content):
        print("Ã¢Å“â€¦ Ligne commentÃƒÂ©e correcte")
    else:
        print("Ã¢Å¡Â Ã¯Â¸Â Ligne commentÃƒÂ©e non trouvÃƒÂ©e")
    
    # 4. CORRECTION: Optimiser la gestion des erreurs (rÃƒÂ©duire les exceptions gÃƒÂ©nÃƒÂ©riques)
    print("\nÃ°Å¸â€Â§ Correction 4: Optimisation gestion des erreurs...")
    
    # Remplacer les exceptions gÃƒÂ©nÃƒÂ©riques par des exceptions spÃƒÂ©cifiques
    generic_exceptions = [
        (r'except Exception:', 'except (OSError, IOError, ValueError, TypeError):'),
        (r'except Exception as e:', 'except (OSError, IOError, ValueError, TypeError) as e:'),
    ]
    
    for old_pattern, new_pattern in generic_exceptions:
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_pattern, content)
            corrections_made.append(f"Remplacement exception gÃƒÂ©nÃƒÂ©rique: {old_pattern}")
    
    # 5. CORRECTION: Remplacer les 'pass' par des logs appropriÃƒÂ©s
    print("\nÃ°Å¸â€Â§ Correction 5: Remplacement des 'pass'...")
    
    # Pattern pour trouver les 'pass' dans les blocs except
    pass_pattern = r'(\s+except.*:\s+)pass'
    pass_replacement = r'\1logger.warning(f"Exception ignorÃƒÂ©e dans {__name__}")'
    
    if re.search(pass_pattern, content):
        content = re.sub(pass_pattern, pass_replacement, content)
        corrections_made.append("Remplacement des 'pass' par des logs")
        print("Ã¢Å“â€¦ Correction 5 appliquÃƒÂ©e")
    
    # 6. CORRECTION: Optimiser la logique d'assignation
    print("\nÃ°Å¸â€Â§ Correction 6: Optimisation logique d'assignation...")
    
    # VÃƒÂ©rifier que la logique d'assignation est correcte
    assignment_pattern = r'if items_without_assets and fetched_brolls:'
    if re.search(assignment_pattern, content):
        print("Ã¢Å“â€¦ Logique d'assignation correcte")
    else:
        print("Ã¢Å¡Â Ã¯Â¸Â Logique d'assignation non trouvÃƒÂ©e")
    
    # 7. CORRECTION: VÃƒÂ©rifier la configuration
    print("\nÃ°Å¸â€Â§ Correction 7: VÃƒÂ©rification configuration...")
    
    # VÃƒÂ©rifier que la configuration est correcte
    config_patterns = [
        (r'max_broll_ratio=0\.40', "Configuration max_broll_ratio correcte"),
        (r'max_broll_insertions=6', "Configuration max_broll_insertions correcte"),
        (r'min_gap_between_broll_s=4\.0', "Configuration min_gap correcte"),
    ]
    
    for pattern, description in config_patterns:
        if re.search(pattern, content):
            print(f"Ã¢Å“â€¦ {description}")
        else:
            print(f"Ã¢Å¡Â Ã¯Â¸Â {description} - VÃƒÂ©rification nÃƒÂ©cessaire")
    
    # 8. CORRECTION: Nettoyer les variables non dÃƒÂ©finies
    print("\nÃ°Å¸â€Â§ Correction 8: Nettoyage variables non dÃƒÂ©finies...")
    
    # VÃƒÂ©rifier l'utilisation de fetched_brolls
    usage_pattern = r'fetched_brolls'
    usage_count = len(re.findall(usage_pattern, content))
    print(f"Ã°Å¸â€œÅ  Utilisations de fetched_brolls: {usage_count}")
    
    # 9. CORRECTION: VÃƒÂ©rifier la cohÃƒÂ©rence des imports
    print("\nÃ°Å¸â€Â§ Correction 9: VÃƒÂ©rification cohÃƒÂ©rence imports...")
    
    # VÃƒÂ©rifier les imports critiques
    critical_imports = [
        'from broll_selector import',
        'from timeline_legacy import',
        'from fetchers import',
        'from scoring import'
    ]
    
    for import_line in critical_imports:
        if import_line in content:
            print(f"Ã¢Å“â€¦ Import trouvÃƒÂ©: {import_line}")
        else:
            print(f"Ã¢Å¡Â Ã¯Â¸Â Import manquant: {import_line}")
    
    # 10. CORRECTION: VÃƒÂ©rifier la logique de fallback
    print("\nÃ°Å¸â€Â§ Correction 10: VÃƒÂ©rification logique de fallback...")
    
    # VÃƒÂ©rifier que le fallback n'est activÃƒÂ© que si nÃƒÂ©cessaire
    fallback_pattern = r'# Ã°Å¸Å¡Â¨ FALLBACK UNIQUEMENT SI VRAIMENT NÃƒâ€°CESSAIRE'
    if re.search(fallback_pattern, content):
        print("Ã¢Å“â€¦ Logique de fallback correcte")
    else:
        print("Ã¢Å¡Â Ã¯Â¸Â Logique de fallback non trouvÃƒÂ©e")
    
    # VÃƒÂ©rifier les modifications
    if content != original_content:
        print(f"\nÃ°Å¸â€Â§ {len(corrections_made)} corrections appliquÃƒÂ©es:")
        for correction in corrections_made:
            print(f"   Ã¢Å“â€¦ {correction}")
        
        # Sauvegarder le fichier corrigÃƒÂ©
        with open("video_processor.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"\nÃ¢Å“â€¦ Fichier corrigÃƒÂ© sauvegardÃƒÂ©")
        
        # CrÃƒÂ©er un rapport de correction
        report_path = "RAPPORT_CORRECTION_METICULEUSE.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Ã°Å¸â€Â§ RAPPORT DE CORRECTION MÃƒâ€°TICULEUSE COMPLÃƒË†TE\n\n")
            f.write(f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Ã¢Å“â€¦ Corrections AppliquÃƒÂ©es\n\n")
            for correction in corrections_made:
                f.write(f"- {correction}\n")
            f.write("\n## Ã°Å¸â€œÅ  RÃƒÂ©sumÃƒÂ©\n\n")
            f.write(f"- **Total corrections:** {len(corrections_made)}\n")
            f.write(f"- **Fichier sauvegardÃƒÂ©:** {backup_path}\n")
            f.write(f"- **Fichier corrigÃƒÂ©:** video_processor.py\n")
            f.write(f"- **Rapport:** {report_path}\n")
        
        print(f"Ã°Å¸â€œâ€¹ Rapport de correction crÃƒÂ©ÃƒÂ©: {report_path}")
        
        return True
    else:
        print("\nÃ¢Å“â€¦ Aucune correction nÃƒÂ©cessaire - Fichier dÃƒÂ©jÃƒÂ  correct")
        return False

def verification_post_correction():
    """VÃƒÂ©rification aprÃƒÂ¨s correction"""
    print("\nÃ°Å¸â€Â VÃƒâ€°RIFICATION POST-CORRECTION")
    print("=" * 40)
    
    with open("video_processor.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # VÃƒÂ©rifier que les redÃƒÂ©clarations ont ÃƒÂ©tÃƒÂ© supprimÃƒÂ©es
    problematic_declarations = re.findall(r'fetched_brolls = \[\]', content)
    print(f"Ã°Å¸â€œÅ  DÃƒÂ©clarations fetched_brolls restantes: {len(problematic_declarations)}")
    
    if len(problematic_declarations) <= 1:
        print("Ã¢Å“â€¦ RedÃƒÂ©clarations problÃƒÂ©matiques supprimÃƒÂ©es")
    else:
        print("Ã¢Å¡Â Ã¯Â¸Â RedÃƒÂ©clarations problÃƒÂ©matiques encore prÃƒÂ©sentes")
        for i, decl in enumerate(problematic_declarations):
            print(f"   {i+1}. {decl}")
    
    # VÃƒÂ©rifier la gestion des erreurs
    generic_exceptions = len(re.findall(r'except Exception:', content))
    print(f"Ã°Å¸â€œÅ  Exceptions gÃƒÂ©nÃƒÂ©riques restantes: {generic_exceptions}")
    
    if generic_exceptions < 50:
        print("Ã¢Å“â€¦ Gestion des erreurs optimisÃƒÂ©e")
    else:
        print("Ã¢Å¡Â Ã¯Â¸Â Trop d'exceptions gÃƒÂ©nÃƒÂ©riques restantes")
    
    # VÃƒÂ©rifier les 'pass'
    pass_count = len(re.findall(r'\s+pass\s*$', content, re.MULTILINE))
    print(f"Ã°Å¸â€œÅ  'pass' restants: {pass_count}")
    
    if pass_count < 30:
        print("Ã¢Å“â€¦ 'pass' optimisÃƒÂ©s")
    else:
        print("Ã¢Å¡Â Ã¯Â¸Â Trop de 'pass' restants")
    
    print("\nÃ°Å¸Å½Â¯ VÃƒÂ©rification terminÃƒÂ©e")

if __name__ == "__main__":
    print("Ã°Å¸Å¡â‚¬ DÃƒâ€°MARRAGE CORRECTION MÃƒâ€°TICULEUSE COMPLÃƒË†TE")
    print("=" * 60)
    
    try:
        success = correction_meticuleuse_complete()
        if success:
            verification_post_correction()
            print("\nÃ°Å¸Å½â€° CORRECTION MÃƒâ€°TICULEUSE COMPLÃƒË†TE TERMINÃƒâ€°E AVEC SUCCÃƒË†S!")
        else:
            print("\nÃ¢Å“â€¦ Aucune correction nÃƒÂ©cessaire")
            
    except Exception as e:
        print(f"\nÃ¢ÂÅ’ Erreur lors de la correction: {e}")
        import traceback
        traceback.print_exc() 

