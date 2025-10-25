ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Script pour corriger les erreurs d'indentation dans video_processor.py
"""

def fix_indentation():
    print("Ã°Å¸â€Â§ Correction des erreurs d'indentation...")
    
    # Lire le fichier
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Corrections spÃƒÂ©cifiques
    corrections_made = 0
    
    # Correction ligne 2859 (index 2858)
    if len(lines) > 2858:
        if lines[2858].strip().startswith('if Path(cfg.output_video).exists():'):
            lines[2858] = '        if Path(cfg.output_video).exists():\n'
            corrections_made += 1
            print(f"   Ã¢Å“â€¦ CorrigÃƒÂ© ligne 2859: indentation 'if Path(cfg.output_video)'")
    
    # Correction ligne 2860 (index 2859)
    if len(lines) > 2859:
        if lines[2859].strip().startswith('print("    Ã¢Å“â€¦ B-roll insÃƒÂ©rÃƒÂ©s avec succÃƒÂ¨s")'):
            lines[2859] = '            print("    Ã¢Å“â€¦ B-roll insÃƒÂ©rÃƒÂ©s avec succÃƒÂ¨s")\n'
            corrections_made += 1
            print(f"   Ã¢Å“â€¦ CorrigÃƒÂ© ligne 2860: indentation print B-roll")
    
    # VÃƒÂ©rifier et corriger d'autres problÃƒÂ¨mes d'indentation potentiels
    for i, line in enumerate(lines):
        # Rechercher des lignes avec des indentations bizarres
        if line.startswith('                                                        '):
            # Ligne avec trop d'espaces - probablement une erreur
            stripped = line.strip()
            if stripped:
                # DÃƒÂ©terminer l'indentation appropriÃƒÂ©e basÃƒÂ©e sur le contexte
                appropriate_indent = '                        '  # 24 espaces pour le niveau standard
                lines[i] = appropriate_indent + stripped + '\n'
                corrections_made += 1
                print(f"   Ã¢Å“â€¦ CorrigÃƒÂ© ligne {i+1}: sur-indentation")
    
    # Sauvegarder si des corrections ont ÃƒÂ©tÃƒÂ© faites
    if corrections_made > 0:
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"Ã¢Å“â€¦ {corrections_made} corrections d'indentation appliquÃƒÂ©es")
        return True
    else:
        print("Ã¢â€žÂ¹Ã¯Â¸Â Aucune correction d'indentation nÃƒÂ©cessaire")
        return False

def test_import():
    """Tester l'import aprÃƒÂ¨s correction"""
    print("\nÃ°Å¸Â§Âª Test d'import aprÃƒÂ¨s correction...")
    try:
        # Supprimer le module du cache s'il existe
        import sys
        if 'video_processor' in sys.modules:
            del sys.modules['video_processor']
        
        # Tenter l'import
        import video_processor
        print("Ã¢Å“â€¦ SUCCESS: Import video_processor rÃƒÂ©ussi !")
        return True
    except SyntaxError as e:
        print(f"Ã¢ÂÅ’ SYNTAX ERROR: {e}")
        print(f"   Fichier: {e.filename}")
        print(f"   Ligne: {e.lineno}")
        print(f"   Position: {e.offset}")
        return False
    except Exception as e:
        print(f"Ã¢Å¡Â Ã¯Â¸Â OTHER ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Ã°Å¸Å½Â¯ CORRECTION AUTOMATIQUE DES INDENTATIONS")
    print("=" * 50)
    
    # Corriger les indentations
    fixed = fix_indentation()
    
    # Tester l'import
    success = test_import()
    
    # RÃƒÂ©sumÃƒÂ©
    print(f"\nÃ°Å¸Ââ€  RÃƒâ€°SUMÃƒâ€°:")
    if success:
        print("   Ã¢Å“â€¦ Fichier corrigÃƒÂ© avec succÃƒÂ¨s")
        print("   Ã°Å¸Å¡â‚¬ Pipeline prÃƒÂªt ÃƒÂ  utiliser")
    else:
        print("   Ã¢ÂÅ’ Erreurs persistantes")
        print("   Ã°Å¸â€Â§ Correction manuelle requise") 

