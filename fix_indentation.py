#!/usr/bin/env python3
"""
Script pour corriger les erreurs d'indentation dans video_processor.py
"""

def fix_indentation():
    print("🔧 Correction des erreurs d'indentation...")
    
    # Lire le fichier
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Corrections spécifiques
    corrections_made = 0
    
    # Correction ligne 2859 (index 2858)
    if len(lines) > 2858:
        if lines[2858].strip().startswith('if Path(cfg.output_video).exists():'):
            lines[2858] = '        if Path(cfg.output_video).exists():\n'
            corrections_made += 1
            print(f"   ✅ Corrigé ligne 2859: indentation 'if Path(cfg.output_video)'")
    
    # Correction ligne 2860 (index 2859)
    if len(lines) > 2859:
        if lines[2859].strip().startswith('print("    ✅ B-roll insérés avec succès")'):
            lines[2859] = '            print("    ✅ B-roll insérés avec succès")\n'
            corrections_made += 1
            print(f"   ✅ Corrigé ligne 2860: indentation print B-roll")
    
    # Vérifier et corriger d'autres problèmes d'indentation potentiels
    for i, line in enumerate(lines):
        # Rechercher des lignes avec des indentations bizarres
        if line.startswith('                                                        '):
            # Ligne avec trop d'espaces - probablement une erreur
            stripped = line.strip()
            if stripped:
                # Déterminer l'indentation appropriée basée sur le contexte
                appropriate_indent = '                        '  # 24 espaces pour le niveau standard
                lines[i] = appropriate_indent + stripped + '\n'
                corrections_made += 1
                print(f"   ✅ Corrigé ligne {i+1}: sur-indentation")
    
    # Sauvegarder si des corrections ont été faites
    if corrections_made > 0:
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ {corrections_made} corrections d'indentation appliquées")
        return True
    else:
        print("ℹ️ Aucune correction d'indentation nécessaire")
        return False

def test_import():
    """Tester l'import après correction"""
    print("\n🧪 Test d'import après correction...")
    try:
        # Supprimer le module du cache s'il existe
        import sys
        if 'video_processor' in sys.modules:
            del sys.modules['video_processor']
        
        # Tenter l'import
        import video_processor
        print("✅ SUCCESS: Import video_processor réussi !")
        return True
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR: {e}")
        print(f"   Fichier: {e.filename}")
        print(f"   Ligne: {e.lineno}")
        print(f"   Position: {e.offset}")
        return False
    except Exception as e:
        print(f"⚠️ OTHER ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🎯 CORRECTION AUTOMATIQUE DES INDENTATIONS")
    print("=" * 50)
    
    # Corriger les indentations
    fixed = fix_indentation()
    
    # Tester l'import
    success = test_import()
    
    # Résumé
    print(f"\n🏆 RÉSUMÉ:")
    if success:
        print("   ✅ Fichier corrigé avec succès")
        print("   🚀 Pipeline prêt à utiliser")
    else:
        print("   ❌ Erreurs persistantes")
        print("   🔧 Correction manuelle requise") 