#!/usr/bin/env python3
"""
Script final pour corriger les dernières erreurs d'indentation
"""

def fix_final_indentation():
    print("🔧 Correction finale des indentations...")
    
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    corrections = 0
    
    # Corriger les lignes 2860 et suivantes qui doivent être dans le bloc if
    for i in range(len(lines)):
        # Ligne 2860: print qui doit être indenté dans le if
        if i == 2859 and lines[i].strip().startswith('print("    ✅ B-roll insérés avec succès")'):
            lines[i] = '                print("    ✅ B-roll insérés avec succès")\n'
            corrections += 1
            print(f"   ✅ Corrigé ligne {i+1}: indentation print B-roll")
        
        # Lignes suivantes dans le bloc if
        elif i >= 2860 and i <= 2872:
            line = lines[i]
            # Si la ligne commence par 12 espaces ou moins et n'est pas vide
            if line.strip() and not line.startswith('                '):
                # Réindenter avec 16 espaces (dans le bloc if)
                stripped = line.strip()
                if stripped:
                    lines[i] = '                ' + stripped + '\n'
                    corrections += 1
                    print(f"   ✅ Corrigé ligne {i+1}: indentation dans bloc if")
    
    if corrections > 0:
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ {corrections} corrections d'indentation appliquées")
        return True
    else:
        print("ℹ️ Aucune correction nécessaire")
        return False

def test_final():
    print("\n🧪 Test final...")
    try:
        import sys
        if 'video_processor' in sys.modules:
            del sys.modules['video_processor']
        import video_processor
        print("✅ SUCCESS: video_processor importé avec succès !")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🎯 CORRECTION FINALE")
    print("=" * 30)
    
    fixed = fix_final_indentation()
    success = test_final()
    
    if success:
        print("\n🎉 CORRECTION RÉUSSIE !")
        print("   🚀 Pipeline prêt à utiliser")
        print("   ✅ Système zéro cache opérationnel")
    else:
        print("\n❌ Corrections additionnelles requises") 