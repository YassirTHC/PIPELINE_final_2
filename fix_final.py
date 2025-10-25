ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Script final pour corriger les derniÃƒÂ¨res erreurs d'indentation
"""

def fix_final_indentation():
    print("Ã°Å¸â€Â§ Correction finale des indentations...")
    
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    corrections = 0
    
    # Corriger les lignes 2860 et suivantes qui doivent ÃƒÂªtre dans le bloc if
    for i in range(len(lines)):
        # Ligne 2860: print qui doit ÃƒÂªtre indentÃƒÂ© dans le if
        if i == 2859 and lines[i].strip().startswith('print("    Ã¢Å“â€¦ B-roll insÃƒÂ©rÃƒÂ©s avec succÃƒÂ¨s")'):
            lines[i] = '                print("    Ã¢Å“â€¦ B-roll insÃƒÂ©rÃƒÂ©s avec succÃƒÂ¨s")\n'
            corrections += 1
            print(f"   Ã¢Å“â€¦ CorrigÃƒÂ© ligne {i+1}: indentation print B-roll")
        
        # Lignes suivantes dans le bloc if
        elif i >= 2860 and i <= 2872:
            line = lines[i]
            # Si la ligne commence par 12 espaces ou moins et n'est pas vide
            if line.strip() and not line.startswith('                '):
                # RÃƒÂ©indenter avec 16 espaces (dans le bloc if)
                stripped = line.strip()
                if stripped:
                    lines[i] = '                ' + stripped + '\n'
                    corrections += 1
                    print(f"   Ã¢Å“â€¦ CorrigÃƒÂ© ligne {i+1}: indentation dans bloc if")
    
    if corrections > 0:
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"Ã¢Å“â€¦ {corrections} corrections d'indentation appliquÃƒÂ©es")
        return True
    else:
        print("Ã¢â€žÂ¹Ã¯Â¸Â Aucune correction nÃƒÂ©cessaire")
        return False

def test_final():
    print("\nÃ°Å¸Â§Âª Test final...")
    try:
        import sys
        if 'video_processor' in sys.modules:
            del sys.modules['video_processor']
        import video_processor
        print("Ã¢Å“â€¦ SUCCESS: video_processor importÃƒÂ© avec succÃƒÂ¨s !")
        return True
    except Exception as e:
        print(f"Ã¢ÂÅ’ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Ã°Å¸Å½Â¯ CORRECTION FINALE")
    print("=" * 30)
    
    fixed = fix_final_indentation()
    success = test_final()
    
    if success:
        print("\nÃ°Å¸Å½â€° CORRECTION RÃƒâ€°USSIE !")
        print("   Ã°Å¸Å¡â‚¬ Pipeline prÃƒÂªt ÃƒÂ  utiliser")
        print("   Ã¢Å“â€¦ SystÃƒÂ¨me zÃƒÂ©ro cache opÃƒÂ©rationnel")
    else:
        print("\nÃ¢ÂÅ’ Corrections additionnelles requises") 

