ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Script pour corriger la structure try/except dans video_processor.py
"""

def fix_structure():
    print("Ã°Å¸â€Â§ Correction de la structure try/except...")
    
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Correction spÃƒÂ©cifique: le if ÃƒÂ  la ligne 2859 doit ÃƒÂªtre indentÃƒÂ© dans le try
    old_pattern = """            except Exception as e:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur lors de la vÃƒÂ©rification/nettoyage: {e}")
                # En cas d'erreur, ne pas supprimer les B-rolls
                pass

        if Path(cfg.output_video).exists():"""
    
    new_pattern = """            except Exception as e:
                print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur lors de la vÃƒÂ©rification/nettoyage: {e}")
                # En cas d'erreur, ne pas supprimer les B-rolls
                pass

            if Path(cfg.output_video).exists():"""
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print("   Ã¢Å“â€¦ CorrigÃƒÂ©: indentation du bloc if Path(cfg.output_video)")
        
        # Sauvegarder
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    else:
        print("   Ã¢Å¡Â Ã¯Â¸Â Pattern non trouvÃƒÂ© - structure dÃƒÂ©jÃƒÂ  correcte?")
        return False

def test_syntax():
    """Tester la syntaxe aprÃƒÂ¨s correction"""
    print("\nÃ°Å¸Â§Âª Test de syntaxe...")
    try:
        with open('video_processor.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'video_processor.py', 'exec')
        print("Ã¢Å“â€¦ SUCCESS: Syntaxe correcte !")
        return True
    except SyntaxError as e:
        print(f"Ã¢ÂÅ’ SYNTAX ERROR: {e}")
        print(f"   Ligne: {e.lineno}")
        print(f"   Position: {e.offset}")
        if e.text:
            print(f"   Code: {e.text.strip()}")
        return False

if __name__ == "__main__":
    print("Ã°Å¸Å½Â¯ CORRECTION STRUCTURE TRY/EXCEPT")
    print("=" * 40)
    
    # Corriger la structure
    fixed = fix_structure()
    
    # Tester la syntaxe
    success = test_syntax()
    
    # RÃƒÂ©sumÃƒÂ©
    print(f"\nÃ°Å¸Ââ€  RÃƒâ€°SUMÃƒâ€°:")
    if success:
        print("   Ã¢Å“â€¦ Structure corrigÃƒÂ©e avec succÃƒÂ¨s")
        print("   Ã°Å¸Å¡â‚¬ Pipeline syntaxiquement correct")
    else:
        print("   Ã¢ÂÅ’ Erreurs de syntaxe persistantes")
        print("   Ã°Å¸â€Â§ Correction additionnelle requise") 

