#!/usr/bin/env python3
"""
Script pour corriger la structure try/except dans video_processor.py
"""

def fix_structure():
    print("🔧 Correction de la structure try/except...")
    
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Correction spécifique: le if à la ligne 2859 doit être indenté dans le try
    old_pattern = """            except Exception as e:
                print(f"    ⚠️ Erreur lors de la vérification/nettoyage: {e}")
                # En cas d'erreur, ne pas supprimer les B-rolls
                pass

        if Path(cfg.output_video).exists():"""
    
    new_pattern = """            except Exception as e:
                print(f"    ⚠️ Erreur lors de la vérification/nettoyage: {e}")
                # En cas d'erreur, ne pas supprimer les B-rolls
                pass

            if Path(cfg.output_video).exists():"""
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print("   ✅ Corrigé: indentation du bloc if Path(cfg.output_video)")
        
        # Sauvegarder
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    else:
        print("   ⚠️ Pattern non trouvé - structure déjà correcte?")
        return False

def test_syntax():
    """Tester la syntaxe après correction"""
    print("\n🧪 Test de syntaxe...")
    try:
        with open('video_processor.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'video_processor.py', 'exec')
        print("✅ SUCCESS: Syntaxe correcte !")
        return True
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR: {e}")
        print(f"   Ligne: {e.lineno}")
        print(f"   Position: {e.offset}")
        if e.text:
            print(f"   Code: {e.text.strip()}")
        return False

if __name__ == "__main__":
    print("🎯 CORRECTION STRUCTURE TRY/EXCEPT")
    print("=" * 40)
    
    # Corriger la structure
    fixed = fix_structure()
    
    # Tester la syntaxe
    success = test_syntax()
    
    # Résumé
    print(f"\n🏆 RÉSUMÉ:")
    if success:
        print("   ✅ Structure corrigée avec succès")
        print("   🚀 Pipeline syntaxiquement correct")
    else:
        print("   ❌ Erreurs de syntaxe persistantes")
        print("   🔧 Correction additionnelle requise") 