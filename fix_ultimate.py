ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Script ultime pour corriger dÃƒÂ©finitivement les indentations
"""

def fix_ultimate():
    print("Ã°Å¸â€Â§ Correction ultime des indentations...")
    
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    corrections = 0
    
    # Corrections spÃƒÂ©cifiques ligne par ligne
    fixes = [
        # (ligne_index, nouveau_contenu)
        (2863, '                    if \'clip_broll_dir\' in locals() and clip_broll_dir.exists():\n'),  # dans le try
        (2864, '                        folder_size = sum(f.stat().st_size for f in clip_broll_dir.rglob(\'*\') if f.is_file()) / (1024**2)  # MB\n'),
        (2865, '                        shutil.rmtree(clip_broll_dir)\n'),
        (2866, '                        print(f"    Ã°Å¸â€”â€˜Ã¯Â¸Â Cache B-roll nettoyÃƒÂ©: {folder_size:.1f} MB libÃƒÂ©rÃƒÂ©s")\n'),
        (2867, '                        print(f"    Ã°Å¸â€™Â¾ Dossier temporaire supprimÃƒÂ©: {clip_broll_dir.name}")\n'),
        (2868, '                except Exception as e:\n'),
        (2869, '                    print(f"    Ã¢Å¡Â Ã¯Â¸Â Erreur nettoyage cache: {e}")\n'),
        (2871, '                return Path(cfg.output_video)\n'),
        (2872, '            else:\n'),
        (2873, '                print("    Ã¢Å¡Â Ã¯Â¸Â Sortie B-roll introuvable, retour ÃƒÂ  la vidÃƒÂ©o d\'origine")\n'),
    ]
    
    for line_idx, new_content in fixes:
        if line_idx < len(lines):
            old_content = lines[line_idx].strip()
            if old_content:  # Ne modifier que si la ligne n'est pas vide
                lines[line_idx] = new_content
                corrections += 1
                print(f"   Ã¢Å“â€¦ CorrigÃƒÂ© ligne {line_idx+1}: {old_content[:50]}...")
    
    if corrections > 0:
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"Ã¢Å“â€¦ {corrections} corrections appliquÃƒÂ©es")
        return True
    else:
        print("Ã¢â€žÂ¹Ã¯Â¸Â Aucune correction nÃƒÂ©cessaire")
        return False

def test_ultimate():
    print("\nÃ°Å¸Â§Âª Test ultime...")
    try:
        import sys
        if 'video_processor' in sys.modules:
            del sys.modules['video_processor']
        
        with open('video_processor.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'video_processor.py', 'exec')
        
        import video_processor
        print("Ã¢Å“â€¦ SUCCESS: video_processor syntaxiquement correct et importÃƒÂ© !")
        return True
    except SyntaxError as e:
        print(f"Ã¢ÂÅ’ SYNTAX ERROR: {e}")
        print(f"   Ligne: {e.lineno}")
        return False
    except Exception as e:
        print(f"Ã¢ÂÅ’ IMPORT ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Ã°Å¸Å½Â¯ CORRECTION ULTIME")
    print("=" * 30)
    
    fixed = fix_ultimate()
    success = test_ultimate()
    
    if success:
        print("\nÃ°Å¸Å½â€° CORRECTION DÃƒâ€°FINITIVE RÃƒâ€°USSIE !")
        print("   Ã¢Å“â€¦ Pipeline syntaxiquement correct")
        print("   Ã°Å¸Å¡â‚¬ SystÃƒÂ¨me zÃƒÂ©ro cache opÃƒÂ©rationnel")
        print("   Ã°Å¸â€™Â¾ PrÃƒÂªt pour traitement vidÃƒÂ©o")
    else:
        print("\nÃ¢ÂÅ’ Intervention manuelle requise") 

