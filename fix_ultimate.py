#!/usr/bin/env python3
"""
Script ultime pour corriger définitivement les indentations
"""

def fix_ultimate():
    print("🔧 Correction ultime des indentations...")
    
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    corrections = 0
    
    # Corrections spécifiques ligne par ligne
    fixes = [
        # (ligne_index, nouveau_contenu)
        (2863, '                    if \'clip_broll_dir\' in locals() and clip_broll_dir.exists():\n'),  # dans le try
        (2864, '                        folder_size = sum(f.stat().st_size for f in clip_broll_dir.rglob(\'*\') if f.is_file()) / (1024**2)  # MB\n'),
        (2865, '                        shutil.rmtree(clip_broll_dir)\n'),
        (2866, '                        print(f"    🗑️ Cache B-roll nettoyé: {folder_size:.1f} MB libérés")\n'),
        (2867, '                        print(f"    💾 Dossier temporaire supprimé: {clip_broll_dir.name}")\n'),
        (2868, '                except Exception as e:\n'),
        (2869, '                    print(f"    ⚠️ Erreur nettoyage cache: {e}")\n'),
        (2871, '                return Path(cfg.output_video)\n'),
        (2872, '            else:\n'),
        (2873, '                print("    ⚠️ Sortie B-roll introuvable, retour à la vidéo d\'origine")\n'),
    ]
    
    for line_idx, new_content in fixes:
        if line_idx < len(lines):
            old_content = lines[line_idx].strip()
            if old_content:  # Ne modifier que si la ligne n'est pas vide
                lines[line_idx] = new_content
                corrections += 1
                print(f"   ✅ Corrigé ligne {line_idx+1}: {old_content[:50]}...")
    
    if corrections > 0:
        with open('video_processor.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ {corrections} corrections appliquées")
        return True
    else:
        print("ℹ️ Aucune correction nécessaire")
        return False

def test_ultimate():
    print("\n🧪 Test ultime...")
    try:
        import sys
        if 'video_processor' in sys.modules:
            del sys.modules['video_processor']
        
        with open('video_processor.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'video_processor.py', 'exec')
        
        import video_processor
        print("✅ SUCCESS: video_processor syntaxiquement correct et importé !")
        return True
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR: {e}")
        print(f"   Ligne: {e.lineno}")
        return False
    except Exception as e:
        print(f"❌ IMPORT ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🎯 CORRECTION ULTIME")
    print("=" * 30)
    
    fixed = fix_ultimate()
    success = test_ultimate()
    
    if success:
        print("\n🎉 CORRECTION DÉFINITIVE RÉUSSIE !")
        print("   ✅ Pipeline syntaxiquement correct")
        print("   🚀 Système zéro cache opérationnel")
        print("   💾 Prêt pour traitement vidéo")
    else:
        print("\n❌ Intervention manuelle requise") 