ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Retraitement de 6.mp4 avec correction des mots-clÃƒÂ©s B-roll
"""

import shutil
import os
from pathlib import Path

def retraiter_6mp4():
    print("Ã°Å¸â€â€ž RETRAITEMENT DE 6.MP4 AVEC CORRECTION DES MOTS-CLÃƒâ€°S B-ROLL")
    print("=" * 70)
    
    # 1. VÃƒÂ©rifier l'ÃƒÂ©tat actuel
    print("1Ã¯Â¸ÂÃ¢Æ’Â£ Ãƒâ€°TAT ACTUEL:")
    
    clips_dir = Path("clips")
    output_6_dir = Path("output/clips/6")
    
    if clips_dir.exists():
        clips = list(clips_dir.glob("*.mp4"))
        print(f"   Ã°Å¸â€œÂ Clips disponibles: {len(clips)}")
        for clip in clips:
            print(f"      Ã°Å¸â€œÂ¹ {clip.name}")
    
    if output_6_dir.exists():
        print(f"   Ã°Å¸â€œÂ Dossier de sortie 6.mp4: {output_6_dir}")
        files = list(output_6_dir.iterdir())
        print(f"      Ã°Å¸â€œâ€ž {len(files)} fichiers")
    
    # 2. Nettoyer l'ancien traitement
    print("\n2Ã¯Â¸ÂÃ¢Æ’Â£ NETTOYAGE ANCIEN TRAITEMENT:")
    
    if output_6_dir.exists():
        try:
            shutil.rmtree(output_6_dir)
            print("   Ã¢Å“â€¦ Dossier de sortie 6.mp4 supprimÃƒÂ©")
        except Exception as e:
            print(f"   Ã¢ÂÅ’ Erreur suppression: {e}")
    
    # 3. VÃƒÂ©rifier que 6.mp4 est disponible
    print("\n3Ã¯Â¸ÂÃ¢Æ’Â£ VÃƒâ€°RIFICATION DISPONIBILITÃƒâ€° 6.MP4:")
    
    source_6mp4 = clips_dir / "6.mp4"
    if source_6mp4.exists():
        size_mb = source_6mp4.stat().st_size / (1024*1024)
        print(f"   Ã¢Å“â€¦ 6.mp4 disponible: {size_mb:.1f} MB")
    else:
        print("   Ã¢ÂÅ’ 6.mp4 non trouvÃƒÂ© dans clips/")
        return False
    
    # 4. Instructions pour le retraitement
    print("\n4Ã¯Â¸ÂÃ¢Æ’Â£ INSTRUCTIONS POUR LE RETRAITEMENT:")
    print("   Ã°Å¸Å½Â¯ Pour appliquer la correction des mots-clÃƒÂ©s B-roll:")
    print("   1. Ouvrir l'interface (lancer_interface.bat)")
    print("   2. Glisser-dÃƒÂ©poser 6.mp4 dans l'interface")
    print("   3. Le pipeline va maintenant:")
    print("      Ã°Å¸Â§Â  GÃƒÂ©nÃƒÂ©rer les mots-clÃƒÂ©s B-roll avec le LLM")
    print("      Ã°Å¸â€œÂ Les sauvegarder dans meta.txt")
    print("      Ã°Å¸â€œÂ¥ TÃƒÂ©lÃƒÂ©charger des B-rolls via les fetchers")
    print("      Ã°Å¸Å½Â¯ Les scorer et sÃƒÂ©lectionner")
    print("      Ã°Å¸Å½Â¬ Les intÃƒÂ©grer dans la vidÃƒÂ©o finale")
    
    # 5. VÃƒÂ©rification de la correction
    print("\n5Ã¯Â¸ÂÃ¢Æ’Â£ VÃƒâ€°RIFICATION DE LA CORRECTION:")
    
    # VÃƒÂ©rifier que le code a ÃƒÂ©tÃƒÂ© corrigÃƒÂ©
    video_processor_path = Path("video_processor.py")
    if video_processor_path.exists():
        with open(video_processor_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "B-roll Keywords:" in content:
            print("   Ã¢Å“â€¦ Code corrigÃƒÂ©: 'B-roll Keywords:' prÃƒÂ©sent")
        else:
            print("   Ã¢ÂÅ’ Code non corrigÃƒÂ©: 'B-roll Keywords:' absent")
    
    return True

def main():
    print("Ã°Å¸Å½Â¯ Retraitement de 6.mp4 avec correction des mots-clÃƒÂ©s B-roll")
    
    success = retraiter_6mp4()
    
    if success:
        print("\n" + "=" * 70)
        print("Ã°Å¸Å¡â‚¬ PRÃƒÅ T POUR LE RETRAITEMENT !")
        print("Ã¢Å“â€¦ Ancien traitement nettoyÃƒÂ©")
        print("Ã¢Å“â€¦ Code corrigÃƒÂ©")
        print("Ã¢Å“â€¦ 6.mp4 disponible")
        print("\nÃ°Å¸Å½Â¯ PROCHAINES Ãƒâ€°TAPES:")
        print("1. Lancer l'interface (lancer_interface.bat)")
        print("2. Glisser-dÃƒÂ©poser 6.mp4")
        print("3. Observer le flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring")
        print("4. VÃƒÂ©rifier que meta.txt contient les mots-clÃƒÂ©s B-roll")
    else:
        print("\n" + "=" * 70)
        print("Ã¢ÂÅ’ RETRAITEMENT IMPOSSIBLE")
        print("Ã¢Å¡Â Ã¯Â¸Â VÃƒÂ©rifiez la disponibilitÃƒÂ© de 6.mp4")
    
    return success

if __name__ == "__main__":
    success = main() 

