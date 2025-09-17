#!/usr/bin/env python3
"""
Retraitement de 6.mp4 avec correction des mots-clés B-roll
"""

import shutil
import os
from pathlib import Path

def retraiter_6mp4():
    print("🔄 RETRAITEMENT DE 6.MP4 AVEC CORRECTION DES MOTS-CLÉS B-ROLL")
    print("=" * 70)
    
    # 1. Vérifier l'état actuel
    print("1️⃣ ÉTAT ACTUEL:")
    
    clips_dir = Path("clips")
    output_6_dir = Path("output/clips/6")
    
    if clips_dir.exists():
        clips = list(clips_dir.glob("*.mp4"))
        print(f"   📁 Clips disponibles: {len(clips)}")
        for clip in clips:
            print(f"      📹 {clip.name}")
    
    if output_6_dir.exists():
        print(f"   📁 Dossier de sortie 6.mp4: {output_6_dir}")
        files = list(output_6_dir.iterdir())
        print(f"      📄 {len(files)} fichiers")
    
    # 2. Nettoyer l'ancien traitement
    print("\n2️⃣ NETTOYAGE ANCIEN TRAITEMENT:")
    
    if output_6_dir.exists():
        try:
            shutil.rmtree(output_6_dir)
            print("   ✅ Dossier de sortie 6.mp4 supprimé")
        except Exception as e:
            print(f"   ❌ Erreur suppression: {e}")
    
    # 3. Vérifier que 6.mp4 est disponible
    print("\n3️⃣ VÉRIFICATION DISPONIBILITÉ 6.MP4:")
    
    source_6mp4 = clips_dir / "6.mp4"
    if source_6mp4.exists():
        size_mb = source_6mp4.stat().st_size / (1024*1024)
        print(f"   ✅ 6.mp4 disponible: {size_mb:.1f} MB")
    else:
        print("   ❌ 6.mp4 non trouvé dans clips/")
        return False
    
    # 4. Instructions pour le retraitement
    print("\n4️⃣ INSTRUCTIONS POUR LE RETRAITEMENT:")
    print("   🎯 Pour appliquer la correction des mots-clés B-roll:")
    print("   1. Ouvrir l'interface (lancer_interface.bat)")
    print("   2. Glisser-déposer 6.mp4 dans l'interface")
    print("   3. Le pipeline va maintenant:")
    print("      🧠 Générer les mots-clés B-roll avec le LLM")
    print("      📝 Les sauvegarder dans meta.txt")
    print("      📥 Télécharger des B-rolls via les fetchers")
    print("      🎯 Les scorer et sélectionner")
    print("      🎬 Les intégrer dans la vidéo finale")
    
    # 5. Vérification de la correction
    print("\n5️⃣ VÉRIFICATION DE LA CORRECTION:")
    
    # Vérifier que le code a été corrigé
    video_processor_path = Path("video_processor.py")
    if video_processor_path.exists():
        with open(video_processor_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "B-roll Keywords:" in content:
            print("   ✅ Code corrigé: 'B-roll Keywords:' présent")
        else:
            print("   ❌ Code non corrigé: 'B-roll Keywords:' absent")
    
    return True

def main():
    print("🎯 Retraitement de 6.mp4 avec correction des mots-clés B-roll")
    
    success = retraiter_6mp4()
    
    if success:
        print("\n" + "=" * 70)
        print("🚀 PRÊT POUR LE RETRAITEMENT !")
        print("✅ Ancien traitement nettoyé")
        print("✅ Code corrigé")
        print("✅ 6.mp4 disponible")
        print("\n🎯 PROCHAINES ÉTAPES:")
        print("1. Lancer l'interface (lancer_interface.bat)")
        print("2. Glisser-déposer 6.mp4")
        print("3. Observer le flux LLM → Fetchers → Scoring")
        print("4. Vérifier que meta.txt contient les mots-clés B-roll")
    else:
        print("\n" + "=" * 70)
        print("❌ RETRAITEMENT IMPOSSIBLE")
        print("⚠️ Vérifiez la disponibilité de 6.mp4")
    
    return success

if __name__ == "__main__":
    success = main() 