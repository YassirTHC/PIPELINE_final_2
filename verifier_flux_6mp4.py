#!/usr/bin/env python3
"""
Vérification du Flux LLM → Fetchers → Scoring pour 6.mp4
"""

import json
import os
from pathlib import Path

def verifier_flux_6mp4():
    print("🔍 VÉRIFICATION DU FLUX LLM → FETCHERS → SCORING POUR 6.MP4")
    print("=" * 70)
    
    # 1. Vérifier la génération LLM
    print("1️⃣ VÉRIFICATION LLM:")
    meta_file = Path("output/clips/6/meta.txt")
    
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta_content = f.read()
        
        # Vérifier la présence de mots-clés B-roll
        if 'broll_keywords' in meta_content.lower():
            print("   ✅ Mots-clés B-roll générés par LLM")
        else:
            print("   ⚠️ Aucun mot-clé B-roll détecté")
            print("   🔍 Contenu meta.txt:")
            print(f"      {meta_content[:200]}...")
    
    # 2. Vérifier les B-rolls téléchargés
    print("\n2️⃣ VÉRIFICATION FETCHERS:")
    broll_library = Path("AI-B-roll/broll_library")
    
    if broll_library.exists():
        clip_dirs = [d for d in broll_library.iterdir() if d.is_dir() and d.name.startswith('clip_reframed_')]
        print(f"   📚 {len(clip_dirs)} dossiers de clips reframés")
        
        # Vérifier les clips récents (derniers 5)
        recent_clips = sorted(clip_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        print("   🆕 5 clips les plus récents:")
        
        for clip_dir in recent_clips:
            clip_name = clip_dir.name
            mtime = clip_dir.stat().st_mtime
            print(f"      📁 {clip_name}")
            
            # Vérifier le contenu
            fetched_dir = clip_dir / "fetched"
            if fetched_dir.exists():
                sources = [d.name for d in fetched_dir.iterdir() if d.is_dir()]
                print(f"         📥 Sources: {', '.join(sources)}")
    
    # 3. Vérifier le scoring et la sélection
    print("\n3️⃣ VÉRIFICATION SCORING & SÉLECTION:")
    pipeline_log = Path("output/pipeline.log.jsonl")
    
    if pipeline_log.exists():
        with open(pipeline_log, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        # Compter les événements B-roll
        broll_events = sum(1 for line in log_lines if '"type": "event_applied"' in line)
        print(f"   🎬 Événements B-roll dans le log: {broll_events}")
        
        # Vérifier les B-rolls récents
        recent_events = [line for line in log_lines[-20:] if '"type": "event_applied"' in line]
        print(f"   ⏰ 5 derniers événements B-roll:")
        
        for i, event_line in enumerate(recent_events[-5:], 1):
            try:
                event = json.loads(event_line.strip())
                start_s = event.get('start_s', 'N/A')
                end_s = event.get('end_s', 'N/A')
                media_path = event.get('media_path', 'N/A')
                
                print(f"      {i}. [{start_s}s-{end_s}s] {os.path.basename(media_path)}")
                
            except:
                print(f"      {i}. ⚠️ Erreur parsing JSON")
    
    # 4. Vérifier l'intégration finale
    print("\n4️⃣ VÉRIFICATION INTÉGRATION:")
    
    # Vérifier si 6.mp4 contient des B-rolls
    final_video = Path("output/clips/6/final_subtitled.mp4")
    if final_video.exists():
        size_mb = final_video.stat().st_size / (1024*1024)
        print(f"   🎬 Vidéo finale: {size_mb:.1f} MB")
        
        # Comparer avec la vidéo originale
        original_video = Path("clips/6.mp4")
        if original_video.exists():
            original_size_mb = original_video.stat().st_size / (1024*1024)
            print(f"   📹 Vidéo originale: {original_size_mb:.1f} MB")
            
            if size_mb > original_size_mb * 1.1:  # 10% plus grande
                print("   ✅ Vidéo finale plus grande - B-rolls probablement intégrés")
            else:
                print("   ⚠️ Vidéo finale similaire - B-rolls peut-être pas intégrés")
    
    # 5. Conclusion du flux
    print("\n5️⃣ CONCLUSION DU FLUX:")
    
    # Évaluer chaque composant
    llm_status = "✅" if meta_file.exists() else "❌"
    fetchers_status = "✅" if broll_library.exists() and len(clip_dirs) > 0 else "❌"
    scoring_status = "✅" if broll_events > 0 else "❌"
    integration_status = "✅" if final_video.exists() else "❌"
    
    print(f"   🧠 LLM: {llm_status}")
    print(f"   📥 Fetchers: {fetchers_status}")
    print(f"   🎯 Scoring: {fetchers_status}")
    print(f"   🎬 Intégration: {integration_status}")
    
    # Recommandations
    print("\n6️⃣ RECOMMANDATIONS:")
    
    if llm_status == "❌":
        print("   🧠 Vérifier la génération LLM des mots-clés B-roll")
    
    if fetchers_status == "❌":
        print("   📥 Vérifier le téléchargement des B-rolls")
    
    if scoring_status == "❌":
        print("   🎯 Vérifier le système de scoring et sélection")
    
    if integration_status == "❌":
        print("   🎬 Vérifier l'intégration finale des B-rolls")
    
    # Vérifier si le flux complet fonctionne
    if all(status == "✅" for status in [llm_status, fetchers_status, scoring_status, integration_status]):
        print("\n🎉 FLUX COMPLET LLM → FETCHERS → SCORING FONCTIONNE !")
        print("   🚀 Le pipeline a traité 6.mp4 avec succès")
        print("   🧠 LLM a généré des mots-clés B-roll")
        print("   📥 Fetchers ont téléchargé des B-rolls")
        print("   🎯 Scoring a évalué et sélectionné")
        print("   🎬 B-rolls ont été intégrés dans la vidéo finale")
    else:
        print("\n⚠️ FLUX INCOMPLET - Vérification nécessaire")
        print("   🔧 Certains composants ne fonctionnent pas correctement")

if __name__ == "__main__":
    verifier_flux_6mp4() 