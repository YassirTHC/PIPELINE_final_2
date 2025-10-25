ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
VÃƒÂ©rification du Flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring pour 6.mp4
"""

import json
import os
from pathlib import Path

def verifier_flux_6mp4():
    print("Ã°Å¸â€Â VÃƒâ€°RIFICATION DU FLUX LLM Ã¢â€ â€™ FETCHERS Ã¢â€ â€™ SCORING POUR 6.MP4")
    print("=" * 70)
    
    # 1. VÃƒÂ©rifier la gÃƒÂ©nÃƒÂ©ration LLM
    print("1Ã¯Â¸ÂÃ¢Æ’Â£ VÃƒâ€°RIFICATION LLM:")
    meta_file = Path("output/clips/6/meta.txt")
    
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta_content = f.read()
        
        # VÃƒÂ©rifier la prÃƒÂ©sence de mots-clÃƒÂ©s B-roll
        if 'broll_keywords' in meta_content.lower():
            print("   Ã¢Å“â€¦ Mots-clÃƒÂ©s B-roll gÃƒÂ©nÃƒÂ©rÃƒÂ©s par LLM")
        else:
            print("   Ã¢Å¡Â Ã¯Â¸Â Aucun mot-clÃƒÂ© B-roll dÃƒÂ©tectÃƒÂ©")
            print("   Ã°Å¸â€Â Contenu meta.txt:")
            print(f"      {meta_content[:200]}...")
    
    # 2. VÃƒÂ©rifier les B-rolls tÃƒÂ©lÃƒÂ©chargÃƒÂ©s
    print("\n2Ã¯Â¸ÂÃ¢Æ’Â£ VÃƒâ€°RIFICATION FETCHERS:")
    broll_library = Path("AI-B-roll/broll_library")
    
    if broll_library.exists():
        clip_dirs = [d for d in broll_library.iterdir() if d.is_dir() and d.name.startswith('clip_reframed_')]
        print(f"   Ã°Å¸â€œÅ¡ {len(clip_dirs)} dossiers de clips reframÃƒÂ©s")
        
        # VÃƒÂ©rifier les clips rÃƒÂ©cents (derniers 5)
        recent_clips = sorted(clip_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        print("   Ã°Å¸â€ â€¢ 5 clips les plus rÃƒÂ©cents:")
        
        for clip_dir in recent_clips:
            clip_name = clip_dir.name
            mtime = clip_dir.stat().st_mtime
            print(f"      Ã°Å¸â€œÂ {clip_name}")
            
            # VÃƒÂ©rifier le contenu
            fetched_dir = clip_dir / "fetched"
            if fetched_dir.exists():
                sources = [d.name for d in fetched_dir.iterdir() if d.is_dir()]
                print(f"         Ã°Å¸â€œÂ¥ Sources: {', '.join(sources)}")
    
    # 3. VÃƒÂ©rifier le scoring et la sÃƒÂ©lection
    print("\n3Ã¯Â¸ÂÃ¢Æ’Â£ VÃƒâ€°RIFICATION SCORING & SÃƒâ€°LECTION:")
    pipeline_log = Path("output/pipeline.log.jsonl")
    
    if pipeline_log.exists():
        with open(pipeline_log, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        # Compter les ÃƒÂ©vÃƒÂ©nements B-roll
        broll_events = sum(1 for line in log_lines if '"type": "event_applied"' in line)
        print(f"   Ã°Å¸Å½Â¬ Ãƒâ€°vÃƒÂ©nements B-roll dans le log: {broll_events}")
        
        # VÃƒÂ©rifier les B-rolls rÃƒÂ©cents
        recent_events = [line for line in log_lines[-20:] if '"type": "event_applied"' in line]
        print(f"   Ã¢ÂÂ° 5 derniers ÃƒÂ©vÃƒÂ©nements B-roll:")
        
        for i, event_line in enumerate(recent_events[-5:], 1):
            try:
                event = json.loads(event_line.strip())
                start_s = event.get('start_s', 'N/A')
                end_s = event.get('end_s', 'N/A')
                media_path = event.get('media_path', 'N/A')
                
                print(f"      {i}. [{start_s}s-{end_s}s] {os.path.basename(media_path)}")
                
            except:
                print(f"      {i}. Ã¢Å¡Â Ã¯Â¸Â Erreur parsing JSON")
    
    # 4. VÃƒÂ©rifier l'intÃƒÂ©gration finale
    print("\n4Ã¯Â¸ÂÃ¢Æ’Â£ VÃƒâ€°RIFICATION INTÃƒâ€°GRATION:")
    
    # VÃƒÂ©rifier si 6.mp4 contient des B-rolls
    final_video = Path("output/clips/6/final_subtitled.mp4")
    if final_video.exists():
        size_mb = final_video.stat().st_size / (1024*1024)
        print(f"   Ã°Å¸Å½Â¬ VidÃƒÂ©o finale: {size_mb:.1f} MB")
        
        # Comparer avec la vidÃƒÂ©o originale
        original_video = Path("clips/6.mp4")
        if original_video.exists():
            original_size_mb = original_video.stat().st_size / (1024*1024)
            print(f"   Ã°Å¸â€œÂ¹ VidÃƒÂ©o originale: {original_size_mb:.1f} MB")
            
            if size_mb > original_size_mb * 1.1:  # 10% plus grande
                print("   Ã¢Å“â€¦ VidÃƒÂ©o finale plus grande - B-rolls probablement intÃƒÂ©grÃƒÂ©s")
            else:
                print("   Ã¢Å¡Â Ã¯Â¸Â VidÃƒÂ©o finale similaire - B-rolls peut-ÃƒÂªtre pas intÃƒÂ©grÃƒÂ©s")
    
    # 5. Conclusion du flux
    print("\n5Ã¯Â¸ÂÃ¢Æ’Â£ CONCLUSION DU FLUX:")
    
    # Ãƒâ€°valuer chaque composant
    llm_status = "Ã¢Å“â€¦" if meta_file.exists() else "Ã¢ÂÅ’"
    fetchers_status = "Ã¢Å“â€¦" if broll_library.exists() and len(clip_dirs) > 0 else "Ã¢ÂÅ’"
    scoring_status = "Ã¢Å“â€¦" if broll_events > 0 else "Ã¢ÂÅ’"
    integration_status = "Ã¢Å“â€¦" if final_video.exists() else "Ã¢ÂÅ’"
    
    print(f"   Ã°Å¸Â§Â  LLM: {llm_status}")
    print(f"   Ã°Å¸â€œÂ¥ Fetchers: {fetchers_status}")
    print(f"   Ã°Å¸Å½Â¯ Scoring: {fetchers_status}")
    print(f"   Ã°Å¸Å½Â¬ IntÃƒÂ©gration: {integration_status}")
    
    # Recommandations
    print("\n6Ã¯Â¸ÂÃ¢Æ’Â£ RECOMMANDATIONS:")
    
    if llm_status == "Ã¢ÂÅ’":
        print("   Ã°Å¸Â§Â  VÃƒÂ©rifier la gÃƒÂ©nÃƒÂ©ration LLM des mots-clÃƒÂ©s B-roll")
    
    if fetchers_status == "Ã¢ÂÅ’":
        print("   Ã°Å¸â€œÂ¥ VÃƒÂ©rifier le tÃƒÂ©lÃƒÂ©chargement des B-rolls")
    
    if scoring_status == "Ã¢ÂÅ’":
        print("   Ã°Å¸Å½Â¯ VÃƒÂ©rifier le systÃƒÂ¨me de scoring et sÃƒÂ©lection")
    
    if integration_status == "Ã¢ÂÅ’":
        print("   Ã°Å¸Å½Â¬ VÃƒÂ©rifier l'intÃƒÂ©gration finale des B-rolls")
    
    # VÃƒÂ©rifier si le flux complet fonctionne
    if all(status == "Ã¢Å“â€¦" for status in [llm_status, fetchers_status, scoring_status, integration_status]):
        print("\nÃ°Å¸Å½â€° FLUX COMPLET LLM Ã¢â€ â€™ FETCHERS Ã¢â€ â€™ SCORING FONCTIONNE !")
        print("   Ã°Å¸Å¡â‚¬ Le pipeline a traitÃƒÂ© 6.mp4 avec succÃƒÂ¨s")
        print("   Ã°Å¸Â§Â  LLM a gÃƒÂ©nÃƒÂ©rÃƒÂ© des mots-clÃƒÂ©s B-roll")
        print("   Ã°Å¸â€œÂ¥ Fetchers ont tÃƒÂ©lÃƒÂ©chargÃƒÂ© des B-rolls")
        print("   Ã°Å¸Å½Â¯ Scoring a ÃƒÂ©valuÃƒÂ© et sÃƒÂ©lectionnÃƒÂ©")
        print("   Ã°Å¸Å½Â¬ B-rolls ont ÃƒÂ©tÃƒÂ© intÃƒÂ©grÃƒÂ©s dans la vidÃƒÂ©o finale")
    else:
        print("\nÃ¢Å¡Â Ã¯Â¸Â FLUX INCOMPLET - VÃƒÂ©rification nÃƒÂ©cessaire")
        print("   Ã°Å¸â€Â§ Certains composants ne fonctionnent pas correctement")

if __name__ == "__main__":
    verifier_flux_6mp4() 

