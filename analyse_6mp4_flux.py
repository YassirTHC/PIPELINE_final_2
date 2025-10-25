ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Analyse du Flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring pour 6.mp4
VÃƒÂ©rification de ce qui s'est passÃƒÂ© lors du traitement
"""

import json
import os
from pathlib import Path

def analyser_flux_6mp4():
    """Analyse du flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring pour 6.mp4"""
    print("Ã°Å¸â€Â ANALYSE DU FLUX LLM Ã¢â€ â€™ FETCHERS Ã¢â€ â€™ SCORING POUR 6.MP4")
    print("=" * 80)
    
    # Dossier de sortie de 6.mp4
    output_6_dir = Path("output/clips/6")
    
    if not output_6_dir.exists():
        print("Ã¢ÂÅ’ Dossier output/clips/6 non trouvÃƒÂ©")
        return
    
    print(f"Ã°Å¸â€œÂ Dossier analysÃƒÂ©: {output_6_dir}")
    
    # 1. VÃƒÂ©rifier les fichiers gÃƒÂ©nÃƒÂ©rÃƒÂ©s
    print("\nÃ°Å¸â€Â 1. FICHIERS GÃƒâ€°NÃƒâ€°RÃƒâ€°S:")
    files = list(output_6_dir.iterdir())
    for file in files:
        if file.is_file():
            size_mb = file.stat().st_size / (1024*1024)
            print(f"   Ã°Å¸â€œâ€ž {file.name}: {size_mb:.1f} MB")
    
    # 2. Analyser les mÃƒÂ©tadonnÃƒÂ©es
    print("\nÃ°Å¸â€Â 2. MÃƒâ€°TADONNÃƒâ€°ES:")
    meta_file = output_6_dir / "meta.txt"
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"   Ã°Å¸â€œÂ Contenu meta.txt:")
        for line in content.split('\n'):
            if line.strip():
                print(f"      {line}")
    else:
        print("   Ã¢ÂÅ’ meta.txt non trouvÃƒÂ©")
    
    # 3. Analyser la transcription
    print("\nÃ°Å¸â€Â 3. TRANSCRIPTION:")
    segments_file = output_6_dir / "6_segments.json"
    if segments_file.exists():
        with open(segments_file, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        print(f"   Ã°Å¸â€œÅ  {len(segments)} segments de transcription")
        
        # Extraire les mots-clÃƒÂ©s potentiels
        all_text = " ".join([seg.get('text', '') for seg in segments])
        words = [word.lower().strip() for word in all_text.split() if len(word) > 3]
        
        # Mots-clÃƒÂ©s liÃƒÂ©s ÃƒÂ  la santÃƒÂ©
        health_keywords = ['healthcare', 'medical', 'doctor', 'hospital', 'operation', 'medicare', 'medicaid']
        found_health = [word for word in words if any(health in word for health in health_keywords)]
        
        print(f"   Ã°Å¸ÂÂ¥ Mots-clÃƒÂ©s santÃƒÂ© trouvÃƒÂ©s: {len(found_health)}")
        if found_health:
            print(f"      Exemples: {', '.join(set(found_health[:10]))}")
    
    # 4. Analyser les tokens avec couleurs
    print("\nÃ°Å¸â€Â 4. TOKENS AVEC COULEURS:")
    tokens_file = output_6_dir / "final_subtitled.tokens.json"
    if tokens_file.exists():
        with open(tokens_file, 'r', encoding='utf-8') as f:
            tokens_data = json.load(f)
        
        print(f"   Ã°Å¸Å½Â¨ {len(tokens_data)} segments avec tokens")
        
        # Compter les mots-clÃƒÂ©s colorÃƒÂ©s
        colored_keywords = []
        for segment in tokens_data:
            for token in segment.get('tokens', []):
                if token.get('is_keyword', False):
                    colored_keywords.append(token.get('text', ''))
        
        print(f"   Ã°Å¸Å½Â¯ Mots-clÃƒÂ©s colorÃƒÂ©s: {len(colored_keywords)}")
        if colored_keywords:
            unique_colored = list(set(colored_keywords))
            print(f"      Exemples: {', '.join(unique_colored[:15])}")
    
    # 5. VÃƒÂ©rifier le log du pipeline
    print("\nÃ°Å¸â€Â 5. LOG DU PIPELINE:")
    pipeline_log = Path("output/pipeline.log.jsonl")
    if pipeline_log.exists():
        with open(pipeline_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"   Ã°Å¸â€œâ€¹ {len(lines)} lignes dans le log")
        
        # Chercher des informations sur 6.mp4
        lines_6mp4 = [line for line in lines if "6" in line]
        print(f"   Ã°Å¸Å½Â¬ Lignes contenant '6': {len(lines_6mp4)}")
        
        # Analyser les derniers ÃƒÂ©vÃƒÂ©nements
        recent_events = lines[-20:] if len(lines) > 20 else lines
        print(f"   Ã¢ÂÂ° 20 derniers ÃƒÂ©vÃƒÂ©nements:")
        
        for i, line in enumerate(recent_events[-5:], 1):
            try:
                event = json.loads(line.strip())
                event_type = event.get('type', 'N/A')
                media_path = event.get('media_path', 'N/A')
                start_s = event.get('start_s', 'N/A')
                end_s = event.get('end_s', 'N/A')
                
                print(f"      {i}. [{start_s}s-{end_s}s] {event_type}")
                if '6' in media_path:
                    print(f"         Ã°Å¸Å½Â¬ 6.mp4: {os.path.basename(media_path)}")
                else:
                    print(f"         Ã°Å¸â€œÂ¹ B-roll: {os.path.basename(media_path)}")
                    
            except:
                print(f"      {i}. Ã¢Å¡Â Ã¯Â¸Â Ligne non-JSON")
    
    # 6. VÃƒÂ©rifier la bibliothÃƒÂ¨que B-roll
    print("\nÃ°Å¸â€Â 6. BIBLIOTHÃƒË†QUE B-ROLL:")
    broll_library = Path("AI-B-roll/broll_library")
    if broll_library.exists():
        clip_dirs = [d for d in broll_library.iterdir() if d.is_dir() and d.name.startswith('clip_reframed_')]
        print(f"   Ã°Å¸â€œÅ¡ {len(clip_dirs)} dossiers de clips reframÃƒÂ©s")
        
        # VÃƒÂ©rifier les clips rÃƒÂ©cents
        recent_clips = sorted(clip_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        print(f"   Ã°Å¸â€ â€¢ 5 clips les plus rÃƒÂ©cents:")
        
        for clip_dir in recent_clips:
            clip_name = clip_dir.name
            mtime = clip_dir.stat().st_mtime
            print(f"      Ã°Å¸â€œÂ {clip_name}")
            
            # VÃƒÂ©rifier le contenu
            fetched_dir = clip_dir / "fetched"
            if fetched_dir.exists():
                sources = [d.name for d in fetched_dir.iterdir() if d.is_dir()]
                print(f"         Ã°Å¸â€œÂ¥ Sources: {', '.join(sources)}")
    
    # 7. Analyse du flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring
    print("\nÃ°Å¸â€Â 7. ANALYSE DU FLUX LLM Ã¢â€ â€™ FETCHERS Ã¢â€ â€™ SCORING:")
    
    # VÃƒÂ©rifier si les mots-clÃƒÂ©s B-roll ont ÃƒÂ©tÃƒÂ© gÃƒÂ©nÃƒÂ©rÃƒÂ©s
    if meta_file.exists():
        meta_content = open(meta_file, 'r', encoding='utf-8').read()
        
        # VÃƒÂ©rifier la prÃƒÂ©sence de mots-clÃƒÂ©s B-roll
        if 'broll_keywords' in meta_content.lower() or 'keywords' in meta_content.lower():
            print("   Ã¢Å“â€¦ Mots-clÃƒÂ©s B-roll dÃƒÂ©tectÃƒÂ©s dans les mÃƒÂ©tadonnÃƒÂ©es")
        else:
            print("   Ã¢Å¡Â Ã¯Â¸Â Aucun mot-clÃƒÂ© B-roll dÃƒÂ©tectÃƒÂ© dans les mÃƒÂ©tadonnÃƒÂ©es")
            print("   Ã°Å¸â€Â VÃƒÂ©rification du fichier tokens.json...")
            
            # VÃƒÂ©rifier dans tokens.json
            if tokens_file.exists():
                with open(tokens_file, 'r', encoding='utf-8') as f:
                    tokens_content = f.read()
                
                if 'broll_keywords' in tokens_content.lower():
                    print("      Ã¢Å“â€¦ Mots-clÃƒÂ©s B-roll trouvÃƒÂ©s dans tokens.json")
                else:
                    print("      Ã¢ÂÅ’ Aucun mot-clÃƒÂ© B-roll trouvÃƒÂ©")
    
    # VÃƒÂ©rifier l'utilisation des B-rolls
    if pipeline_log.exists():
        with open(pipeline_log, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Compter les B-rolls utilisÃƒÂ©s
        broll_events = log_content.count('"type": "event_applied"')
        print(f"   Ã°Å¸Å½Â¬ Ãƒâ€°vÃƒÂ©nements B-roll dans le log: {broll_events}")
        
        if broll_events > 0:
            print("   Ã¢Å“â€¦ B-rolls ont ÃƒÂ©tÃƒÂ© appliquÃƒÂ©s")
        else:
            print("   Ã¢ÂÅ’ Aucun B-roll appliquÃƒÂ©")
    
    # 8. Conclusion
    print("\nÃ°Å¸â€Â 8. CONCLUSION:")
    
    # VÃƒÂ©rifier les composants du flux
    components_status = {
        "LLM": "Ã¢Ââ€œ Ãƒâ‚¬ vÃƒÂ©rifier",
        "Fetchers": "Ã¢Ââ€œ Ãƒâ‚¬ vÃƒÂ©rifier", 
        "Scoring": "Ã¢Ââ€œ Ãƒâ‚¬ vÃƒÂ©rifier",
        "SÃƒÂ©lection": "Ã¢Ââ€œ Ãƒâ‚¬ vÃƒÂ©rifier"
    }
    
    # Mettre ÃƒÂ  jour le statut basÃƒÂ© sur l'analyse
    if meta_file.exists() and "healthcare" in open(meta_file, 'r', encoding='utf-8').read().lower():
        components_status["LLM"] = "Ã¢Å“â€¦ Actif (mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©rÃƒÂ©s)"
    
    if broll_library.exists() and len(list(broll_library.iterdir())) > 0:
        components_status["Fetchers"] = "Ã¢Å“â€¦ Actif (bibliothÃƒÂ¨que B-roll)"
    
    if pipeline_log.exists() and broll_events > 0:
        components_status["Scoring"] = "Ã¢Å“â€¦ Actif (B-rolls appliquÃƒÂ©s)"
        components_status["SÃƒÂ©lection"] = "Ã¢Å“â€¦ Actif (B-rolls sÃƒÂ©lectionnÃƒÂ©s)"
    
    for component, status in components_status.items():
        print(f"   {component}: {status}")
    
    # Recommandations
    print("\nÃ°Å¸â€Â 9. RECOMMANDATIONS:")
    
    if components_status["LLM"] == "Ã¢Ââ€œ Ãƒâ‚¬ vÃƒÂ©rifier":
        print("   Ã°Å¸Â§Â  VÃƒÂ©rifier la gÃƒÂ©nÃƒÂ©ration LLM des mots-clÃƒÂ©s B-roll")
    
    if components_status["Fetchers"] == "Ã¢Ââ€œ Ãƒâ‚¬ vÃƒÂ©rifier":
        print("   Ã°Å¸â€œÂ¥ VÃƒÂ©rifier le tÃƒÂ©lÃƒÂ©chargement des B-rolls")
    
    if components_status["Scoring"] == "Ã¢Ââ€œ Ãƒâ‚¬ vÃƒÂ©rifier":
        print("   Ã°Å¸Å½Â¯ VÃƒÂ©rifier le systÃƒÂ¨me de scoring")
    
    if components_status["SÃƒÂ©lection"] == "Ã¢Ââ€œ Ãƒâ‚¬ vÃƒÂ©rifier":
        print("   Ã°Å¸Å½Â¬ VÃƒÂ©rifier la sÃƒÂ©lection finale des B-rolls")
    
    if all("Ã¢Å“â€¦" in status for status in components_status.values()):
        print("   Ã°Å¸Å½â€° Tous les composants du flux sont actifs !")
        print("   Ã°Å¸Å¡â‚¬ Le pipeline LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring fonctionne parfaitement")

def main():
    """Fonction principale"""
    print("Ã°Å¸Å½Â¯ Analyse du flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring pour 6.mp4")
    
    analyser_flux_6mp4()

if __name__ == "__main__":
    main() 

