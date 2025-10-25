ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Surveillance en Temps RÃƒÂ©el du Flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring
Observation du traitement de 6.mp4 avec la correction des mots-clÃƒÂ©s B-roll
"""

import time
import json
import os
from pathlib import Path
from datetime import datetime

def surveiller_flux_temps_reel():
    """Surveillance en temps rÃƒÂ©el du flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring"""
    print("Ã°Å¸Å¡â‚¬ SURVEILLANCE EN TEMPS RÃƒâ€°EL DU FLUX LLM Ã¢â€ â€™ FETCHERS Ã¢â€ â€™ SCORING")
    print("=" * 80)
    print("Ã°Å¸Å½Â¯ Observation du traitement de 6.mp4 avec correction des mots-clÃƒÂ©s B-roll")
    print("Ã¢ÂÂ° DÃƒÂ©marrage:", datetime.now().strftime("%H:%M:%S"))
    
    # Dossiers ÃƒÂ  surveiller
    clips_dir = Path("clips")
    output_dir = Path("output")
    output_6_dir = Path("output/clips/6")
    broll_library = Path("AI-B-roll/broll_library")
    
    # Compteurs de surveillance
    iteration = 0
    last_output_count = 0
    last_clips_count = 0
    last_broll_count = 0
    
    print("\nÃ°Å¸â€Â Dossiers surveillÃƒÂ©s:")
    print(f"   Ã°Å¸â€œÂ Clips: {clips_dir}")
    print(f"   Ã°Å¸â€œÂ Output: {output_dir}")
    print(f"   Ã°Å¸Å½Â¬ 6.mp4 Output: {output_6_dir}")
    print(f"   Ã°Å¸â€œÅ¡ B-roll Library: {broll_library}")
    
    try:
        while True:
            iteration += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # VÃƒÂ©rifier les changements
            output_files = list(output_dir.rglob("*")) if output_dir.exists() else []
            clips_files = list(clips_dir.rglob("*")) if clips_dir.exists() else []
            broll_files = list(broll_library.rglob("*")) if broll_library.exists() else []
            
            # Compter les fichiers
            output_count = len(output_files)
            clips_count = len(clips_files)
            broll_count = len(broll_files)
            
            # DÃƒÂ©tecter les changements
            output_changed = output_count != last_output_count
            clips_changed = clips_count != last_clips_count
            broll_changed = broll_count != last_broll_count
            
            print(f"\nÃ¢ÂÂ° [{current_time}] Surveillance #{iteration}")
            print("-" * 60)
            print(f"   Ã°Å¸â€œÂ Output: {output_count} fichiers")
            print(f"   Ã°Å¸â€œÂ Clips: {clips_count} fichiers")
            print(f"   Ã°Å¸â€œÅ¡ B-roll Library: {broll_count} fichiers")
            
            # Analyser les changements
            if output_changed or clips_changed or broll_changed:
                print(f"   Ã°Å¸â€â€ž Changements dÃƒÂ©tectÃƒÂ©s !")
                
                # 1. VÃƒÂ©rifier la crÃƒÂ©ation du dossier 6.mp4
                if output_6_dir.exists():
                    print(f"   Ã°Å¸Å½Â¬ Dossier 6.mp4 crÃƒÂ©ÃƒÂ©: {output_6_dir}")
                    
                    # VÃƒÂ©rifier les fichiers dans le dossier 6
                    files_in_6 = list(output_6_dir.rglob("*"))
                    print(f"      Ã°Å¸â€œÂ Fichiers dans 6/: {len(files_in_6)}")
                    
                    for file in files_in_6:
                        if file.is_file():
                            file_size = file.stat().st_size / (1024*1024)
                            print(f"      Ã°Å¸â€œâ€ž {file.name}: {file_size:.1f} MB")
                    
                    # VÃƒÂ©rifier meta.txt pour les mots-clÃƒÂ©s B-roll
                    meta_file = output_6_dir / "meta.txt"
                    if meta_file.exists():
                        try:
                            with open(meta_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            if "B-roll Keywords:" in content:
                                print(f"      Ã°Å¸Å½Â¯ MOTS-CLÃƒâ€°S B-ROLL DÃƒâ€°TECTÃƒâ€°S !")
                                
                                # Extraire les mots-clÃƒÂ©s
                                lines = content.split('\n')
                                for line in lines:
                                    if line.startswith("B-roll Keywords:"):
                                        keywords_part = line.replace("B-roll Keywords:", "").strip()
                                        keywords = [kw.strip() for kw in keywords_part.split(',') if kw.strip()]
                                        print(f"         Ã°Å¸Å½Â¬ {len(keywords)} mots-clÃƒÂ©s: {', '.join(keywords[:5])}...")
                                        break
                            else:
                                print(f"      Ã¢Å¡Â Ã¯Â¸Â Mots-clÃƒÂ©s B-roll non trouvÃƒÂ©s dans meta.txt")
                                
                        except Exception as e:
                            print(f"      Ã¢ÂÅ’ Erreur lecture meta.txt: {e}")
                
                # 2. VÃƒÂ©rifier la bibliothÃƒÂ¨que B-roll
                if broll_library.exists():
                    clip_dirs = [d for d in broll_library.iterdir() if d.is_dir() and d.name.startswith('clip_reframed_')]
                    if len(clip_dirs) > last_broll_count:
                        print(f"   Ã°Å¸â€œÅ¡ Nouveaux clips B-roll dÃƒÂ©tectÃƒÂ©s: {len(clip_dirs)}")
                        
                        # VÃƒÂ©rifier les nouveaux clips
                        new_clips = clip_dirs[-3:] if len(clip_dirs) > 3 else clip_dirs
                        for clip_dir in new_clips:
                            clip_name = clip_dir.name
                            print(f"      Ã°Å¸â€œÂ {clip_name}")
                            
                            # VÃƒÂ©rifier le contenu
                            fetched_dir = clip_dir / "fetched"
                            if fetched_dir.exists():
                                sources = [d.name for d in fetched_dir.iterdir() if d.is_dir()]
                                print(f"         Ã°Å¸â€œÂ¥ Sources: {', '.join(sources)}")
                
                # 3. VÃƒÂ©rifier les logs du pipeline
                log_file = Path("output/pipeline.log.jsonl")
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        # Analyser les nouvelles lignes
                        if len(lines) > last_output_count:
                            new_lines = lines[-10:] if len(lines) > 10 else lines
                            print(f"   Ã°Å¸â€œâ€¹ Nouvelles lignes du log ({len(new_lines)} lignes):")
                            
                            for line in new_lines[-3:]:  # Afficher les 3 derniÃƒÂ¨res
                                try:
                                    log_entry = json.loads(line.strip())
                                    event_type = log_entry.get('type', 'N/A')
                                    if event_type == 'event_applied':
                                        start_s = log_entry.get('start_s', 'N/A')
                                        end_s = log_entry.get('end_s', 'N/A')
                                        media_path = log_entry.get('media_path', 'N/A')
                                        print(f"      Ã°Å¸Å½Â¬ B-roll appliquÃƒÂ©: [{start_s}s-{end_s}s] {os.path.basename(media_path)}")
                                    else:
                                        print(f"      Ã°Å¸â€œÂ {event_type}: {line[:80]}...")
                                except:
                                    print(f"      Ã¢Å¡Â Ã¯Â¸Â Ligne non-JSON: {line[:80]}...")
                    
                    except Exception as e:
                        print(f"      Ã¢Å¡Â Ã¯Â¸Â Erreur lecture log: {e}")
                
                # Mettre ÃƒÂ  jour les compteurs
                last_output_count = output_count
                last_clips_count = clips_count
                last_broll_count = broll_count
            
            # Attendre avant la prochaine vÃƒÂ©rification
            print(f"   Ã¢ÂÂ³ Attente 10 secondes... (Ctrl+C pour arrÃƒÂªter)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print(f"\nÃ°Å¸â€ºâ€˜ Surveillance arrÃƒÂªtÃƒÂ©e par l'utilisateur")
        print(f"Ã¢ÂÂ° DurÃƒÂ©e totale: {iteration * 10} secondes")
        print(f"Ã°Å¸Å½Â¯ VÃƒÂ©rifiez l'interface pour voir le traitement en cours")
    except Exception as e:
        print(f"\nÃ¢ÂÅ’ Erreur lors de la surveillance: {e}")

def main():
    """Fonction principale de surveillance"""
    print("Ã°Å¸Å½Â¯ Surveillance en temps rÃƒÂ©el du flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring")
    print("Ã°Å¸â€Â Observation du traitement de 6.mp4 avec correction des mots-clÃƒÂ©s B-roll")
    
    surveiller_flux_temps_reel()

if __name__ == "__main__":
    main() 

