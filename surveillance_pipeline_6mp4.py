ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Surveillance en Temps RÃƒÂ©el du Pipeline 6.mp4
Observation du flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring
"""

import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime

def surveiller_pipeline_6mp4():
    """Surveillance en temps rÃƒÂ©el du pipeline avec 6.mp4"""
    print("Ã°Å¸Å¡â‚¬ SURVEILLANCE EN TEMPS RÃƒâ€°EL DU PIPELINE 6.MP4")
    print("=" * 80)
    print("Ã°Å¸Å½Â¯ Observation du flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring")
    print("Ã¢ÂÂ° DÃƒÂ©marrage:", datetime.now().strftime("%H:%M:%S"))
    
    # Dossiers ÃƒÂ  surveiller
    output_dir = Path("output")
    clips_dir = Path("output/clips")
    meta_dir = Path("output/meta")
    
    # Compteurs de surveillance
    iteration = 0
    last_output_count = 0
    last_clips_count = 0
    
    print("\nÃ°Å¸â€Â Dossiers surveillÃƒÂ©s:")
    print(f"   Ã°Å¸â€œÂ Output: {output_dir}")
    print(f"   Ã°Å¸Å½Â¬ Clips: {clips_dir}")
    print(f"   Ã°Å¸â€œÅ  Meta: {meta_dir}")
    
    try:
        while True:
            iteration += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # VÃƒÂ©rifier les changements
            output_files = list(output_dir.rglob("*")) if output_dir.exists() else []
            clips_files = list(clips_dir.rglob("*")) if clips_dir.exists() else []
            meta_files = list(meta_dir.rglob("*")) if meta_dir.exists() else []
            
            # Compter les fichiers
            output_count = len(output_files)
            clips_count = len(clips_files)
            meta_count = len(meta_files)
            
            # DÃƒÂ©tecter les changements
            output_changed = output_count != last_output_count
            clips_changed = clips_count != last_clips_count
            
            print(f"\nÃ¢ÂÂ° [{current_time}] Surveillance #{iteration}")
            print("-" * 60)
            print(f"   Ã°Å¸â€œÂ Output: {output_count} fichiers")
            print(f"   Ã°Å¸Å½Â¬ Clips: {clips_count} fichiers")
            print(f"   Ã°Å¸â€œÅ  Meta: {meta_count} fichiers")
            
            # Analyser les changements
            if output_changed or clips_changed:
                print(f"   Ã°Å¸â€â€ž Changements dÃƒÂ©tectÃƒÂ©s !")
                
                # VÃƒÂ©rifier les nouveaux fichiers de clips
                if clips_dir.exists():
                    clip_dirs = [d for d in clips_dir.iterdir() if d.is_dir()]
                    for clip_dir in clip_dirs:
                        if clip_dir.name == "6":
                            print(f"   Ã°Å¸Å½Â¬ Dossier 6.mp4 trouvÃƒÂ©: {clip_dir}")
                            
                            # VÃƒÂ©rifier les fichiers dans le dossier 6
                            files_in_6 = list(clip_dir.rglob("*"))
                            print(f"      Ã°Å¸â€œÂ Fichiers dans 6/: {len(files_in_6)}")
                            
                            for file in files_in_6:
                                if file.is_file():
                                    file_size = file.stat().st_size / (1024*1024)
                                    print(f"      Ã°Å¸â€œâ€ž {file.name}: {file_size:.1f} MB")
                
                # VÃƒÂ©rifier les nouveaux fichiers meta
                if meta_dir.exists():
                    meta_files_6 = [f for f in meta_dir.iterdir() if "6" in f.name]
                    for meta_file in meta_files_6:
                        if meta_file.is_file():
                            print(f"   Ã°Å¸â€œÅ  Fichier meta trouvÃƒÂ©: {meta_file.name}")
                            
                            # Essayer de lire le contenu JSON
                            try:
                                with open(meta_file, 'r', encoding='utf-8') as f:
                                    content = json.load(f)
                                print(f"      Ã°Å¸â€œÂ Contenu: {len(content)} clÃƒÂ©s")
                                
                                # Afficher les clÃƒÂ©s importantes
                                if 'broll_keywords' in content:
                                    keywords = content['broll_keywords']
                                    print(f"      Ã°Å¸Å½Â¯ Mots-clÃƒÂ©s B-roll LLM: {len(keywords)} termes")
                                    print(f"         Exemples: {', '.join(keywords[:5])}")
                                
                                if 'selection_report' in content:
                                    report = content['selection_report']
                                    print(f"      Ã°Å¸â€œÅ  Rapport de sÃƒÂ©lection: {len(report)} clÃƒÂ©s")
                                
                            except Exception as e:
                                print(f"      Ã¢Å¡Â Ã¯Â¸Â Erreur lecture JSON: {e}")
                
                # VÃƒÂ©rifier les logs du pipeline
                log_file = Path("output/pipeline.log.jsonl")
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        # Analyser les derniÃƒÂ¨res lignes
                        recent_lines = lines[-10:] if len(lines) > 10 else lines
                        print(f"   Ã°Å¸â€œâ€¹ DerniÃƒÂ¨res lignes du log ({len(recent_lines)} lignes):")
                        
                        for line in recent_lines[-3:]:  # Afficher les 3 derniÃƒÂ¨res
                            try:
                                log_entry = json.loads(line.strip())
                                timestamp = log_entry.get('timestamp', 'N/A')
                                level = log_entry.get('level', 'INFO')
                                message = log_entry.get('message', 'N/A')
                                print(f"      [{timestamp}] {level}: {message[:80]}...")
                            except:
                                print(f"      Ã¢Å¡Â Ã¯Â¸Â Ligne non-JSON: {line[:80]}...")
                    
                    except Exception as e:
                        print(f"      Ã¢Å¡Â Ã¯Â¸Â Erreur lecture log: {e}")
                
                # Mettre ÃƒÂ  jour les compteurs
                last_output_count = output_count
                last_clips_count = clips_count
            
            # Attendre avant la prochaine vÃƒÂ©rification
            print(f"   Ã¢ÂÂ³ Attente 15 secondes... (Ctrl+C pour arrÃƒÂªter)")
            time.sleep(15)
            
    except KeyboardInterrupt:
        print(f"\nÃ°Å¸â€ºâ€˜ Surveillance arrÃƒÂªtÃƒÂ©e par l'utilisateur")
        print(f"Ã¢ÂÂ° DurÃƒÂ©e totale: {iteration * 15} secondes")
        print(f"Ã°Å¸Å½Â¯ VÃƒÂ©rifiez l'interface pour voir le traitement en cours")
    except Exception as e:
        print(f"\nÃ¢ÂÅ’ Erreur lors de la surveillance: {e}")

def main():
    """Fonction principale de surveillance"""
    print("Ã°Å¸Å½Â¯ Surveillance en temps rÃƒÂ©el du pipeline 6.mp4")
    print("Ã°Å¸â€Â Observation du flux LLM Ã¢â€ â€™ Fetchers Ã¢â€ â€™ Scoring")
    
    surveiller_pipeline_6mp4()

if __name__ == "__main__":
    main() 

