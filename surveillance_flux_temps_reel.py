#!/usr/bin/env python3
"""
Surveillance en Temps Réel du Flux LLM → Fetchers → Scoring
Observation du traitement de 6.mp4 avec la correction des mots-clés B-roll
"""

import time
import json
import os
from pathlib import Path
from datetime import datetime

def surveiller_flux_temps_reel():
    """Surveillance en temps réel du flux LLM → Fetchers → Scoring"""
    print("🚀 SURVEILLANCE EN TEMPS RÉEL DU FLUX LLM → FETCHERS → SCORING")
    print("=" * 80)
    print("🎯 Observation du traitement de 6.mp4 avec correction des mots-clés B-roll")
    print("⏰ Démarrage:", datetime.now().strftime("%H:%M:%S"))
    
    # Dossiers à surveiller
    clips_dir = Path("clips")
    output_dir = Path("output")
    output_6_dir = Path("output/clips/6")
    broll_library = Path("AI-B-roll/broll_library")
    
    # Compteurs de surveillance
    iteration = 0
    last_output_count = 0
    last_clips_count = 0
    last_broll_count = 0
    
    print("\n🔍 Dossiers surveillés:")
    print(f"   📁 Clips: {clips_dir}")
    print(f"   📁 Output: {output_dir}")
    print(f"   🎬 6.mp4 Output: {output_6_dir}")
    print(f"   📚 B-roll Library: {broll_library}")
    
    try:
        while True:
            iteration += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Vérifier les changements
            output_files = list(output_dir.rglob("*")) if output_dir.exists() else []
            clips_files = list(clips_dir.rglob("*")) if clips_dir.exists() else []
            broll_files = list(broll_library.rglob("*")) if broll_library.exists() else []
            
            # Compter les fichiers
            output_count = len(output_files)
            clips_count = len(clips_files)
            broll_count = len(broll_files)
            
            # Détecter les changements
            output_changed = output_count != last_output_count
            clips_changed = clips_count != last_clips_count
            broll_changed = broll_count != last_broll_count
            
            print(f"\n⏰ [{current_time}] Surveillance #{iteration}")
            print("-" * 60)
            print(f"   📁 Output: {output_count} fichiers")
            print(f"   📁 Clips: {clips_count} fichiers")
            print(f"   📚 B-roll Library: {broll_count} fichiers")
            
            # Analyser les changements
            if output_changed or clips_changed or broll_changed:
                print(f"   🔄 Changements détectés !")
                
                # 1. Vérifier la création du dossier 6.mp4
                if output_6_dir.exists():
                    print(f"   🎬 Dossier 6.mp4 créé: {output_6_dir}")
                    
                    # Vérifier les fichiers dans le dossier 6
                    files_in_6 = list(output_6_dir.rglob("*"))
                    print(f"      📁 Fichiers dans 6/: {len(files_in_6)}")
                    
                    for file in files_in_6:
                        if file.is_file():
                            file_size = file.stat().st_size / (1024*1024)
                            print(f"      📄 {file.name}: {file_size:.1f} MB")
                    
                    # Vérifier meta.txt pour les mots-clés B-roll
                    meta_file = output_6_dir / "meta.txt"
                    if meta_file.exists():
                        try:
                            with open(meta_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            if "B-roll Keywords:" in content:
                                print(f"      🎯 MOTS-CLÉS B-ROLL DÉTECTÉS !")
                                
                                # Extraire les mots-clés
                                lines = content.split('\n')
                                for line in lines:
                                    if line.startswith("B-roll Keywords:"):
                                        keywords_part = line.replace("B-roll Keywords:", "").strip()
                                        keywords = [kw.strip() for kw in keywords_part.split(',') if kw.strip()]
                                        print(f"         🎬 {len(keywords)} mots-clés: {', '.join(keywords[:5])}...")
                                        break
                            else:
                                print(f"      ⚠️ Mots-clés B-roll non trouvés dans meta.txt")
                                
                        except Exception as e:
                            print(f"      ❌ Erreur lecture meta.txt: {e}")
                
                # 2. Vérifier la bibliothèque B-roll
                if broll_library.exists():
                    clip_dirs = [d for d in broll_library.iterdir() if d.is_dir() and d.name.startswith('clip_reframed_')]
                    if len(clip_dirs) > last_broll_count:
                        print(f"   📚 Nouveaux clips B-roll détectés: {len(clip_dirs)}")
                        
                        # Vérifier les nouveaux clips
                        new_clips = clip_dirs[-3:] if len(clip_dirs) > 3 else clip_dirs
                        for clip_dir in new_clips:
                            clip_name = clip_dir.name
                            print(f"      📁 {clip_name}")
                            
                            # Vérifier le contenu
                            fetched_dir = clip_dir / "fetched"
                            if fetched_dir.exists():
                                sources = [d.name for d in fetched_dir.iterdir() if d.is_dir()]
                                print(f"         📥 Sources: {', '.join(sources)}")
                
                # 3. Vérifier les logs du pipeline
                log_file = Path("output/pipeline.log.jsonl")
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        # Analyser les nouvelles lignes
                        if len(lines) > last_output_count:
                            new_lines = lines[-10:] if len(lines) > 10 else lines
                            print(f"   📋 Nouvelles lignes du log ({len(new_lines)} lignes):")
                            
                            for line in new_lines[-3:]:  # Afficher les 3 dernières
                                try:
                                    log_entry = json.loads(line.strip())
                                    event_type = log_entry.get('type', 'N/A')
                                    if event_type == 'event_applied':
                                        start_s = log_entry.get('start_s', 'N/A')
                                        end_s = log_entry.get('end_s', 'N/A')
                                        media_path = log_entry.get('media_path', 'N/A')
                                        print(f"      🎬 B-roll appliqué: [{start_s}s-{end_s}s] {os.path.basename(media_path)}")
                                    else:
                                        print(f"      📝 {event_type}: {line[:80]}...")
                                except:
                                    print(f"      ⚠️ Ligne non-JSON: {line[:80]}...")
                    
                    except Exception as e:
                        print(f"      ⚠️ Erreur lecture log: {e}")
                
                # Mettre à jour les compteurs
                last_output_count = output_count
                last_clips_count = clips_count
                last_broll_count = broll_count
            
            # Attendre avant la prochaine vérification
            print(f"   ⏳ Attente 10 secondes... (Ctrl+C pour arrêter)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Surveillance arrêtée par l'utilisateur")
        print(f"⏰ Durée totale: {iteration * 10} secondes")
        print(f"🎯 Vérifiez l'interface pour voir le traitement en cours")
    except Exception as e:
        print(f"\n❌ Erreur lors de la surveillance: {e}")

def main():
    """Fonction principale de surveillance"""
    print("🎯 Surveillance en temps réel du flux LLM → Fetchers → Scoring")
    print("🔍 Observation du traitement de 6.mp4 avec correction des mots-clés B-roll")
    
    surveiller_flux_temps_reel()

if __name__ == "__main__":
    main() 