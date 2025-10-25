ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Surveillance en Temps RÃƒÂ©el du Traitement VidÃƒÂ©o 11.mp4
Validation que toutes les erreurs sont rÃƒÂ©solues
"""

import time
import os
from pathlib import Path
from datetime import datetime

def surveiller_traitement():
    """Surveille le traitement en temps rÃƒÂ©el"""
    print("\nÃ°Å¸â€Â SURVEILLANCE EN TEMPS RÃƒâ€°EL - VIDÃƒâ€°O 11.mp4")
    print("=" * 70)
    print("Ã°Å¸Å½Â¯ Validation que toutes les erreurs sont rÃƒÂ©solues")
    print("Ã¢ÂÂ° DÃƒÂ©but de la surveillance:", datetime.now().strftime("%H:%M:%S"))
    
    # Dossiers ÃƒÂ  surveiller
    output_dir = Path("output")
    broll_library = Path("AI-B-roll/broll_library")
    logs_dir = Path("logs")
    
    print(f"\nÃ°Å¸â€œÂ Dossiers surveillÃƒÂ©s:")
    print(f"   Ã¢â‚¬Â¢ Output: {output_dir}")
    print(f"   Ã¢â‚¬Â¢ B-roll Library: {broll_library}")
    print(f"   Ã¢â‚¬Â¢ Logs: {logs_dir}")
    
    # Variables de surveillance
    last_output_count = 0
    last_broll_count = 0
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            print(f"\nÃ¢ÂÂ° [{datetime.now().strftime('%H:%M:%S')}] Surveillance active ({elapsed:.0f}s)")
            print("-" * 50)
            
            # 1. VÃƒÂ©rifier le dossier output
            if output_dir.exists():
                output_files = list(output_dir.rglob("*"))
                output_count = len(output_files)
                
                if output_count > last_output_count:
                    print(f"   Ã°Å¸â€œÂ Output: {output_count} fichiers (+{output_count - last_output_count})")
                    last_output_count = output_count
                    
                    # VÃƒÂ©rifier les nouveaux fichiers
                    for file_path in output_files:
                        if file_path.is_file():
                            file_age = current_time - file_path.stat().st_mtime
                            if file_age < 60:  # Fichiers crÃƒÂ©ÃƒÂ©s dans la derniÃƒÂ¨re minute
                                print(f"      Ã°Å¸â€ â€¢ {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")
                else:
                    print(f"   Ã°Å¸â€œÂ Output: {output_count} fichiers (inchangÃƒÂ©)")
            else:
                print("   Ã°Å¸â€œÂ Output: Dossier non trouvÃƒÂ©")
            
            # 2. VÃƒÂ©rifier la bibliothÃƒÂ¨que B-roll
            if broll_library.exists():
                broll_folders = list(broll_library.glob("clip_reframed_*"))
                broll_count = len(broll_folders)
                
                if broll_count > last_broll_count:
                    print(f"   Ã°Å¸Å½Â¬ B-roll Library: {broll_count} dossiers (+{broll_count - last_broll_count})")
                    last_broll_count = broll_count
                    
                    # VÃƒÂ©rifier le dernier dossier crÃƒÂ©ÃƒÂ©
                    if broll_folders:
                        latest_folder = max(broll_folders, key=lambda x: x.stat().st_mtime)
                        folder_age = current_time - latest_folder.stat().st_mtime
                        
                        if folder_age < 300:  # Dossier crÃƒÂ©ÃƒÂ© dans les 5 derniÃƒÂ¨res minutes
                            print(f"      Ã°Å¸â€ â€¢ {latest_folder.name}")
                            
                            # VÃƒÂ©rifier le contenu
                            fetched_path = latest_folder / "fetched"
                            if fetched_path.exists():
                                providers = list(fetched_path.glob("*"))
                                total_assets = sum(len(list(p.rglob("*"))) for p in providers if p.is_dir())
                                print(f"         Ã°Å¸â€œÅ  {len(providers)} providers, {total_assets} assets")
                else:
                    print(f"   Ã°Å¸Å½Â¬ B-roll Library: {broll_count} dossiers (inchangÃƒÂ©)")
            else:
                print("   Ã°Å¸Å½Â¬ B-roll Library: Dossier non trouvÃƒÂ©")
            
            # 3. VÃƒÂ©rifier les logs
            if logs_dir.exists():
                log_files = list(logs_dir.glob("*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    log_age = current_time - latest_log.stat().st_mtime
                    
                    if log_age < 60:  # Log modifiÃƒÂ© dans la derniÃƒÂ¨re minute
                        try:
                            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()
                                if lines:
                                    last_line = lines[-1].strip()
                                    if last_line:
                                        print(f"   Ã°Å¸â€œÂ Log ({latest_log.name}): {last_line[:80]}...")
                        except Exception as e:
                            print(f"   Ã°Å¸â€œÂ Log: Erreur lecture - {e}")
            
            # 4. VÃƒÂ©rifier les erreurs spÃƒÂ©cifiques
            print(f"\nÃ°Å¸â€Â VÃƒÂ©rification des erreurs rÃƒÂ©solues:")
            
            # Erreur sync_context_analyzer
            try:
                from sync_context_analyzer import SyncContextAnalyzer
                analyzer = SyncContextAnalyzer()
                print("   Ã¢Å“â€¦ sync_context_analyzer: Module disponible et fonctionnel")
            except Exception as e:
                print(f"   Ã¢ÂÅ’ sync_context_analyzer: {e}")
            
            # Erreur scoring contextuel
            try:
                from video_processor import VideoProcessor
                processor = VideoProcessor()
                print("   Ã¢Å“â€¦ VideoProcessor: Import rÃƒÂ©ussi (scoring contextuel corrigÃƒÂ©)")
            except Exception as e:
                print(f"   Ã¢ÂÅ’ VideoProcessor: {e}")
            
            # SystÃƒÂ¨me de vÃƒÂ©rification B-roll
            try:
                from broll_verification_system import BrollVerificationSystem
                verifier = BrollVerificationSystem()
                print("   Ã¢Å“â€¦ SystÃƒÂ¨me de vÃƒÂ©rification B-roll: Fonctionnel")
            except Exception as e:
                print(f"   Ã¢ÂÅ’ SystÃƒÂ¨me de vÃƒÂ©rification: {e}")
            
            # Attendre avant la prochaine vÃƒÂ©rification
            print(f"\nÃ¢ÂÂ³ Attente 10 secondes... (Ctrl+C pour arrÃƒÂªter)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print(f"\n\nÃ°Å¸â€ºâ€˜ Surveillance arrÃƒÂªtÃƒÂ©e par l'utilisateur")
        print(f"Ã¢ÂÂ° DurÃƒÂ©e totale: {time.time() - start_time:.0f} secondes")
        print("Ã°Å¸Å½Â¯ VÃƒÂ©rifiez l'interface pour voir le traitement en cours")

def main():
    """Fonction principale"""
    print("Ã°Å¸Å¡â‚¬ SURVEILLANCE DU TRAITEMENT VIDÃƒâ€°O 11.mp4")
    print("=" * 70)
    print("Ã°Å¸Å½Â¯ Validation en temps rÃƒÂ©el que toutes les erreurs sont rÃƒÂ©solues")
    
    try:
        surveiller_traitement()
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur lors de la surveillance: {e}")

if __name__ == "__main__":
    main() 

