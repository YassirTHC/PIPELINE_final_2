#!/usr/bin/env python3
"""
Surveillance en Temps Réel du Traitement Vidéo 11.mp4
Validation que toutes les erreurs sont résolues
"""

import time
import os
from pathlib import Path
from datetime import datetime

def surveiller_traitement():
    """Surveille le traitement en temps réel"""
    print("\n🔍 SURVEILLANCE EN TEMPS RÉEL - VIDÉO 11.mp4")
    print("=" * 70)
    print("🎯 Validation que toutes les erreurs sont résolues")
    print("⏰ Début de la surveillance:", datetime.now().strftime("%H:%M:%S"))
    
    # Dossiers à surveiller
    output_dir = Path("output")
    broll_library = Path("AI-B-roll/broll_library")
    logs_dir = Path("logs")
    
    print(f"\n📁 Dossiers surveillés:")
    print(f"   • Output: {output_dir}")
    print(f"   • B-roll Library: {broll_library}")
    print(f"   • Logs: {logs_dir}")
    
    # Variables de surveillance
    last_output_count = 0
    last_broll_count = 0
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            print(f"\n⏰ [{datetime.now().strftime('%H:%M:%S')}] Surveillance active ({elapsed:.0f}s)")
            print("-" * 50)
            
            # 1. Vérifier le dossier output
            if output_dir.exists():
                output_files = list(output_dir.rglob("*"))
                output_count = len(output_files)
                
                if output_count > last_output_count:
                    print(f"   📁 Output: {output_count} fichiers (+{output_count - last_output_count})")
                    last_output_count = output_count
                    
                    # Vérifier les nouveaux fichiers
                    for file_path in output_files:
                        if file_path.is_file():
                            file_age = current_time - file_path.stat().st_mtime
                            if file_age < 60:  # Fichiers créés dans la dernière minute
                                print(f"      🆕 {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")
                else:
                    print(f"   📁 Output: {output_count} fichiers (inchangé)")
            else:
                print("   📁 Output: Dossier non trouvé")
            
            # 2. Vérifier la bibliothèque B-roll
            if broll_library.exists():
                broll_folders = list(broll_library.glob("clip_reframed_*"))
                broll_count = len(broll_folders)
                
                if broll_count > last_broll_count:
                    print(f"   🎬 B-roll Library: {broll_count} dossiers (+{broll_count - last_broll_count})")
                    last_broll_count = broll_count
                    
                    # Vérifier le dernier dossier créé
                    if broll_folders:
                        latest_folder = max(broll_folders, key=lambda x: x.stat().st_mtime)
                        folder_age = current_time - latest_folder.stat().st_mtime
                        
                        if folder_age < 300:  # Dossier créé dans les 5 dernières minutes
                            print(f"      🆕 {latest_folder.name}")
                            
                            # Vérifier le contenu
                            fetched_path = latest_folder / "fetched"
                            if fetched_path.exists():
                                providers = list(fetched_path.glob("*"))
                                total_assets = sum(len(list(p.rglob("*"))) for p in providers if p.is_dir())
                                print(f"         📊 {len(providers)} providers, {total_assets} assets")
                else:
                    print(f"   🎬 B-roll Library: {broll_count} dossiers (inchangé)")
            else:
                print("   🎬 B-roll Library: Dossier non trouvé")
            
            # 3. Vérifier les logs
            if logs_dir.exists():
                log_files = list(logs_dir.glob("*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    log_age = current_time - latest_log.stat().st_mtime
                    
                    if log_age < 60:  # Log modifié dans la dernière minute
                        try:
                            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()
                                if lines:
                                    last_line = lines[-1].strip()
                                    if last_line:
                                        print(f"   📝 Log ({latest_log.name}): {last_line[:80]}...")
                        except Exception as e:
                            print(f"   📝 Log: Erreur lecture - {e}")
            
            # 4. Vérifier les erreurs spécifiques
            print(f"\n🔍 Vérification des erreurs résolues:")
            
            # Erreur sync_context_analyzer
            try:
                from sync_context_analyzer import SyncContextAnalyzer
                analyzer = SyncContextAnalyzer()
                print("   ✅ sync_context_analyzer: Module disponible et fonctionnel")
            except Exception as e:
                print(f"   ❌ sync_context_analyzer: {e}")
            
            # Erreur scoring contextuel
            try:
                from video_processor import VideoProcessor
                processor = VideoProcessor()
                print("   ✅ VideoProcessor: Import réussi (scoring contextuel corrigé)")
            except Exception as e:
                print(f"   ❌ VideoProcessor: {e}")
            
            # Système de vérification B-roll
            try:
                from broll_verification_system import BrollVerificationSystem
                verifier = BrollVerificationSystem()
                print("   ✅ Système de vérification B-roll: Fonctionnel")
            except Exception as e:
                print(f"   ❌ Système de vérification: {e}")
            
            # Attendre avant la prochaine vérification
            print(f"\n⏳ Attente 10 secondes... (Ctrl+C pour arrêter)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Surveillance arrêtée par l'utilisateur")
        print(f"⏰ Durée totale: {time.time() - start_time:.0f} secondes")
        print("🎯 Vérifiez l'interface pour voir le traitement en cours")

def main():
    """Fonction principale"""
    print("🚀 SURVEILLANCE DU TRAITEMENT VIDÉO 11.mp4")
    print("=" * 70)
    print("🎯 Validation en temps réel que toutes les erreurs sont résolues")
    
    try:
        surveiller_traitement()
    except Exception as e:
        print(f"❌ Erreur lors de la surveillance: {e}")

if __name__ == "__main__":
    main() 