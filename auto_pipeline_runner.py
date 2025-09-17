#!/usr/bin/env python3
"""
🤖 AUTOMATEUR COMPLET DU PIPELINE VIDÉO
Automatise entièrement le traitement de vidéos avec musique background
"""

import os
import time
import schedule
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import json
import logging

class VideoPipelineAutomator:
    def __init__(self):
        self.input_folder = Path("input")
        self.output_folder = Path("output")
        self.processed_folder = Path("processed")
        self.failed_folder = Path("failed")
        
        # Créer les dossiers nécessaires
        for folder in [self.input_folder, self.output_folder, self.processed_folder, self.failed_folder]:
            folder.mkdir(exist_ok=True)
        
        # Configuration logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('auto_pipeline.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def process_video(self, video_path: Path) -> bool:
        """Traite une vidéo avec le pipeline complet."""
        try:
            self.logger.info(f"DEBUT: Traitement {video_path.name}")
            
            # 1. Traitement principal
            cmd = f'python video_processor.py "{video_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"ERREUR: Traitement {result.stderr}")
                return False
            
            # 2. Ajout musique background
            output_video = self.output_folder / "clips" / video_path.stem / "final_subtitled.mp4"
            if output_video.exists():
                self._add_background_music(output_video)
            
            # 3. Déplacer vers traité
            processed_path = self.processed_folder / video_path.name
            video_path.rename(processed_path)
            
            self.logger.info(f"SUCCES: Traitement terminé {video_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"ERREUR CRITIQUE: {e}")
            return False
    
    def _add_background_music(self, video_path: Path):
        """Ajoute musique background à la vidéo finale."""
        try:
            from video_processor import _add_background_music
            
            output_path = video_path.parent / f"{video_path.stem}_with_music.mp4"
            success = _add_background_music(str(video_path), str(output_path))
            
            if success:
                self.logger.info(f"MUSIQUE AJOUTÉE: {output_path.name}")
            else:
                self.logger.warning(f"ÉCHEC: Ajout musique échoué pour {video_path.name}")
                
        except Exception as e:
            self.logger.error(f"ERREUR MUSIQUE: Ajout musique échoué: {e}")
    
    def watch_folder(self):
        """Surveille le dossier input pour nouveaux fichiers."""
        class VideoHandler(FileSystemEventHandler):
            def __init__(self, automator):
                self.automator = automator
            
            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith(('.mp4', '.mov', '.avi')):
                    video_path = Path(event.src_path)
                    self.automator.logger.info(f"VIDÉO DÉTECTÉE: Nouvelle vidéo détectée: {video_path.name}")
                    
                    # Attendre que le fichier soit complètement écrit
                    time.sleep(2)
                    
                    # Traitement automatique
                    success = self.automator.process_video(video_path)
                    if not success:
                        # Déplacer vers échec
                        failed_path = self.automator.failed_folder / video_path.name
                        video_path.rename(failed_path)
        
        event_handler = VideoHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.input_folder), recursive=False)
        observer.start()
        
        self.logger.info(f"SURVEILLANCE: Surveillance active sur {self.input_folder}")
        return observer
    
    def batch_process(self):
        """Traite toutes les vidéos en attente."""
        self.logger.info("DEBUT: Traitement par lot")
        
        video_files = list(self.input_folder.glob("*.mp4")) + \
                     list(self.input_folder.glob("*.mov")) + \
                     list(self.input_folder.glob("*.avi"))
        
        if not video_files:
            self.logger.info("AUCUNE VIDEO: Aucune vidéo en attente")
            return
        
        self.logger.info(f"VIDEOS: {len(video_files)} vidéos à traiter")
        
        for video_path in video_files:
            self.process_video(video_path)
            time.sleep(5)  # Pause entre traitements
        
        self.logger.info("TERMINE: Traitement par lot terminé")
    
    def run_scheduled(self):
        """Lance le pipeline selon un planning."""
        # Traitement toutes les heures
        schedule.every().hour.do(self.batch_process)
        
        # Traitement au démarrage
        schedule.every().day.at("09:00").do(self.batch_process)
        schedule.every().day.at("14:00").do(self.batch_process)
        schedule.every().day.at("19:00").do(self.batch_process)
        
        self.logger.info("PLANNING: Planning configuré: 9h, 14h, 19h + toutes les heures")
        
        while True:
            schedule.run_pending()
            time.sleep(60)

def main():
    """Fonction principale."""
    print("DÉMARRAGE: AUTOMATEUR PIPELINE VIDÉO")
    
    automator = VideoPipelineAutomator()
    
    # Mode surveillance continue
    if "--watch" in os.sys.argv:
        print("SURVEILLANCE: Mode surveillance active")
        observer = automator.watch_folder()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            observer.join()
    
    # Mode planning
    elif "--scheduled" in os.sys.argv:
        print("PLANNING: Mode planning actif")
        automator.run_scheduled()
    
    # Mode traitement unique
    else:
        print("TRAITEMENT: Mode traitement unique")
        automator.batch_process()

if __name__ == "__main__":
    main() 