ï»¿# -*- coding: utf-8 -*-
from video_processor import VideoProcessor
from pathlib import Path

processor = VideoProcessor()

for clip in Path("clips").glob("*.mp4"):
    print(f"Ã°Å¸Å½Â¬ Traitement de : {clip.name}")
    processor.process_single_clip(clip)


