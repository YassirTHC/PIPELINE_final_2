from video_processor import VideoProcessor
from pathlib import Path

processor = VideoProcessor()

for clip in Path("clips").glob("*.mp4"):
    print(f"🎬 Traitement de : {clip.name}")
    processor.process_single_clip(clip)
