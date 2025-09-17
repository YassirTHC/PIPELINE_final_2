import sys, importlib.util

UTILS_DIR = r"C:\\Users\\Administrator\\Desktop\\video_pipeline - Copy\\utils"
INIT_PATH = UTILS_DIR + "\\__init__.py"
spec = importlib.util.spec_from_file_location('utils', INIT_PATH, submodule_search_locations=[UTILS_DIR])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules['utils'] = module

from pathlib import Path
from video_processor import VideoProcessor

clip_path = Path('clips/121.mp4')
print(f"[INFO] Checking clip path: {clip_path.resolve()}")
if not clip_path.exists():
    raise SystemExit('Clip 121.mp4 introuvable dans clips/')

processor = VideoProcessor()
print('[INFO] VideoProcessor initialised')

output = processor.process_single_clip(clip_path)
print('[INFO] Processing finished:', output)
