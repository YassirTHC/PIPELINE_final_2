from pathlib import Path

path = Path("video_processor.py")
text = path.read_text(encoding="utf-8")
old = "from typing import List, Dict, Any, Optional, Union, Sequence, Set\n"
new = "from typing import List, Dict, Any, Optional, Union, Sequence, Set\nfrom collections import Counter\n"
if old not in text:
    raise SystemExit('import block not found')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
