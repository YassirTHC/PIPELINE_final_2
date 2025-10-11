import json
from pathlib import Path
from pipeline_core.llm_service import _build_json_metadata_prompt, _ollama_generate_json, _normalise_string, _default_metadata_payload
segments_path = Path('output/clips/121-001/121_segments.json')
data = json.loads(segments_path.read_text(encoding='utf-8'))
full_text = ' '.join(seg.get('text', '') for seg in data if seg.get('text'))
prompt = _build_json_metadata_prompt(full_text, video_id='clip121_debug')
parsed, raw, raw_len = _ollama_generate_json(prompt, model='qwen3:8b', json_mode=True, timeout=60)
defaults = _default_metadata_payload()
metadata_section = parsed if isinstance(parsed, dict) else {}
title = _normalise_string(metadata_section.get('title')) or defaults['title']
Path('_meta_debug.json').write_text(json.dumps({'metadata_section': metadata_section, 'title': title}, ensure_ascii=False, indent=2), encoding='utf-8')
