#!/usr/bin/env python3
import json
import os

print("=== ANALYSE 6.MP4 ===")

# 1. Fichiers générés
print("1. Fichiers générés:")
files = os.listdir('output/clips/6')
for f in files:
    print(f"   {f}")

# 2. Meta.txt
print("\n2. Meta.txt:")
try:
    meta = open('output/clips/6/meta.txt', 'r').read()
    print(meta)
except:
    print("   Erreur lecture meta.txt")

# 3. Segments
print("\n3. Segments:")
try:
    segments = json.load(open('output/clips/6/6_segments.json', 'r'))
    print(f"   {len(segments)} segments")
except:
    print("   Erreur lecture segments")

# 4. Pipeline log
print("\n4. Pipeline log:")
try:
    log_lines = open('output/pipeline.log.jsonl', 'r').readlines()
    print(f"   {len(log_lines)} lignes")
    recent = [l for l in log_lines[-5:] if '6' in l]
    print(f"   {len(recent)} lignes récentes avec 6")
    for l in recent:
        print(f"      {l[:100]}...")
except:
    print("   Erreur lecture log") 