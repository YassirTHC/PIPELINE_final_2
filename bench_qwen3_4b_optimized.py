#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 BENCH QWEN3:4B OPTIMISÉ - PROMPT DIRECTIF
Test de qwen3:4b avec un prompt optimisé pour éviter le mode "thinking"
"""

import time
import requests
import json
import psutil
from datetime import datetime

def bench_qwen3_4b_optimized():
    """Benchmark de qwen3:4b avec prompt optimisé"""
    
    print("🧪 BENCH QWEN3:4B OPTIMISÉ - PROMPT DIRECTIF")
    print("=" * 60)
    print(f"⏰ Début: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Configuration
    URL = "http://localhost:11434/api/generate"
    MODEL = "qwen3:4b"
    
    # Prompt ultra-directif (pas de "thinking")
    PROMPT = '''Generate 5 filmable keywords for "family playing in park".
Return ONLY valid JSON: {"keywords":["k1","k2","k3","k4","k5"]}
No explanations, no thinking, just JSON.'''
    
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "temperature": 0.1,  # Plus déterministe
        "max_tokens": 100,   # Réduit pour accélérer
        "stream": False,
        "top_p": 0.9,       # Contrôle de la créativité
        "top_k": 40         # Limite les choix
    }
    
    print(f"🎯 Modèle: {MODEL}")
    print(f"📝 Prompt: {len(PROMPT)} caractères")
    print(f"⏱️ Timeout: 60s (réduit)")
    print(f"📊 Max tokens: 100")
    print(f"🌡️ Temperature: 0.1 (déterministe)")
    print()
    
    # Monitoring système avant
    print("📊 MONITORING SYSTÈME - AVANT")
    print("-" * 40)
    mem_before = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    print(f"💾 RAM disponible: {mem_before.available / 1e9:.2f} GB")
    print(f"💾 RAM utilisée: {mem_before.used / 1e9:.2f} GB")
    print(f"🔄 CPU: {cpu_percent}%")
    print()
    
    # Test LLM
    print("🚀 TEST QWEN3:4B OPTIMISÉ EN COURS...")
    print("-" * 40)
    
    try:
        t0 = time.time()
        r = requests.post(URL, json=payload, timeout=60)
        t1 = time.time()
        
        elapsed = t1 - t0
        status = r.status_code
        
        print(f"✅ Statut: {status}")
        print(f"⏱️ Temps total: {elapsed:.2f}s")
        print(f"📊 Latence: {elapsed*1000:.0f}ms")
        
        if r.status_code == 200:
            try:
                response_data = r.json()
                response_text = response_data.get('response', '')
                print(f"📝 Réponse: {len(response_text)} caractères")
                print(f"🔍 Début réponse: {response_text[:200]}...")
                
                # Test parsing JSON
                try:
                    parsed_json = json.loads(response_text)
                    print("✅ JSON valide détecté")
                    
                    # Analyse de la structure
                    if 'keywords' in parsed_json:
                        keywords = parsed_json['keywords']
                        print(f"🎯 Mots-clés trouvés: {len(keywords)}")
                        print(f"📝 Mots-clés: {keywords}")
                        
                        if isinstance(keywords, list) and len(keywords) >= 5:
                            print("✅ Nombre de mots-clés OK (≥5)")
                        else:
                            print("⚠️ Nombre de mots-clés insuffisant")
                    else:
                        print("⚠️ Structure 'keywords' manquante")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON invalide: {e}")
                    print("🔍 Tentative de réparation...")
                    
                    # Tentative de réparation simple
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            parsed_json = json.loads(json_str)
                            print("✅ JSON réparé avec succès")
                        else:
                            print("❌ Impossible de réparer le JSON")
                    except:
                        print("❌ Réparation JSON échouée")
                        
            except Exception as e:
                print(f"❌ Erreur parsing réponse: {e}")
                print(f"📝 Réponse brute: {r.text[:200]}...")
        else:
            print(f"❌ Erreur HTTP: {r.text}")
            
    except requests.exceptions.Timeout:
        print("⏱️ TIMEOUT après 60s")
        elapsed = 60
        status = "TIMEOUT"
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        elapsed = 0
        status = "ERROR"
    
    # Monitoring système après
    print()
    print("📊 MONITORING SYSTÈME - APRÈS")
    print("-" * 40)
    mem_after = psutil.virtual_memory()
    cpu_percent_after = psutil.cpu_percent(interval=1)
    
    mem_delta = mem_before.available - mem_after.available
    print(f"💾 RAM delta: {mem_delta / 1e6:.1f} MB")
    print(f"💾 RAM disponible: {mem_after.available / 1e9:.2f} GB")
    print(f"🔄 CPU: {cpu_percent_after}%")
    
    # Analyse des résultats
    print()
    print("🔍 ANALYSE DES RÉSULTATS")
    print("=" * 60)
    
    if elapsed < 10:
        print("✅ QWEN3:4B OPTIMISÉ - Temps excellent (<10s)")
        print("🎯 Problème résolu: Prompt optimisé fonctionne")
    elif elapsed < 30:
        print("⚠️ QWEN3:4B OPTIMISÉ - Temps acceptable (10-30s)")
        print("🎯 Amélioration significative, peut être optimisé")
    elif elapsed < 60:
        print("❌ QWEN3:4B OPTIMISÉ - Temps élevé (30-60s)")
        print("🎯 Problème persiste, vérifier configuration")
    else:
        print("🚨 QWEN3:4B OPTIMISÉ - Timeout atteint")
        print("🎯 Problème critique, modèle inutilisable")
    
    print()
    print("📋 RECOMMANDATIONS")
    print("-" * 40)
    
    if elapsed < 30:
        print("1. ✅ QWEN3:4B fonctionne avec prompt optimisé")
        print("2. 🎯 Implémenter ce prompt dans le pipeline")
        print("3. 🔧 Supprimer qwen3:8b inutile")
    else:
        print("1. ⚠️ Vérifier configuration Ollama")
        print("2. 🔍 Tester avec d'autres paramètres")
        print("3. 🚨 Considérer un modèle plus léger")
    
    return elapsed, status

if __name__ == "__main__":
    elapsed, status = bench_qwen3_4b_optimized()
    
    print()
    print("=" * 60)
    print(f"🏁 BENCH OPTIMISÉ TERMINÉ - Temps: {elapsed:.2f}s, Statut: {status}")
    
    input("\nAppuyez sur Entrée pour continuer...") 