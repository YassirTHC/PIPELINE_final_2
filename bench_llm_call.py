#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 BENCH LLM CALL - TEST LATENCE MINIMALE
Test de latence pour identifier si le problème vient de l'infrastructure ou du modèle
"""

import time
import requests
import json
import psutil
from datetime import datetime

def bench_llm_call():
    """Benchmark d'un appel LLM minimal"""
    
    print("🧪 BENCH LLM CALL - TEST LATENCE MINIMALE")
    print("=" * 60)
    print(f"⏰ Début: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Configuration
    URL = "http://localhost:11434/api/generate"
    MODEL = "qwen3:8b"
    
    # Prompt ultra-simple
    PROMPT = '''Generate 5 filmable keywords for: "family playing in park". 
Return JSON: {"keywords":["k1","k2","k3","k4","k5"]}'''
    
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "temperature": 0.2,
        "max_tokens": 200,
        "stream": False
    }
    
    print(f"🎯 Modèle: {MODEL}")
    print(f"📝 Prompt: {len(PROMPT)} caractères")
    print(f"⏱️ Timeout: 120s")
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
    print("🚀 TEST LLM EN COURS...")
    print("-" * 40)
    
    try:
        t0 = time.time()
        r = requests.post(URL, json=payload, timeout=120)
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
                    json.loads(response_text)
                    print("✅ JSON valide détecté")
                except:
                    print("⚠️ JSON invalide dans la réponse")
                    
            except Exception as e:
                print(f"❌ Erreur parsing réponse: {e}")
                print(f"📝 Réponse brute: {r.text[:200]}...")
        else:
            print(f"❌ Erreur HTTP: {r.text}")
            
    except requests.exceptions.Timeout:
        print("⏱️ TIMEOUT après 120s")
        elapsed = 120
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
    
    if elapsed < 5:
        print("✅ INFRA OK - Latence normale (<5s)")
        print("🎯 Problème probable: Prompt trop complexe")
    elif elapsed < 10:
        print("⚠️ INFRA LENTE - Latence élevée (5-10s)")
        print("🎯 Problème probable: Modèle ou configuration Ollama")
    elif elapsed < 30:
        print("❌ INFRA PROBLÉMATIQUE - Latence très élevée (10-30s)")
        print("🎯 Problème probable: Modèle quantisé mal ou RAM insuffisante")
    else:
        print("🚨 INFRA CRITIQUE - Latence excessive (>30s)")
        print("🎯 Problème probable: Swapping, modèle corrompu, ou configuration critique")
    
    print()
    print("📋 RECOMMANDATIONS")
    print("-" * 40)
    
    if elapsed < 5:
        print("1. ✅ Infra OK - Tester prompt complexe maintenant")
        print("2. 🎯 Simplifier le prompt de génération de mots-clés")
        print("3. 🔧 Implémenter fallback heuristique")
    elif elapsed < 30:
        print("1. ⚠️ Vérifier configuration Ollama")
        print("2. 🔍 Tester modèle quantisé (qwen3:4b)")
        print("3. 💾 Vérifier utilisation RAM/swap")
    else:
        print("1. 🚨 Vérifier immédiatement l'état du système")
        print("2. 🔄 Redémarrer Ollama")
        print("3. 📦 Réinstaller le modèle")
    
    return elapsed, status

if __name__ == "__main__":
    elapsed, status = bench_llm_call()
    
    print()
    print("=" * 60)
    print(f"🏁 BENCH TERMINÉ - Temps: {elapsed:.2f}s, Statut: {status}")
    
    input("\nAppuyez sur Entrée pour continuer...") 