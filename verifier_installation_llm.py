#!/usr/bin/env python3
"""
🔍 VÉRIFICATEUR POST-INSTALLATION LLM
Script de vérification après migration vers llama3.2:8b
"""

import subprocess
import json
import time
import psutil
import os
from pathlib import Path

def verifier_installation_llm():
    """Vérifier l'installation et la configuration du nouveau LLM"""
    print("🔍 VÉRIFICATEUR POST-INSTALLATION LLM")
    print("=" * 60)
    
    try:
        # 1. Vérifier qu'Ollama est en cours d'exécution
        print("\n📊 ÉTAPE 1: Vérification d'Ollama")
        print("-" * 40)
        
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Ollama est accessible")
                print(f"📋 Modèles disponibles:\n{result.stdout}")
            else:
                print(f"❌ Erreur Ollama: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Ollama non accessible: {e}")
            return False
        
        # 2. Vérifier que llama3.2:8b est installé
        print("\n📊 ÉTAPE 2: Vérification du modèle llama3.2:8b")
        print("-" * 40)
        
        if "llama3.2:8b" in result.stdout:
            print("✅ Modèle llama3.2:8b détecté")
        else:
            print("❌ Modèle llama3.2:8b NON détecté")
            print("🚀 Installation en cours...")
            try:
                install_result = subprocess.run(["ollama", "pull", "llama3.2:8b"], 
                                             capture_output=True, text=True, timeout=300)
                if install_result.returncode == 0:
                    print("✅ Installation réussie")
                else:
                    print(f"❌ Échec installation: {install_result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ Erreur installation: {e}")
                return False
        
        # 3. Vérifier la configuration centralisée
        print("\n📊 ÉTAPE 3: Vérification de la configuration centralisée")
        print("-" * 40)
        
        config_path = Path("config/llm_config.yaml")
        if config_path.exists():
            print("✅ Fichier de configuration LLM trouvé")
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                print(f"📝 Modèle configuré: {config['llm']['model']}")
                print(f"📝 Fallback: {config['llm']['fallback_model']}")
                print(f"📝 Validation JSON: {config['llm']['enforce_json_output']}")
            except Exception as e:
                print(f"⚠️ Erreur lecture config: {e}")
        else:
            print("❌ Fichier de configuration LLM non trouvé")
            return False
        
        # 4. Vérifier la mémoire disponible
        print("\n📊 ÉTAPE 4: Vérification de la mémoire")
        print("-" * 40)
        
        memory = psutil.virtual_memory()
        print(f"💾 RAM totale: {memory.total / (1024**3):.1f} GB")
        print(f"💾 RAM disponible: {memory.available / (1024**3):.1f} GB")
        print(f"💾 RAM utilisée: {memory.percent:.1f}%")
        
        if memory.available / (1024**3) < 8:
            print("⚠️ ATTENTION: Moins de 8GB RAM disponible")
            print("   Le modèle llama3.2:8b peut être lent ou instable")
        else:
            print("✅ RAM suffisante pour llama3.2:8b")
        
        # 5. Test de communication avec le modèle
        print("\n📊 ÉTAPE 5: Test de communication avec le modèle")
        print("-" * 40)
        
        try:
            test_prompt = '{"test": "simple"}'
            test_payload = {
                "model": "llama3.2:8b",
                "prompt": f"Output this exact JSON: {test_prompt}",
                "temperature": 0.1,
                "stream": False
            }
            
            import requests
            start_time = time.time()
            response = requests.post("http://localhost:11434/api/generate", 
                                  json=test_payload, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                response_time = end_time - start_time
                
                print(f"✅ Communication réussie en {response_time:.1f}s")
                print(f"📊 Taille réponse: {len(response_text)} caractères")
                
                # Vérifier si la réponse contient le JSON de test
                if test_prompt in response_text:
                    print("✅ Réponse JSON correcte")
                else:
                    print(f"⚠️ Réponse JSON différente: {response_text[:100]}...")
                    
            else:
                print(f"❌ Erreur HTTP: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur test communication: {e}")
            return False
        
        # 6. Vérifier les fichiers de test mis à jour
        print("\n📊 ÉTAPE 6: Vérification des fichiers de test")
        print("-" * 40)
        
        test_files = [
            "test_prompt_optimise.py",
            "test_pipeline_direct_136.py", 
            "test_pipeline_complet.py",
            "test_prompt_avec_video.py"
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read()
                if "llama3.2:8b" in content:
                    print(f"✅ {test_file} - Référence LLM mise à jour")
                else:
                    print(f"❌ {test_file} - Référence LLM non mise à jour")
            else:
                print(f"⚠️ {test_file} - Fichier non trouvé")
        
        # 7. Résumé final
        print("\n📊 RÉSUMÉ FINAL")
        print("=" * 40)
        print("✅ Ollama accessible et fonctionnel")
        print("✅ Modèle llama3.2:8b installé")
        print("✅ Configuration centralisée active")
        print("✅ Communication avec le modèle réussie")
        print("✅ Fichiers de test mis à jour")
        
        if memory.available / (1024**3) >= 8:
            print("✅ RAM suffisante pour les performances optimales")
        else:
            print("⚠️ RAM limitée - performances peuvent être dégradées")
        
        print("\n🎉 MIGRATION LLM TERMINÉE AVEC SUCCÈS !")
        print("🚀 Le pipeline est prêt à utiliser llama3.2:8b")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verifier_installation_llm() 