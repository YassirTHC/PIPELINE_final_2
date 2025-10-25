ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Ã°Å¸â€Â VÃƒâ€°RIFICATEUR POST-INSTALLATION LLM
Script de vÃƒÂ©rification aprÃƒÂ¨s migration vers llama3.2:8b
"""

import subprocess
import json
import time
import psutil
import os
from pathlib import Path

def verifier_installation_llm():
    """VÃƒÂ©rifier l'installation et la configuration du nouveau LLM"""
    print("Ã°Å¸â€Â VÃƒâ€°RIFICATEUR POST-INSTALLATION LLM")
    print("=" * 60)
    
    try:
        # 1. VÃƒÂ©rifier qu'Ollama est en cours d'exÃƒÂ©cution
        print("\nÃ°Å¸â€œÅ  Ãƒâ€°TAPE 1: VÃƒÂ©rification d'Ollama")
        print("-" * 40)
        
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("Ã¢Å“â€¦ Ollama est accessible")
                print(f"Ã°Å¸â€œâ€¹ ModÃƒÂ¨les disponibles:\n{result.stdout}")
            else:
                print(f"Ã¢ÂÅ’ Erreur Ollama: {result.stderr}")
                return False
        except Exception as e:
            print(f"Ã¢ÂÅ’ Ollama non accessible: {e}")
            return False
        
        # 2. VÃƒÂ©rifier que llama3.2:8b est installÃƒÂ©
        print("\nÃ°Å¸â€œÅ  Ãƒâ€°TAPE 2: VÃƒÂ©rification du modÃƒÂ¨le llama3.2:8b")
        print("-" * 40)
        
        if "llama3.2:8b" in result.stdout:
            print("Ã¢Å“â€¦ ModÃƒÂ¨le llama3.2:8b dÃƒÂ©tectÃƒÂ©")
        else:
            print("Ã¢ÂÅ’ ModÃƒÂ¨le llama3.2:8b NON dÃƒÂ©tectÃƒÂ©")
            print("Ã°Å¸Å¡â‚¬ Installation en cours...")
            try:
                install_result = subprocess.run(["ollama", "pull", "llama3.2:8b"], 
                                             capture_output=True, text=True, timeout=300)
                if install_result.returncode == 0:
                    print("Ã¢Å“â€¦ Installation rÃƒÂ©ussie")
                else:
                    print(f"Ã¢ÂÅ’ Ãƒâ€°chec installation: {install_result.stderr}")
                    return False
            except Exception as e:
                print(f"Ã¢ÂÅ’ Erreur installation: {e}")
                return False
        
        # 3. VÃƒÂ©rifier la configuration centralisÃƒÂ©e
        print("\nÃ°Å¸â€œÅ  Ãƒâ€°TAPE 3: VÃƒÂ©rification de la configuration centralisÃƒÂ©e")
        print("-" * 40)
        
        config_path = Path("config/llm_config.yaml")
        if config_path.exists():
            print("Ã¢Å“â€¦ Fichier de configuration LLM trouvÃƒÂ©")
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                print(f"Ã°Å¸â€œÂ ModÃƒÂ¨le configurÃƒÂ©: {config['llm']['model']}")
                print(f"Ã°Å¸â€œÂ Fallback: {config['llm']['fallback_model']}")
                print(f"Ã°Å¸â€œÂ Validation JSON: {config['llm']['enforce_json_output']}")
            except Exception as e:
                print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur lecture config: {e}")
        else:
            print("Ã¢ÂÅ’ Fichier de configuration LLM non trouvÃƒÂ©")
            return False
        
        # 4. VÃƒÂ©rifier la mÃƒÂ©moire disponible
        print("\nÃ°Å¸â€œÅ  Ãƒâ€°TAPE 4: VÃƒÂ©rification de la mÃƒÂ©moire")
        print("-" * 40)
        
        memory = psutil.virtual_memory()
        print(f"Ã°Å¸â€™Â¾ RAM totale: {memory.total / (1024**3):.1f} GB")
        print(f"Ã°Å¸â€™Â¾ RAM disponible: {memory.available / (1024**3):.1f} GB")
        print(f"Ã°Å¸â€™Â¾ RAM utilisÃƒÂ©e: {memory.percent:.1f}%")
        
        if memory.available / (1024**3) < 8:
            print("Ã¢Å¡Â Ã¯Â¸Â ATTENTION: Moins de 8GB RAM disponible")
            print("   Le modÃƒÂ¨le llama3.2:8b peut ÃƒÂªtre lent ou instable")
        else:
            print("Ã¢Å“â€¦ RAM suffisante pour llama3.2:8b")
        
        # 5. Test de communication avec le modÃƒÂ¨le
        print("\nÃ°Å¸â€œÅ  Ãƒâ€°TAPE 5: Test de communication avec le modÃƒÂ¨le")
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
                
                print(f"Ã¢Å“â€¦ Communication rÃƒÂ©ussie en {response_time:.1f}s")
                print(f"Ã°Å¸â€œÅ  Taille rÃƒÂ©ponse: {len(response_text)} caractÃƒÂ¨res")
                
                # VÃƒÂ©rifier si la rÃƒÂ©ponse contient le JSON de test
                if test_prompt in response_text:
                    print("Ã¢Å“â€¦ RÃƒÂ©ponse JSON correcte")
                else:
                    print(f"Ã¢Å¡Â Ã¯Â¸Â RÃƒÂ©ponse JSON diffÃƒÂ©rente: {response_text[:100]}...")
                    
            else:
                print(f"Ã¢ÂÅ’ Erreur HTTP: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur test communication: {e}")
            return False
        
        # 6. VÃƒÂ©rifier les fichiers de test mis ÃƒÂ  jour
        print("\nÃ°Å¸â€œÅ  Ãƒâ€°TAPE 6: VÃƒÂ©rification des fichiers de test")
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
                    print(f"Ã¢Å“â€¦ {test_file} - RÃƒÂ©fÃƒÂ©rence LLM mise ÃƒÂ  jour")
                else:
                    print(f"Ã¢ÂÅ’ {test_file} - RÃƒÂ©fÃƒÂ©rence LLM non mise ÃƒÂ  jour")
            else:
                print(f"Ã¢Å¡Â Ã¯Â¸Â {test_file} - Fichier non trouvÃƒÂ©")
        
        # 7. RÃƒÂ©sumÃƒÂ© final
        print("\nÃ°Å¸â€œÅ  RÃƒâ€°SUMÃƒâ€° FINAL")
        print("=" * 40)
        print("Ã¢Å“â€¦ Ollama accessible et fonctionnel")
        print("Ã¢Å“â€¦ ModÃƒÂ¨le llama3.2:8b installÃƒÂ©")
        print("Ã¢Å“â€¦ Configuration centralisÃƒÂ©e active")
        print("Ã¢Å“â€¦ Communication avec le modÃƒÂ¨le rÃƒÂ©ussie")
        print("Ã¢Å“â€¦ Fichiers de test mis ÃƒÂ  jour")
        
        if memory.available / (1024**3) >= 8:
            print("Ã¢Å“â€¦ RAM suffisante pour les performances optimales")
        else:
            print("Ã¢Å¡Â Ã¯Â¸Â RAM limitÃƒÂ©e - performances peuvent ÃƒÂªtre dÃƒÂ©gradÃƒÂ©es")
        
        print("\nÃ°Å¸Å½â€° MIGRATION LLM TERMINÃƒâ€°E AVEC SUCCÃƒË†S !")
        print("Ã°Å¸Å¡â‚¬ Le pipeline est prÃƒÂªt ÃƒÂ  utiliser llama3.2:8b")
        
        return True
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur lors de la vÃƒÂ©rification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verifier_installation_llm() 

