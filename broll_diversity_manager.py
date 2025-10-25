ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
GESTIONNAIRE DE DIVERSITÃƒâ€° B-ROLL - Ãƒâ€°VITE LA RÃƒâ€°PÃƒâ€°TITION
"""
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Set, List, Optional
import hashlib

class BrollDiversityManager:
    """GÃƒÂ¨re la diversitÃƒÂ© et ÃƒÂ©vite la rÃƒÂ©pÃƒÂ©tition des B-rolls"""
    
    def __init__(self):
        self.used_brolls: Set[str] = set()
        self.broll_usage_count: Dict[str, int] = {}
        self.last_usage_time: Dict[str, datetime] = {}
        self.session_start = datetime.now()
        self.diversity_config = self.load_diversity_config()
        
    def load_diversity_config(self) -> Dict:
        """Charge la configuration de diversitÃƒÂ©"""
        config_path = "broll_diversity_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Adapter la structure de configuration
                    return {
                        "max_reuse_per_broll": config.get("diversity_settings", {}).get("max_reuse_per_broll", 2),
                        "rotation_threshold": config.get("diversity_settings", {}).get("rotation_threshold", 5),
                        "context_similarity_threshold": config.get("diversity_settings", {}).get("context_similarity_threshold", 0.8),
                        "force_new_search_after": config.get("diversity_settings", {}).get("force_new_search_after", 3),
                        "max_consecutive_uses": config.get("forbidden_reuse", {}).get("max_consecutive_uses", 1),
                        "min_time_between_uses": config.get("forbidden_reuse", {}).get("min_time_between_uses", 300),
                        "max_uses_per_session": config.get("forbidden_reuse", {}).get("max_uses_per_session", 2)
                    }
            except Exception as e:
                print(f"Ã¢Å¡Â Ã¯Â¸Â  Erreur chargement config diversitÃƒÂ©: {e}")
        
        # Configuration par dÃƒÂ©faut
        return {
            "max_reuse_per_broll": 2,
            "rotation_threshold": 5,
            "context_similarity_threshold": 0.8,
            "force_new_search_after": 3,
            "max_consecutive_uses": 1,
            "min_time_between_uses": 300,  # 5 minutes
            "max_uses_per_session": 2
        }
    
    def can_use_broll(self, broll_path: str, context: str) -> bool:
        """VÃƒÂ©rifie si un B-roll peut ÃƒÂªtre utilisÃƒÂ©"""
        try:
            # CrÃƒÂ©er une signature unique du B-roll
            broll_signature = self.create_broll_signature(broll_path)
            
            # VÃƒÂ©rifier le nombre d'utilisations
            usage_count = self.broll_usage_count.get(broll_signature, 0)
            if usage_count >= self.diversity_config["max_reuse_per_broll"]:
                print(f"    Ã°Å¸Å¡Â« B-roll bloquÃƒÂ©: utilisation maximale atteinte ({usage_count})")
                return False
            
            # VÃƒÂ©rifier l'utilisation consÃƒÂ©cutive
            if broll_signature in self.used_brolls:
                print(f"    Ã°Å¸Å¡Â« B-roll bloquÃƒÂ©: utilisation consÃƒÂ©cutive dÃƒÂ©tectÃƒÂ©e")
                return False
            
            # VÃƒÂ©rifier le temps entre utilisations
            if broll_signature in self.last_usage_time:
                time_diff = (datetime.now() - self.last_usage_time[broll_signature]).total_seconds()
                if time_diff < self.diversity_config["min_time_between_uses"]:
                    print(f"    Ã°Å¸Å¡Â« B-roll bloquÃƒÂ©: temps minimum non respectÃƒÂ© ({time_diff:.0f}s)")
                    return False
            
            # VÃƒÂ©rifier l'utilisation par session
            session_usage = sum(1 for sig in self.used_brolls if sig == broll_signature)
            if session_usage >= self.diversity_config["max_uses_per_session"]:
                print(f"    Ã°Å¸Å¡Â« B-roll bloquÃƒÂ©: limite session atteinte ({session_usage})")
                return False
            
            return True
            
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur vÃƒÂ©rification diversitÃƒÂ©: {e}")
            return True  # En cas d'erreur, autoriser l'utilisation
    
    def create_broll_signature(self, broll_path: str) -> str:
        """CrÃƒÂ©e une signature unique pour un B-roll"""
        try:
            # Utiliser le nom du fichier et la taille pour crÃƒÂ©er une signature
            path_obj = Path(broll_path)
            file_name = path_obj.name
            file_size = os.path.getsize(broll_path) if os.path.exists(broll_path) else 0
            
            signature_data = f"{file_name}_{file_size}"
            return hashlib.md5(signature_data.encode()).hexdigest()
            
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur crÃƒÂ©ation signature: {e}")
            return broll_path
    
    def mark_broll_used(self, broll_path: str):
        """Marque un B-roll comme utilisÃƒÂ©"""
        try:
            broll_signature = self.create_broll_signature(broll_path)
            
            # Ajouter aux B-rolls utilisÃƒÂ©s
            self.used_brolls.add(broll_signature)
            
            # IncrÃƒÂ©menter le compteur d'utilisation
            self.broll_usage_count[broll_signature] = self.broll_usage_count.get(broll_signature, 0) + 1
            
            # Mettre ÃƒÂ  jour le temps d'utilisation
            self.last_usage_time[broll_signature] = datetime.now()
            
            print(f"    Ã¢Å“â€¦ B-roll marquÃƒÂ© comme utilisÃƒÂ©: {Path(broll_path).name}")
            
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur marquage B-roll: {e}")
    
    def get_diversity_score(self) -> float:
        """Calcule le score de diversitÃƒÂ© actuel"""
        try:
            total_brolls = len(self.broll_usage_count)
            if total_brolls == 0:
                return 1.0
            
            # Calculer la diversitÃƒÂ© basÃƒÂ©e sur la rÃƒÂ©partition des utilisations
            usage_values = list(self.broll_usage_count.values())
            avg_usage = sum(usage_values) / len(usage_values)
            max_usage = max(usage_values) if usage_values else 0
            
            if max_usage == 0:
                return 1.0
            
            # Score basÃƒÂ© sur la rÃƒÂ©partition (plus c'est ÃƒÂ©quilibrÃƒÂ©, meilleur c'est)
            diversity_score = 1.0 - (avg_usage / max_usage)
            return max(0.0, min(1.0, diversity_score))
            
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur calcul diversitÃƒÂ©: {e}")
            return 0.5
    
    def reset_session(self):
        """RÃƒÂ©initialise la session pour une nouvelle vidÃƒÂ©o"""
        self.used_brolls.clear()
        self.session_start = datetime.now()
        print("    Ã°Å¸â€â€ž Session diversitÃƒÂ© B-roll rÃƒÂ©initialisÃƒÂ©e")
    
    def get_diversity_report(self) -> Dict:
        """GÃƒÂ©nÃƒÂ¨re un rapport de diversitÃƒÂ©"""
        try:
            return {
                "diversity_score": self.get_diversity_score(),
                "total_brolls": len(self.broll_usage_count),
                "session_duration": (datetime.now() - self.session_start).total_seconds(),
                "most_used_broll": max(self.broll_usage_count.items(), key=lambda x: x[1]) if self.broll_usage_count else None,
                "least_used_broll": min(self.broll_usage_count.items(), key=lambda x: x[1]) if self.broll_usage_count else None
            }
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur rapport diversitÃƒÂ©: {e}")
            return {} 

