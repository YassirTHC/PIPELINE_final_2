#!/usr/bin/env python3
"""
NETTOYAGE URGENT DE L'ESPACE DISQUE
Supprime les doublons B-roll et libère l'espace
"""

import os
import shutil
from pathlib import Path
import json
from typing import Dict, List
import time

def get_folder_size(folder_path: Path) -> float:
    """Calcule la taille d'un dossier en GB"""
    try:
        total_size = sum(
            f.stat().st_size for f in folder_path.rglob('*') if f.is_file()
        )
        return total_size / (1024**3)  # Convert to GB
    except Exception:
        return 0.0

def analyze_broll_duplicates(broll_dir: Path) -> Dict[str, List[Path]]:
    """Analyse les doublons de B-roll par clip"""
    duplicates = {}
    
    for folder in broll_dir.iterdir():
        if folder.is_dir() and folder.name.startswith('clip_'):
            # Extraire le nom du clip (sans timestamp)
            parts = folder.name.split('_')
            if len(parts) >= 3:
                clip_name = '_'.join(parts[1:-1])  # Exclure "clip" et le timestamp
                if clip_name not in duplicates:
                    duplicates[clip_name] = []
                duplicates[clip_name].append(folder)
    
    return duplicates

def clean_broll_storage():
    """Nettoyage principal de l'espace B-roll"""
    print("🧹 NETTOYAGE URGENT DE L'ESPACE DISQUE")
    print("=" * 50)
    
    broll_dir = Path("AI-B-roll/broll_library")
    if not broll_dir.exists():
        print("❌ Dossier AI-B-roll/broll_library introuvable")
        return
    
    # Analyser la taille initiale
    initial_size = get_folder_size(broll_dir)
    print(f"📊 Taille initiale: {initial_size:.2f} GB")
    
    # Analyser les doublons
    duplicates = analyze_broll_duplicates(broll_dir)
    
    total_cleaned = 0.0
    folders_removed = 0
    
    for clip_name, folders in duplicates.items():
        if len(folders) > 1:
            print(f"\n🔍 Clip '{clip_name}' trouvé {len(folders)} fois:")
            
            # Trier par date de modification (garder le plus récent)
            folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Garder le premier (plus récent), supprimer les autres
            keep_folder = folders[0]
            remove_folders = folders[1:]
            
            print(f"   ✅ Garder: {keep_folder.name}")
            
            for folder in remove_folders:
                folder_size = get_folder_size(folder)
                print(f"   🗑️ Supprimer: {folder.name} ({folder_size:.2f} GB)")
                
                try:
                    shutil.rmtree(folder)
                    total_cleaned += folder_size
                    folders_removed += 1
                    print(f"      ✅ Supprimé avec succès")
                except Exception as e:
                    print(f"      ❌ Erreur: {e}")
    
    # Supprimer les dossiers vides ou corrompus
    print(f"\n🔍 Nettoyage des dossiers vides/corrompus...")
    for folder in broll_dir.iterdir():
        if folder.is_dir():
            if not any(folder.rglob('*')):  # Dossier vide
                try:
                    shutil.rmtree(folder)
                    print(f"   🗑️ Dossier vide supprimé: {folder.name}")
                    folders_removed += 1
                except Exception as e:
                    print(f"   ❌ Erreur suppression {folder.name}: {e}")
    
    # Calcul final
    final_size = get_folder_size(broll_dir)
    space_freed = initial_size - final_size
    
    print(f"\n📊 RÉSULTATS DU NETTOYAGE:")
    print(f"   📁 Dossiers supprimés: {folders_removed}")
    print(f"   💾 Espace libéré: {space_freed:.2f} GB")
    print(f"   📊 Taille finale: {final_size:.2f} GB")
    print(f"   📈 Réduction: {(space_freed/initial_size)*100:.1f}%")
    
    if space_freed > 0:
        print(f"\n🎉 NETTOYAGE RÉUSSI ! {space_freed:.2f} GB libérés")
    else:
        print(f"\n⚠️ Aucun espace libéré")

def clean_temp_files():
    """Nettoie les fichiers temporaires"""
    print(f"\n🧹 Nettoyage des fichiers temporaires...")
    
    temp_dirs = [
        Path("temp"),
        Path("cache"),
        Path("output/locks"),
        Path("__pycache__"),
    ]
    
    total_freed = 0.0
    
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            size = get_folder_size(temp_dir)
            try:
                if temp_dir.name == "__pycache__":
                    # Supprimer récursivement tous les __pycache__
                    for pycache in Path(".").rglob("__pycache__"):
                        shutil.rmtree(pycache, ignore_errors=True)
                else:
                    shutil.rmtree(temp_dir)
                    temp_dir.mkdir(exist_ok=True)
                
                total_freed += size
                print(f"   ✅ {temp_dir}: {size:.2f} GB libérés")
            except Exception as e:
                print(f"   ❌ Erreur {temp_dir}: {e}")
    
    print(f"   💾 Total temp libéré: {total_freed:.2f} GB")

def analyze_disk_usage():
    """Analyse complète de l'utilisation disque"""
    print(f"\n📊 ANALYSE COMPLÈTE DE L'UTILISATION DISQUE")
    print("=" * 50)
    
    # Analyser chaque dossier principal
    main_dirs = [
        "AI-B-roll",
        "venv311", 
        "output",
        "cache",
        "temp",
        "music_freeCopyright",
        "clips",
        "assets"
    ]
    
    total_project_size = 0.0
    
    for dir_name in main_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            size = get_folder_size(dir_path)
            total_project_size += size
            print(f"   📁 {dir_name:<20}: {size:>8.2f} GB")
    
    print(f"   📊 TOTAL PROJET: {total_project_size:>13.2f} GB")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    if total_project_size > 5:
        print(f"   🚨 Projet très volumineux ({total_project_size:.1f}GB)")
        print(f"   🔧 Activez le nettoyage automatique")
        print(f"   📦 Considérez un cache externe pour les B-rolls")
    
    return total_project_size

if __name__ == "__main__":
    print("🎯 SCRIPT DE NETTOYAGE URGENT")
    print("Libération de l'espace disque du pipeline vidéo")
    
    # Analyser d'abord
    initial_project_size = analyze_disk_usage()
    
    # Confirmer le nettoyage
    response = input(f"\n❓ Procéder au nettoyage ? (y/N): ").strip().lower()
    
    if response == 'y':
        start_time = time.time()
        
        # Nettoyer les B-rolls dupliqués
        clean_broll_storage()
        
        # Nettoyer les fichiers temporaires
        clean_temp_files()
        
        # Analyser après nettoyage
        final_project_size = analyze_disk_usage()
        
        cleanup_time = time.time() - start_time
        total_freed = initial_project_size - final_project_size
        
        print(f"\n🏆 NETTOYAGE TERMINÉ EN {cleanup_time:.1f}s")
        print(f"💾 ESPACE TOTAL LIBÉRÉ: {total_freed:.2f} GB")
        
        if total_freed > 1.0:
            print(f"🎉 EXCELLENT ! Plus d'1 GB libéré")
        elif total_freed > 0.1:
            print(f"👍 BIEN ! Espace libéré avec succès")
        else:
            print(f"⚠️ Peu d'espace libéré, vérifiez manuellement")
    else:
        print(f"❌ Nettoyage annulé") 