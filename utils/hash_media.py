# -*- coding: utf-8 -*-
"""
Module utilitaire pour le hachage des fichiers mÃ©dia
"""

import hashlib
import os
from pathlib import Path
from typing import Optional


def hash_media(path: str) -> str:
    """
    GÃ©nÃ¨re un hash SHA-256 d'un fichier mÃ©dia
    
    Args:
        path: Chemin vers le fichier mÃ©dia
        
    Returns:
        Hash SHA-256 en hexadÃ©cimal
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        OSError: Si erreur de lecture
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouve: {path}")
            
        if file_path.is_dir() or file_path.name in {'.', ''}:
            raise ValueError(f"Chemin invalide pour hash_media: {path}")

        # Hash SHA-256
        sha256_hash = hashlib.sha256()
        
        # Lecture par blocs pour Ã©viter la mÃ©moire
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
                
        return sha256_hash.hexdigest()
        
    except Exception as e:
        print(f"?? Erreur lors du hachage de {path}: {e}")
        return ""


def hash_string(text: str) -> str:
    """
    GÃ©nÃ¨re un hash SHA-256 d'une chaÃ®ne de texte
    
    Args:
        text: Texte Ã  hasher
        
    Returns:
        Hash SHA-256 en hexadÃ©cimal
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_file_info(path: str) -> Optional[dict]:
    """
    RÃ©cupÃ¨re les informations d'un fichier mÃ©dia
    
    Args:
        path: Chemin vers le fichier
        
    Returns:
        Dictionnaire avec taille, hash et type MIME
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return None
            
        stat = file_path.stat()
        
        return {
            'path': str(file_path),
            'size': stat.st_size,
            'hash': hash_media(path),
            'modified': stat.st_mtime,
            'extension': file_path.suffix.lower()
        }
        
    except Exception as e:
        print(f"âš ï¸ Erreur lors de la rÃ©cupÃ©ration des infos de {path}: {e}")
        return None 

