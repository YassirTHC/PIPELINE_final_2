#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration des clés API pour le système B-roll
"""

import os

# Configuration des clés API via variables d'environnement (sécurisé)
# Définir ces variables dans votre système ou fichier .env
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
PIXABAY_API_KEY = os.getenv('PIXABAY_API_KEY')

if PEXELS_API_KEY:
    os.environ['PEXELS_API_KEY'] = PEXELS_API_KEY
if PIXABAY_API_KEY:
    os.environ['PIXABAY_API_KEY'] = PIXABAY_API_KEY

# Configuration du fetching B-roll
os.environ['BROLL_FETCH_ENABLE'] = 'True'
os.environ['BROLL_FETCH_PROVIDER'] = 'pexels'
os.environ['BROLL_FETCH_ALLOW_VIDEOS'] = 'True'
os.environ['BROLL_FETCH_ALLOW_IMAGES'] = 'False'
os.environ['BROLL_FETCH_MAX_PER_KEYWORD'] = '8'

# Configuration du nettoyage
os.environ['BROLL_DELETE_AFTER_USE'] = 'True'
os.environ['BROLL_PURGE_AFTER_RUN'] = 'True'

print(" Configuration des clés API appliquée")
if PEXELS_API_KEY:
    print(f" PEXELS_API_KEY: {PEXELS_API_KEY[:8]}******")
else:
    print("  PEXELS_API_KEY non définie - définir la variable d'environnement")
if PIXABAY_API_KEY:
    print(f" PIXABAY_API_KEY: {PIXABAY_API_KEY[:8]}******")
else:
    print("  PIXABAY_API_KEY non définie - définir la variable d'environnement")
print(f" BROLL_FETCH_ENABLE: {os.environ['BROLL_FETCH_ENABLE']}")
