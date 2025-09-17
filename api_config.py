#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration des clés API pour le système B-roll
"""

import os

# Configurer les clés API directement
os.environ['PEXELS_API_KEY'] = 'pwhBa9K7fa9IQJCmfCy0NfHFWy8QyqoCkGnWLK3NC2SbDTtUeuhxpDoD'
os.environ['PIXABAY_API_KEY'] = '51724939-ee09a81ccfce0f5623df46a69'

# Configuration du fetching B-roll
os.environ['BROLL_FETCH_ENABLE'] = 'True'
os.environ['BROLL_FETCH_PROVIDER'] = 'pexels'
os.environ['BROLL_FETCH_ALLOW_VIDEOS'] = 'True'
os.environ['BROLL_FETCH_ALLOW_IMAGES'] = 'True'
os.environ['BROLL_FETCH_MAX_PER_KEYWORD'] = '25'

# Configuration du nettoyage
os.environ['BROLL_DELETE_AFTER_USE'] = 'True'
os.environ['BROLL_PURGE_AFTER_RUN'] = 'True'

print("✅ Configuration des clés API appliquée")
print(f"✅ PEXELS_API_KEY: {os.environ['PEXELS_API_KEY'][:8]}******")
print(f"✅ PIXABAY_API_KEY: {os.environ['PIXABAY_API_KEY'][:8]}******")
print(f"✅ BROLL_FETCH_ENABLE: {os.environ['BROLL_FETCH_ENABLE']}") 