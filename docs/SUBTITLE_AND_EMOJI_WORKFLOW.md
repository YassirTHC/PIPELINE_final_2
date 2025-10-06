# Génération et personnalisation des sous-titres dynamiques

Ce document résume la façon dont le pipeline crée et anime les sous-titres, ajoute les émojis et comment vous pouvez personnaliser le rendu.

## 1. Génération des segments de sous-titres

1. Le module `video_processor_clean.py` utilise Whisper pour transcrire l'audio en segments synchronisés (`start`, `end`, `text`). Les mots individuels sont conservés pour les animations fines et l'ajustement de timing.【F:video_processor_clean.py†L1168-L1201】
2. Les segments générés sont stockés dans le dossier du clip (SRT/VTT) et réutilisés pour le burn-in final ou l'export brut.【F:video_processor_clean.py†L732-L865】

## 2. Stylisation et animation (Hormozi Subtitles)

La classe `HormoziSubtitles` gère toute la partie visuelle des sous-titres.【F:hormozi_subtitles.py†L1380-L1407】

### Palette intelligente et détection de mots-clés

* Un dictionnaire de couleurs associe des thèmes (finance, urgence, santé, etc.) à des codes hexadécimaux. Dès qu'un mot correspond, il est coloré automatiquement.【F:hormozi_subtitles.py†L120-L198】
* Les alias sont enrichis par un lexique externe, ce qui permet de couvrir davantage de variations de mots clés sans modifier le code principal.【F:hormozi_subtitles.py†L199-L214】

### Mise en page dynamique

* Chaque mot actif reçoit une animation de “bounce” contrôlée par `bounce_scale` et `animation_progress` pour donner un effet TikTok/Hormozi.【F:hormozi_subtitles.py†L838-L908】
* Une logique d'auto-redimensionnement garde la ligne de sous-titres sous ~92 % de la largeur vidéo pour éviter les débordements.【F:hormozi_subtitles.py†L872-L910】
* Le positionnement vertical, la marge de sécurité (`subtitle_safe_margin_px`) et la taille par défaut (`font_size`) sont configurables via `self.config` ou les paramètres passés à `add_hormozi_subtitles`.【F:hormozi_subtitles.py†L770-L837】

## 3. Gestion des émojis

### Sélection automatique

* Chaque catégorie de mots possède une liste d'émojis possibles. Le moteur choisit l'émoji le plus pertinent en fonction du mot clé dominant et alterne pour éviter les répétitions.【F:hormozi_subtitles.py†L216-L274】【F:hormozi_subtitles.py†L804-L833】

### Insertion dans la vidéo

* Les émojis peuvent être rendus soit à partir d'images PNG haute résolution, soit directement en Unicode si aucun PNG n'est disponible.【F:hormozi_subtitles.py†L275-L349】
* Des émojis “hero” supplémentaires peuvent être superposés dans un coin via `span_style_map`, pratique pour afficher un pictogramme plus grand sur certains mots clés.【F:hormozi_subtitles.py†L908-L955】

## 4. Comment personnaliser

### Paramètres rapides

Lors de l'appel `add_hormozi_subtitles`, vous pouvez surcharger la configuration :

```python
add_hormozi_subtitles(
    input_video_path,
    transcription_data,
    output_video_path,
    font_size=78,
    subtitle_safe_margin_px=180,
    enable_emojis=True,
    emoji_boost=1.3,
    render_subtitles=True,
)
```

Tout paramètre présent dans `self.default_config` (tailles, couleurs de fond, vitesse d'animation, activation des émojis, etc.) peut être modifié en argument ou via un dictionnaire dédié.【F:hormozi_subtitles.py†L704-L783】

### Couleurs et émojis

* Pour ajuster les palettes : éditez les mappings dans `keyword_colors` ou enrichissez le lexique externe référencé par `_bootstrap_categories()` pour couvrir de nouveaux termes.【F:hormozi_subtitles.py†L120-L214】
* Pour changer les émojis utilisés : modifiez les listes dans `category_emojis` ou fournissez vos propres PNG dans `assets/emojis/` (ou autres chemins reconnus).【F:hormozi_subtitles.py†L216-L349】

### Animation et rendu

* Les vitesses d'apparition, le rebond (`bounce_scale`) ou l'opacité des barres de fond sont réglables dans la config par défaut.【F:hormozi_subtitles.py†L704-L783】
* Le pipeline peut aussi désactiver totalement le burn-in pour ne fournir que les fichiers SRT/VTT en jouant sur `render_subtitles` dans `video_processor_clean.py`.【F:video_processor_clean.py†L1534-L1549】

## 5. Résumé

1. Whisper → segments synchronisés.
2. `HormoziSubtitles` → coloration intelligente, animation bounce, insertion d'émojis.
3. Configurations faciles à surcharger pour ajuster tailles, couleurs, animations et assets graphiques.

En modifiant ces points, vous contrôlez entièrement l'aspect et la dynamique des sous-titres, ainsi que la présence d'émojis.
