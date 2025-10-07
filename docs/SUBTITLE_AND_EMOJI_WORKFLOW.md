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

### Montserrat & options typées

* Les polices **Montserrat ExtraBold/Bold** sont embarquées dans `assets/fonts/` et utilisées en priorité lors du rendu. La résolution choisie est journalisée via `Settings.to_log_payload()` sous la clé `subtitles.font_path` pour garantir la traçabilité du style.【F:video_pipeline/config/settings.py†L217-L255】
* Les paramètres de contour et d'ombre (`stroke_px=6`, `shadow_opacity=0.35`, `shadow_offset=3`) sont fournis par défaut via `SubtitleSettings` ; ils peuvent être surchargés par variables d'environnement si besoin ponctuel, ce qui fixe le rendu Montserrat "viral" sans retoucher le code.【F:video_pipeline/config/settings.py†L560-L598】【F:hormozi_subtitles.py†L930-L1038】
* Le lot 1 expose `SubtitleSettings` (`font_path`, `font_size`, `subtitle_safe_margin_px`, `keyword_background`, `enable_emojis`) accessible via `get_settings().subtitles`. Ces valeurs alimentent `HormoziSubtitles` par défaut et peuvent être surchargées via `PIPELINE_SUB_*` dans l'environnement.【F:video_pipeline/config/settings.py†L137-L202】【F:video_processor_clean.py†L811-L818】
* Le wrapper `add_hormozi_subtitles` accepte désormais un `subtitle_settings` et un `font_path` explicites. Si vous avez besoin d'un style ponctuel, passez ces arguments ; sinon, le pipeline applique automatiquement la configuration typée lors du burn-in final.【F:hormozi_subtitles.py†L1380-L1408】

### Couleurs et émojis

* Pour ajuster les palettes : éditez les mappings dans `keyword_colors` ou enrichissez le lexique externe référencé par `_bootstrap_categories()` pour couvrir de nouveaux termes.【F:hormozi_subtitles.py†L120-L214】
* Pour changer les émojis utilisés : modifiez les listes dans `category_emojis` ou fournissez vos propres PNG dans `assets/emojis/` (ou autres chemins reconnus).【F:hormozi_subtitles.py†L216-L349】
* La densité et la répartition sont gouvernées par `SubtitleSettings.emoji_target_per_10`, `emoji_min_gap_groups` et `emoji_max_per_segment`, ce qui maintient ~5 émojis pour 10 groupes tout en imposant un écart minimal. Un fallback neutre (`emoji_no_context_fallback`) peut être fixé pour conserver un pictogramme discret lorsque le contexte est insuffisant.【F:video_pipeline/config/settings.py†L560-L598】【F:hormozi_subtitles.py†L866-L955】

### Animation et rendu

* Les vitesses d'apparition, le rebond (`bounce_scale`) ou l'opacité des barres de fond sont réglables dans la config par défaut.【F:hormozi_subtitles.py†L704-L783】
* Le pipeline peut aussi désactiver totalement le burn-in pour ne fournir que les fichiers SRT/VTT en jouant sur `render_subtitles` dans `video_processor_clean.py`.【F:video_processor_clean.py†L1534-L1549】

## 5. Résumé

1. Whisper → segments synchronisés.
2. `HormoziSubtitles` → coloration intelligente, animation bounce, insertion d'émojis.
3. Configurations faciles à surcharger pour ajuster tailles, couleurs, animations et assets graphiques.

## 6. Pilotage multi-provider du LLM

* `LLMSettings` conserve désormais le champ `provider` (ollama, lmstudio, openai, etc.) en plus des modèles texte/JSON ; les overrides CLI `--llm-provider`, `--llm-model-text` et `--llm-model-json` appliquent une nouvelle config en mémoire et mettent à jour l'environnement avant le lancement du pipeline.【F:video_pipeline/config/settings.py†L84-L227】【F:run_pipeline.py†L365-L419】【F:video_processor.py†L6678-L6705】
* Le log `[CONFIG]` n'est émis qu'une seule fois à l'initialisation. Si un provider tombe en échec, la stratégie `pipeline_core.llm_service` conserve les paramètres de timeout (`timeout_fallback_s`, `num_predict`) pour relancer automatiquement la génération via le modèle fallback JSON/texte configuré.【F:video_pipeline/config/settings.py†L660-L705】【F:pipeline_core/llm_service.py†L4250-L4314】

En modifiant ces points, vous contrôlez entièrement l'aspect et la dynamique des sous-titres, ainsi que la présence d'émojis.
