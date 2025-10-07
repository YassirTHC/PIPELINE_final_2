# Audit du style Hormozi 1 et de la logique emojis

Ce rapport résume, à partir du code actuel (`hormozi_subtitles.py`) et de la documentation
`SUBTITLE_AND_EMOJI_WORKFLOW.md`, dans quelle mesure le rendu correspond au style Hormozi 1
célèbre sur TikTok, et si la logique d'emojis est cohérente.

## 1. Fidélité au style Hormozi 1

### Points conformes

* **Groupes de 2–3 mots avec bounce** – `parse_transcription_to_word_groups` regroupe
  les tokens par deux en appliquant un effet bounce/zoom sur l'apparition, ce qui
  rappelle bien les animations rapides des montages Hormozi.【F:hormozi_subtitles.py†L369-L454】【F:hormozi_subtitles.py†L812-L878】
* **Texte en majuscule, police Impact/Bold et contour noir** – les mots sont
  normalisés en majuscules, la classe cherche une police Impact/Liberation Bold,
  et applique un contour noir épais lors du rendu, ce qui rapproche le look des
  sous-titres viraux.【F:hormozi_subtitles.py†L339-L411】【F:hormozi_subtitles.py†L584-L612】【F:hormozi_subtitles.py†L878-L1008】
* **Position basse avec marge de sécurité** – la logique maintient les sous-titres
  centrés, en bas de l'image, avec une marge (~200 px) proche des standards Hormozi.【F:hormozi_subtitles.py†L776-L858】

### Mise à niveau Montserrat "viral"

* **Palette resserrée et stable** – la table `category_colors` aligne désormais les
  thèmes principaux (finance, business, sales, content, sports…) sur six couleurs
  haute-contraste (#FFD700, #00E5FF, #FF1493, #32CD32, #FF8C00, #8A2BE2) directement
  appliquées sur le texte.【F:hormozi_subtitles.py†L101-L164】
* **Texte coloré sans blocs** – `keyword_background` est désactivé par défaut et la
  coloration passe par le remplissage du mot lui-même, avec un stroke noir de 6 px
  et un drop shadow léger pour garder la lisibilité sans cartons opaques.【F:hormozi_subtitles.py†L64-L142】【F:hormozi_subtitles.py†L930-L1038】
* **Police Montserrat ExtraBold forcée** – la résolution de police préfère les
  fichiers Montserrat embarqués ; le chemin retenu est journalisé via `Settings` et
  un log `[Subtitles]` est émis à la première utilisation.【F:video_pipeline/config/settings.py†L170-L229】【F:hormozi_subtitles.py†L251-L308】
* **Contour et ombre standardisés** – `SubtitleSettings` fixe `stroke_px=6`, `shadow_opacity=0.35` et `shadow_offset=3` pour stabiliser le rendu ; des overrides d'environnement permettent de les modifier sans toucher au code si un preset différent est requis.【F:video_pipeline/config/settings.py†L560-L598】【F:hormozi_subtitles.py†L930-L1038】
* **Auto-layout stabilisé** – la largeur est rescannée pour rester sous 92 % de la
  vidéo et l'animation conserve un Y lissé pour éviter les sauts, même avec le
  rebond.【F:hormozi_subtitles.py†L820-L912】

> **Conclusion** : le rendu colle désormais à la tendance "viral Montserrat" : mots
> tout en capitales, couleurs punchy directement sur la lettre, stroke noir massif et
> drop shadow subtil, le tout centré en bas avec une marge sécurisée configurable.

## 2. Qualité du mapping et du rendu des emojis

### Points positifs

* **Catégories riches** – `category_emojis` propose une bonne variété par thèmes
  (finance, action, émotions, etc.) avec rotation pour limiter les répétitions.
  L'intensité est prise en compte via la ponctuation et certains mots clés.【F:hormozi_subtitles.py†L188-L256】【F:hormozi_subtitles.py†L768-L838】
* **Évitement des doublons immédiats** – `_choose_emoji_for_tokens` mémorise le dernier
  emoji pour éviter de ressortir exactement le même symbole sur deux groupes
  consécutifs.【F:hormozi_subtitles.py†L768-L824】【F:hormozi_subtitles.py†L824-L838】
* **Fallback PNG/Unicode** – lorsqu'aucun PNG Twemoji n'est disponible, le pipeline
  bascule sur un rendu Unicode en police couleur, évitant un trou dans l'affichage.【F:hormozi_subtitles.py†L612-L652】【F:hormozi_subtitles.py†L1008-L1040】

### Logique emoji revisitée

* **Mapping exhaustif** – toutes les catégories renseignées dans
  `_bootstrap_categories` disposent d'un nuancier et d'une liste d'emojis, alias
  compris. Le fallback générique disparaît au profit d'un retour vide lorsque le
  contexte est absent.【F:hormozi_subtitles.py†L166-L254】【F:hormozi_subtitles.py†L264-L342】
* **Densité contrôlée** – `_plan_emojis_for_segment` distribue environ 4 à 5 emojis
  pour 10 groupes, impose un écart minimal configuré et empêche la répétition dans la
  fenêtre des 4 derniers groupes.【F:hormozi_subtitles.py†L866-L950】【F:tests/test_emojis_density_and_mapping.py†L18-L53】
* **Hero triggers** – des déclencheurs dédiés (🔥 + "offer", ⚡ + "energy", 💰 +
  "profit") permettent de placer un emoji "hero" par segment, option activable via la
  config typée.【F:hormozi_subtitles.py†L220-L247】【F:hormozi_subtitles.py†L866-L950】
* **Placement attaché au mot** – les emojis se superposent désormais dans l'angle
  supérieur droit du groupe coloré, plutôt qu'en suffixe flottant, ce qui renforce la
  lecture verticale sans casser le centrage.【F:hormozi_subtitles.py†L930-L1038】
* **Fallback neutre optionnel** – `SubtitleSettings.emoji_no_context_fallback` autorise un pictogramme de repli (ex. ⭐) lorsque la sélection automatique ne trouve rien, tout en conservant la densité cible via `emoji_target_per_10`, `emoji_min_gap_groups` et `emoji_max_per_segment`.【F:video_pipeline/config/settings.py†L560-L598】【F:hormozi_subtitles.py†L866-L955】

> **Conclusion** : la narration emoji gagne en cohérence (pas de 💼 hors sujet), en
> rythme et en densité, tout en restant maîtrisée grâce au seuil cible paramétrable.

## 3. Recommandations rapides

1. **Élargir les alias sémantiques** : poursuivre l'enrichissement de
   `_bootstrap_categories`/`emoji_alias` pour couvrir de nouveaux vocables marketing
   et francophones.
2. **Documenter les presets** : exposer dans la doc produit des presets prêts à l'emploi
   combinant tailles, stroke et drop shadow pour différents formats (shorts, 9:16, 1:1).
3. **Surveiller les déclencheurs hero** : affiner la liste `_hero_triggers` au gré des
   retours créatifs afin d'éviter toute saturation visuelle.
4. **Optimiser le cache emoji** : prévoir un nettoyage périodique du dossier
   `emoji_assets/` afin de limiter la taille disque lorsque de nouveaux emojis sont
   téléchargés.
5. **Outiller la QA** : conserver une suite de tests dédiés (stroke, densité emojis,
   fallback vide) pour prévenir toute régression lors d'évolutions futures.

Ces ajustements rapprocheraient nettement le rendu du style Hormozi 1 viral et
rendraient les emojis plus cohérents avec le contenu des sous-titres.

## 4. Résolu / Mises à jour

* **Police Montserrat prioritaire** – les fichiers `Montserrat-ExtraBold.ttf` et `Montserrat-Bold.ttf` sont embarqués et chargés avant les polices système. Le chemin effectif est résolu dans la config typée (`SubtitleSettings.font_path`) et consigné dans le log startup.【F:MANIFEST.in†L1-L1】【F:video_pipeline/config/settings.py†L162-L202】
* **Config typée appliquée au burn-in** – `video_processor_clean` transmet désormais `get_settings().subtitles` au wrapper `add_hormozi_subtitles`, garantissant que marge, taille, fonds de mots-clés et émojis suivent la configuration centralisée.【F:video_processor_clean.py†L811-L818】
* **Palette stabilisée + remplissage texte** – `self.category_colors` couvre explicitement `finance`, `sales`, `content`, `mobile`, `sports` et synonymes, avec application directe sur la lettre et stroke configurable.【F:hormozi_subtitles.py†L166-L254】【F:tests/test_subtitles_montserrat_fill.py†L9-L37】
* **Mapping emoji cohérent** – `category_emojis` fournit une liste pour chaque catégorie, les alias héritent de la même base et la planification applique l'anti-repeat fenêtre de 4 groupes ou retourne `""` en l'absence de contexte.【F:hormozi_subtitles.py†L866-L950】【F:tests/test_emojis_density_and_mapping.py†L18-L53】
* **Multi-provider LLM** – `LLMSettings` expose désormais le champ `provider` et les overrides CLI (`--llm-provider`, `--llm-model-text`, `--llm-model-json`) mettent à jour la configuration typée ainsi que les variables d'environnement avant l'exécution, tout en ne loguant `[CONFIG]` qu'une seule fois par process.【F:video_pipeline/config/settings.py†L84-L227】【F:run_pipeline.py†L365-L419】【F:video_processor.py†L6678-L6705】
