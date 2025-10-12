# Mode B-roll piloté par le LLM

## Problème constaté
- Les requêtes segmentées mélangeaient des indices génériques venant des analyseurs (`desk journaling at`, `runner tying shoes`, etc.).
- Ces requêtes étaient injectées même lorsque le LLM fournissait déjà de bons mots-clés, ce qui introduisait des plans incohérents dans les B-rolls.

## Ce qui a été implémenté
- Ajout d'un drapeau d'environnement `PIPELINE_BROLL_LLM_ONLY` lu au démarrage du processeur vidéo. 【F:video_processor.py†L742-L770】
- Lorsque ce drapeau est actif, chaque segment garde uniquement les requêtes proposées par le LLM (`llm_queries`). Les requêtes sont normalisées et limitées à la capacité prévue pour le segment. 【F:video_processor.py†L1179-L1213】
- Si le LLM ne renvoie rien pour un segment, on retombe automatiquement sur les sources contextuelles (briefs, mots-clés détectés, transcript) pour éviter les trous dans la timeline. 【F:video_processor.py†L1215-L1256】

## Impact sur la qualité des B-rolls
- Les segments utilisent en priorité la vision du LLM, ce qui renforce la cohérence avec le discours et réduit les plans génériques répétés.
- Les tests valident que les requêtes issues du LLM sont conservées en mode "LLM only" et que les fallbacks contextuels restent disponibles si besoin. 【F:tests/test_segment_queries.py†L241-L283】

## Comment l'activer
1. Définir la variable d’environnement `PIPELINE_BROLL_LLM_ONLY=1` avant de lancer la pipeline.
2. Relancer `run_pipeline.py` : les logs `DEBUG _combine_broll_queries` montreront uniquement les requêtes normalisées par le LLM tant qu’il en fournit.

Ce mode vous permet donc d’aligner la sélection B-roll avec les mots-clés générés par le modèle tout en conservant une sécurité en cas de réponse vide.
