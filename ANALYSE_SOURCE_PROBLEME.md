# Analyse de l'origine du problème de requêtes B-roll

Les requêtes LLM générées dans `video_processor.py` ne sont pas écrasées par `_merge_segment_query_sources`. La fusion conserve la priorité aux requêtes LLM dès qu'elles sont disponibles. Le véritable problème vient en amont : la variable `base_llm_queries` est souvent vide parce que le module de métadonnées ne retourne pas d'état `llm_status="ok"`.

## Chaîne d'exécution
1. Dans la boucle de segments, les requêtes de base proviennent de `self._latest_metadata` lorsque `llm_status == 'ok'`. Sinon, le pipeline bascule vers des fallbacks génériques (contexte dynamique, mots-clés de secours, transcript). 【F:video_processor.py†L2093-L2145】
2. Lorsque le service LLM échoue ou est indisponible, `generate_caption_and_hashtags` active `_run_fallback` ou `_run_heuristics`, ce qui enregistre un `llm_status` égal à `"fallback"` ou `"heuristic"` au lieu de `"ok"`. 【F:video_processor.py†L4561-L4622】【F:video_processor.py†L4624-L4671】
3. Tant que cet état persiste, `_merge_segment_query_sources` ne reçoit que des requêtes issues des fallbacks précédents et ne peut donc pas propager de "bonnes" requêtes LLM. 【F:video_processor.py†L1126-L1212】

## Conclusion
Le contournement consistant à commenter `_merge_segment_query_sources` masque le symptôme sans résoudre la cause. Il faut restaurer la génération de métadonnées LLM jusqu'à ce que `llm_status` redevienne `"ok"` (service opérationnel, réponses JSON valides). À partir de là, les requêtes LLM sont transmises correctement et `_merge_segment_query_sources` n'écrase pas les valeurs fournies par le modèle.

## Commande de vérification end-to-end
Pour valider que l'intégration LLM et les dépendances optionnelles fonctionnent de bout en bout, exécutez la batterie de tests ciblée suivante depuis la racine du projet :

```bash
pytest tests/test_llm_optional_integration.py tests/test_llm_service_fallback.py tests/test_run_pipeline_env.py
```

Une commande équivalente est désormais encapsulée dans l'outil utilitaire suivant, qui permet aussi de lister la commande exacte ou d'ajouter des arguments Pytest personnalisés :

```bash
python -m tools.llm_env_check --list   # afficher la commande
python -m tools.llm_env_check -- -k optional  # exécuter avec des filtres Pytest
```

Cette commande rejoue les cas où `utils.pipeline_integration` est indisponible, vérifie le repli du service LLM et s'assure que le bootstrap du pipeline n'échoue pas à cause d'importations facultatives manquantes. Si tout passe, vous pouvez considérer que la régression d'import est résolue.
