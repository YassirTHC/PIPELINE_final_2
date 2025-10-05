# CLEANUP PLAN

## Table des matières
- [Suppressions / Archivage](#suppressions--archivage)
- [Refactors groupés](#refactors-groupes)
- [Politique de compatibilité](#politique-de-compatibilite)

## Suppressions / Archivage
| Élément | Action | Justification | Référence |
|---|---|---|---|
| `video_processor_backup.py` & variantes (`*_backup`, `.pre_fix`, etc.) | Déplacer dans `archive/` ou supprimer après extraction des commits pertinents. | Fichiers énormes non importés qui encombrent le PYTHONPATH et dupliquent la logique courante. | `video_processor_backup.py` lignes 1-40.【F:video_processor_backup.py†L1-L40】 |
| Scripts prototypes (`advanced_broll_pipeline.py`, `pipeline_qwen3_8b_unifie.py`, `pipeline_hybride_robuste.py`) | Archiver dans `docs/legacy/` avec note « non maintenu ». | Surcharges du pipeline principal, aucune importation active. | Inspection du dossier racine (absence de références). |
| Batchs `.bat` multiples (`lancer_interface*.bat`) | Conserver seulement les variantes supportées, documenter le chemin officiel (`run_pipeline.py`). | Entretien Windows difficile, certaines scripts pointeront vers anciennes configs. | `lancer_interface_final.bat` etc. |
| `setup.py` | Retirer après migration `pyproject.toml` (PR-1). | Script incompatible Windows (emojis) et non conforme packaging moderne. | `setup.py` lignes 1-34.【F:setup.py†L1-L34】 |

## Refactors groupés
| Ordre | Bloc | Travaux | Impact | Références |
|---|---|---|---|---|
| 1 | Packaging | Créer `pyproject.toml`, supprimer `setup.py`, déplacer scripts de création dossiers dans module `tools/bootstrap.py`. | Moyen : modifie installation, requis pour Windows. | `setup.py` lignes 1-34.【F:setup.py†L1-L34】 |
| 2 | Configuration centralisée | Introduire `pipeline_core/settings.py` (dataclass) pour lire toutes les env (`PIPELINE_LLM_*`, `PIPELINE_BROLL_*`, `BROLL_FETCH_*`). Remplacer accès directs dans `llm_service`, `fetchers`, `video_processor`. | Élevé : réduit divergences, nécessaire pour PR-2/PR-3/PR-4. | 【F:pipeline_core/llm_service.py†L1486-L1520】【F:pipeline_core/configuration.py†L320-L431】【F:video_processor.py†L423-L447】 |
| 3 | B-roll rules | Étendre `enforce_broll_schedule_rules` (min start, gap configurable, anti-repeat par url/id/keyword), intégrer `settings`. Ajouter journaux JSON via `log_broll_decision`. | Critique : alignement besoin métier hook/anti-repeat. | 【F:video_processor.py†L423-L447】【F:config.py†L218-L228】 |
| 4 | LLM service | Extraire la logique `sys.path` dans package, ajouter instrumentation fallback (raison, modèle, temps). Ajouter validation JSON stricte (pydantic) pour `SEGMENT_JSON_PROMPT`. | Élevé : fiabilise métadonnées et diagnostics. | 【F:pipeline_core/llm_service.py†L23-L38】【F:pipeline_core/llm_service.py†L546-L618】 |
| 5 | Fetchers | Normaliser queries (lemmatisation simple, dedupe fuzzy), partager throttling, remplacer `print` par logger, exposer circuits-courts sur timeouts. | Moyen : réduit quotas, logs propres. | 【F:pipeline_core/fetchers.py†L200-L520】【F:pipeline_core/fetchers.py†L724-L733】【F:pipeline_core/fetchers.py†L788-L800】 |
| 6 | Tests | Stabiliser `pytest` (conftest: désactiver auto-load, fixtures pour LLM/B-roll), ajouter tests unitaires pour settings, anti-repeat, min gap, fallback LLM. | Moyen : garantit non-régression. | 【F:tests/conftest.py†L1-L13】【F:tests/test_llm_service_fallback.py†L80-L120】 |
| 7 | Logs & monitoring | Uniformiser `JsonlLogger` usage, supprimer `print` résiduels, ajouter rotation/compaction optionnelle. | Faible : hygiène ops. | 【F:pipeline_core/logging.py†L34-L118】 |

## Politique de compatibilité
- **CLI** : conserver `run_pipeline.py` comme point d'entrée officiel ; fournir wrapper Windows (PowerShell) mis à jour qui appelle `python -m video_pipeline` après packaging. |Réf : `run_pipeline.py` lignes 1-200.【F:run_pipeline.py†L1-L200】|
- **ENV** : maintenir compatibilité ascendante des noms (`PIPELINE_DISABLE_TFIDF_FALLBACK` -> warning) tout en promouvant les nouveaux (`PIPELINE_TFIDF_FALLBACK_DISABLED`).【F:pipeline_core/configuration.py†L88-L118】
- **Feature flags** : lors de l'introduction des nouvelles règles B-roll, prévoir flags `PIPELINE_BROLL_MIN_START_SECONDS`, `PIPELINE_BROLL_MIN_GAP_SECONDS`, `PIPELINE_BROLL_NO_REPEAT_SECONDS` avec défauts sûrs et logs au démarrage.
- **Fallback legacy** : `Config.ENABLE_LEGACY_PIPELINE_FALLBACK` doit rester adressable tant que les workflows historiques existent.【F:config.py†L24-L39】【F:video_processor.py†L1321-L1339】
