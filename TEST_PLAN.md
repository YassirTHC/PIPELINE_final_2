# TEST PLAN

## Table des matières
- [Stabilisation de Pytest](#stabilisation-de-pytest)
- [Couverture fonctionnelle ciblée](#couverture-fonctionnelle-ciblee)
- [Tests de non-régression Windows](#tests-de-non-regression-windows)
- [Commande de synthèse](#commande-de-synthese)

## Stabilisation de Pytest
1. **Désactiver l'autoload de plugins tiers** – Ajouter un fichier `pytest.ini` avec `addopts = -p no:faulthandler -p no:randomly` et définir `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` dans `tests/conftest.py` (compléter l'initialisation actuelle).【F:tests/conftest.py†L1-L13】
2. **Capture fiable** – Utiliser `capsys` / `caplog` plutôt que `redirect_stdout`. Corriger les tests qui invoquent `print` via `FetcherOrchestrator` (remplacé par logs).【F:pipeline_core/fetchers.py†L724-L733】
3. **Fermeture de fichiers** – Remplacer `NamedTemporaryFile(delete=False)` + fermeture manuelle dans `tests/test_no_repeat_assets.py` par `TemporaryDirectory` + `Path.write_bytes` pour éviter `ValueError: I/O operation on closed file`.【F:tests/test_no_repeat_assets.py†L223-L287】
4. **Fixtures centralisées** – Créer `tests/fixtures.py` fournissant :
   - `llm_stub` (réponse JSON valide, vérifie min chars),
   - `fetcher_stub` (retourne B-roll unique avec identifiant).
   Réutiliser dans `tests/test_llm_service_fallback.py` et `tests/test_video_processor.py`.【F:tests/test_llm_service_fallback.py†L80-L120】【F:tests/test_video_processor.py†L225-L300】

## Couverture fonctionnelle ciblée
| Module | Cas à couvrir | Fichier test |
|---|---|---|
| `pipeline_core.settings` (nouveau) | Conversion bool/int/float, defaults, logs de démarrage. | `tests/test_settings.py` (à créer). |
| `pipeline_core/llm_service` | - Stream -> fallback min chars<br>- JSON invalide → `DynamicCompletionError` (validator).<br>- Temps de réponse > timeout → `requests.Timeout`. | `tests/test_llm_service_fallback.py` (complété).【F:tests/test_llm_service_fallback.py†L80-L120】 |
| `video_processor.enforce_broll_schedule_rules` | - Rejet si start < `min_start` (nouvelle règle).<br>- Rejet si gap < `min_gap`.<br>- Rejet si `identifier` déjà utilisé dans fenêtre `no_repeat`. | `tests/test_broll_rules.py` (nouveau).【F:video_processor.py†L423-L447】 |
| `pipeline_core/fetchers` | - Normalisation queries (synonymes, stopwords).<br>- Timeout provider → fallback Pixabay unique.<br>- Logger JSON au lieu de `print`. | `tests/test_fetchers.py` (nouveau).【F:pipeline_core/fetchers.py†L200-L520】【F:pipeline_core/fetchers.py†L724-L733】 |
| `JsonlLogger` | Concurrence (threads) et écriture UTF-8 (caractères accentués). | `tests/test_logging.py` (nouveau).【F:pipeline_core/logging.py†L34-L118】 |

## Tests de non-régression Windows
1. **Chemins backslash** – Cas de test sur `Settings` pour vérifier que `Path("clips")` converti via `resolve()` ne duplique pas les séparateurs. (Créer un test paramétré `WindowsPath` via `pytest.mark.parametrize`).【F:config.py†L12-L39】
2. **Encodage UTF-8** – Exécuter `python -m compileall` sur les modules LLM/B-roll dans CI Windows pour vérifier absence de BOM ; `run_pipeline.py` reconfigure stdout (maintenir).【F:run_pipeline.py†L40-L48】
3. **Setup packaging** – Test d'installation locale : `pip install .` depuis PowerShell (CI) en utilisant le nouveau `pyproject.toml` ; vérifier qu'aucun emoji n'apparaît (PR-1).

## Commande de synthèse
```bash
# POSIX
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pip install -e .[dev] && pytest -q --disable-warnings

# Windows PowerShell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pip install -e .[dev]
pytest -q --disable-warnings
```
Ces commandes valident la configuration centralisée, les règles B-roll, la robustesse LLM et la compatibilité cross-OS.
