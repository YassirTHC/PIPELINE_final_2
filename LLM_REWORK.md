# LLM REWORK

## Table des matières
- [Inventaire des prompts](#inventaire-des-prompts)
- [Schémas attendus et validation](#schemas-attendus-et-validation)
- [Stratégie de fallback & garde min chars](#strategie-de-fallback--garde-min-chars)
- [Normalisation des requêtes B-roll](#normalisation-des-requetes-b-roll)
- [Prompts réécrits proposés](#prompts-reecrits-proposes)
- [Tests unitaires recommandés](#tests-unitaires-recommandes)

## Inventaire des prompts
| Nom | Localisation | Usage | Notes |
|---|---|---|---|
| `SEGMENT_JSON_PROMPT` | `pipeline_core/llm_service.py` lignes 546-553.【F:pipeline_core/llm_service.py†L546-L553】 | Génération principale de `broll_keywords` & `queries` (mode JSON). | Interdit certains tokens génériques mais ne contraint pas le format JSON (pas de balises). |
| `_SCENE_PROMPT_MAP` | `pipeline_core/llm_service.py` lignes 1823-2202.【F:pipeline_core/llm_service.py†L1823-L2202】 | Prompts spécifiques par type de scène pour enrichir le contexte. | Dictionnaire statique rarement mis à jour, certains prompts trop verbeux. |
| `SEGMENT_PLAIN_PROMPT` (fallback texte) | Construit dynamiquement dans `_ollama_generate_text` quand JSON échoue (non nommé, fallback `short_prompt`).【F:pipeline_core/llm_service.py†L1492-L1509】【F:pipeline_core/llm_service.py†L1596-L1616】 | Demande un texte libre que le pipeline doit ensuite parser heuristiquement. | Source d'erreurs car aucun schéma, dépend de `PIPELINE_LLM_FALLBACK_TRUNC`. |
| Prompts heuristiques (keywords-first) | `_segment_brief_terms` & `_dedupe_queries` injectent du texte dans les prompts dynamiques.【F:video_processor.py†L1979-L1999】【F:video_processor.py†L1987-L1994】 | Ajustent la requête selon le contexte dynamique. | Pas explicitement testés. |

## Schémas attendus et validation
Réponse souhaitée pour `SEGMENT_JSON_PROMPT` :
```json
{
  "broll_keywords": ["noun phrase", ...],
  "queries": ["search query", ...]
}
```
Contraintes proposées :
- `broll_keywords` : 6–10 éléments, chacun 2–3 mots, lettres/accents autorisés ; rejeter tokens génériques (`thing`, `stuff`).【F:pipeline_core/llm_service.py†L546-L589】
- `queries` : 6–10 éléments, 2–4 mots, pas de caractères spéciaux ; normalisation minuscule + ASCII simplifié avant scoring.
- JSON strict (pas de texte hors objet) ; si parsing échoue → log explicite `event=llm_parse_error` et fallback.

### Validator proposé (extrait)
```python
from pydantic import BaseModel, Field, validator

class SegmentMetadata(BaseModel):
    broll_keywords: list[str] = Field(min_items=6, max_items=10)
    queries: list[str] = Field(min_items=6, max_items=10)

    @validator('broll_keywords', each_item=True)
    def _kw_format(cls, value: str) -> str:
        cleaned = ' '.join(value.split()).strip()
        if len(cleaned.split()) not in (2, 3):
            raise ValueError('keyword must contain 2-3 words')
        if cleaned.lower().split()[0] in {'thing', 'stuff', 'generic'}:
            raise ValueError('too generic')
        return cleaned
```
Ce modèle remplace les heuristiques manuelles `remove_blocklisted_tokens` et `_normalize_queries`. On déclenche une `ValidationError` que le service journalise puis transforme en `DynamicCompletionError`.

## Stratégie de fallback & garde min chars
Chemin actuel : stream → non-stream fallback `_ollama_generate_sync` si `PIPELINE_LLM_MIN_CHARS` non atteint ou si `PIPELINE_LLM_FORCE_NON_STREAM=1`.【F:pipeline_core/llm_service.py†L1486-L1516】

Actions recommandées :
1. **Journalisation structurée** – Ajouter `logger.warning("llm_fallback", extra={...})` avec `reason`, `model`, `attempts` avant de retourner le texte fallback.【F:pipeline_core/llm_service.py†L1513-L1669】
2. **Propagation** – Inclure `reason` dans l'objet résultat afin que `VideoProcessor` puisse décider de re-tenter ou d'abaisser la sévérité (ex : ignorer segments courts).【F:video_processor.py†L1979-L1999】
3. **Timeouts distincts** – Exposer `settings.llm_timeout_stream` et `settings.llm_timeout_fallback` pour éviter que le fallback réutilise la valeur stream (actuellement `timeout` identique).【F:pipeline_core/llm_service.py†L1475-L1485】
4. **Retry budget** – Limiter les tentatives streaming (`backoffs = (0.6, 1.2, 2.4)`) à `settings.llm_max_attempts`, configurable via `PIPELINE_LLM_MAX_ATTEMPTS`.【F:pipeline_core/llm_service.py†L1513-L1669】

## Normalisation des requêtes B-roll
Actuellement `_build_queries` prend les trois premiers mots-clés et ajoute quelques variantes sans normalisation phonétique. `_normalize_queries` (dans `video_processor.py`) fournit pourtant une pipeline plus robuste (lowercase, remove stopwords, synonymes).【F:pipeline_core/fetchers.py†L788-L800】【F:video_processor.py†L452-L499】

Plan :
1. Centraliser une fonction `settings.normalize_query(term: str)` appliquant :
   - suppression des accents (`unicodedata.normalize`),
   - filtrage stopwords multilingues,
   - lemmatisation légère (optionnelle via `nltk` ou simple mapping).
2. Fusionner `_augment_with_synonyms` pour éviter des doublons internes (utiliser `difflib.SequenceMatcher` > 0.85 pour merges).【F:pipeline_core/llm_service.py†L556-L589】
3. Détecter les répétitions temporelles : conserver un `deque` des requêtes utilisées dans les `N` dernières secondes et bloquer les duplicates (PR-3).

## Prompts réécrits proposés
```text
You are a video content planner. Return ONLY valid JSON with two arrays: "broll_keywords" and "queries".
Constraints:
- Each keyword: 2-3 words, concrete visuals the editor can film (no pronouns, no abstract nouns).
- Each query: <=4 words, suitable for stock providers (no "stock", "footage", "background").
- Respond in the target language: {language}.
Segment transcript (max 900 chars):
"""
{segment_excerpt}
"""
```
Différences majeures : ajout d'une consigne de langue (`PIPELINE_LLM_TARGET_LANG`) et limite stricte de caractères. Ajouter un champ `language` dans le payload pour exiger la traduction.【F:pipeline_core/llm_service.py†L119-L134】

## Tests unitaires recommandés
1. **Validation JSON** – Alimenter `_ollama_generate_text` avec réponses simulées (stream et fallback) et vérifier que le validator Pydantic rejette les payloads invalides (tokens génériques, arrays trop courts).【F:tests/test_llm_service_fallback.py†L80-L120】
2. **Min chars guard** – Test existant `test_llm_service_fallback` force `PIPELINE_LLM_MIN_CHARS=9999`; compléter pour vérifier la présence de l'événement `llm_fallback_reason`.【F:tests/test_llm_service_fallback.py†L80-L120】
3. **Query normalization** – Ajouter un test dans `tests/test_segment_queries.py` pour s'assurer que des termes quasi identiques (`"dopamine release"`, `"dopamine-release"`) ne génèrent qu'une requête unique en s'appuyant sur `_normalize_queries`.【F:video_processor.py†L452-L499】【F:tests/test_segment_queries.py†L60-L120】
4. **Fallback retries** – Simuler trois échecs streaming et vérifier que le service n'effectue pas plus de `settings.llm_max_attempts` appels avant fallback.

### Matrice des modes segmentaires

| Mode `llm_path` | Condition d'activation | Source | Notes |
| --- | --- | --- | --- |
| `segment_stream` | Streaming SSE actif et sain (`PIPELINE_LLM_FORCE_NON_STREAM=0`, aucun `stream_err`). | `_ollama_generate_text` | Bascule automatiquement vers `segment_blocking` après la première erreur SSE ou timeout.【F:pipeline_core/llm_service.py†L1425-L1704】 |
| `segment_blocking` | Forcé par `PIPELINE_LLM_FORCE_NON_STREAM=1` ou après une erreur de flux. | `_ollama_generate_sync` | Mémo interne : toutes les complétions suivantes restent bloquantes jusqu'à la fin du run.【F:pipeline_core/llm_service.py†L1425-L1704】 |
| `metadata_first` | `PIPELINE_DISABLE_DYNAMIC_SEGMENT_LLM=1` | Cache métadonnées global | Pas d'appel LLM segmentaire ; sélection 1–3 requêtes via similarité TF-IDF.【F:pipeline_core/llm_service.py†L3469-L3803】 |

Chaque appel `generate_hints_for_segment` journalise désormais la décision via `logger.info(..., extra={"llm_path": ..., "llm_path_reason": ...})` pour faciliter le suivi des bascules.【F:pipeline_core/llm_service.py†L3518-L3803】

### Exemple PowerShell

```powershell
$env:PIPELINE_DISABLE_DYNAMIC_SEGMENT_LLM = '1'
$env:PIPELINE_LLM_FORCE_NON_STREAM = '1'
python run_pipeline.py --video .\demo.mp4
```
