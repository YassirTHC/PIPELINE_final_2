# Quickstart (Smoke Test)

Ce guide verifie que le pipeline fonctionne avec une video de demonstration et confirme la generation des sorties cles.

## 1. Preparer l'environnement
```powershell
# Depuis la racine du depot
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copier les fichiers d'exemple :
```powershell
Copy-Item .env.example .env
Copy-Item config\pipeline.yaml.example config\pipeline.yaml
```

## 2. Lancer le pipeline sur un clip de test
```powershell
# Exemple avec un clip court present dans clips/
python .\run_pipeline.py --video .\clips\demo.mp4 --verbose
```

Options utiles :
- `--legacy` : utilise les selecteurs historiques (desactive l'orchestrateur pipeline_core).
- `--verbose` : logs detaillees dans la console + fichiers JSONL.

## 3. Verifier les artefacts generes
- Video finale : `output/final/final_<timestamp>.mp4`
- Journal JSONL : `output/meta/broll_pipeline_events.jsonl`
- Logs detaillees : `logs/pipeline_core/*.log`

## 4. Nettoyer (optionnel)
```powershell
Remove-Item output -Recurse -Force
Remove-Item logs -Recurse -Force
```

## Depannage rapide
- Verifier que ffmpeg est accessible via `ffmpeg -version`.
- S'assurer que les cles API Pexels/Pixabay sont valides lorsque le fetcher est actif.
- Utiliser `--legacy` si vous devez comparer avec l'ancien comportement.