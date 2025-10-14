# Pourquoi avons-nous ajouté des valeurs par défaut aux réglages LLM ?

## Ce qui a été fait (version vulgarisée)
- Nous avons mis des valeurs par défaut explicites sur quelques interrupteurs (ex. `disable_hashtags`) dans la fiche de configuration du modèle IA.
- Le but est que Python sache quelle valeur utiliser même si le fichier de configuration ou les variables d'environnement n'en fournissent pas.
- La même correction a été appliquée partout où la configuration est lue (module principal et version `src/...`) pour que l'appli en ligne de commande et les tests partagent exactement les mêmes paramètres.

## Pourquoi le test a échoué au départ ?
- Le test `tests/test_run_pipeline_env.py` charge toute la config via une `@dataclass`.
- Dans une dataclass, les champs sans valeur par défaut doivent apparaître **avant** ceux qui ont un défaut.
- Nous avions laissé certains interrupteurs sans valeur par défaut alors que les suivants en avaient une. Python a donc levé l'erreur :
  > `TypeError: non-default argument 'disable_hashtags' follows default argument`
- Résultat : Pytest s'est arrêté avant même d'exécuter le test parce que l'import de la config plantait.

## Ce qui change après la correction
- Chaque champ optionnel possède désormais une valeur par défaut claire (souvent `False` ou `None`).
- L'import du module de configuration se déroule sans erreur, donc les tests peuvent aller jusqu'au bout.
- Cela évite aussi d'autres surprises si un script externe importe la config sans définir toutes les variables.

## Comment vérifier que tout va bien
```bash
pytest tests/test_run_pipeline_env.py
```
- Ce test repasse maintenant car la dataclass est correctement structurée.
- Tu peux aussi lancer toute la série d'intégration :
```bash
pytest tests/test_llm_optional_integration.py \
       tests/test_llm_service_fallback.py \
       tests/test_run_pipeline_env.py
```

En résumé : on a simplement appris à Python quelles valeurs utiliser par défaut pour ces interrupteurs LLM, ce qui évite une erreur de construction de dataclass et laisse les tests faire leur boulot normalement.
