# Explication simplifiée de la correction PyCaps

## Ce que j'ai constaté
- Le code essayait d'importer `pycaps.pipeline.JsonConfigLoader`.
- Dans la version 0.2.0 de PyCaps livrée sur PyPI, ce sous-module n'existe pas.
- Résultat : n'importe quelle erreur d'import était traduite en « PyCaps n'est pas installé »,
  même lorsque la bibliothèque était bien présente mais simplement structurée autrement.

## Ce que j'ai compris
- Le loader devait pouvoir trouver la classe `JsonConfigLoader` peu importe l'organisation
  interne du paquet PyCaps.
- Il fallait distinguer deux situations :
  1. PyCaps totalement absent de l'environnement Python.
  2. PyCaps installé mais avec une API différente (autre emplacement de `JsonConfigLoader`).

## Ce que j'ai fait
- Ajout d'une fonction `_load_pycaps_loader()` plus robuste :
  - Tentative d'importation via `pycaps.pipeline` (layout historique).
  - Fallback vers `pycaps.JsonConfigLoader` si la classe est exposée à la racine.
  - Exploration automatique des sous-modules `pycaps.*` pour trouver la classe lorsque
    les deux layouts précédents échouent.
- Journalisation (`logger.info`) du layout retenu pour faciliter le diagnostic sur le terrain.
- Messages d'erreur revus pour préciser si PyCaps est absent ou si l'API est incompatible,
  afin de guider directement l'utilisateur vers la bonne action (installation ou mise à jour).
- Lors du rendu, même logique d'erreur : on signale clairement l'absence de PyCaps
  ou une erreur interne du moteur au lieu d'un message générique.

## Résultat
- Le pipeline détecte maintenant PyCaps sur les différentes distributions existantes
  (PyPI, GitHub, installations locales) sans faux positifs.
- Les utilisateurs comprennent immédiatement si PyCaps manque réellement ou si la version
  installée doit être mise à jour.
