# Résoudre l'erreur « untracked working tree files would be overwritten by merge »

## Pourquoi Git bloque `git pull`

Lorsque tu lances `git pull`, Git doit d'abord fusionner les fichiers de la branche distante dans ton dossier de travail. S'il détecte qu'un fichier **non suivi** (`untracked`) existe localement mais qu'il va aussi être créé ou modifié par le `pull`, il stoppe immédiatement l'opération pour ne pas écraser ce fichier sans ton accord.

Dans ton cas, `.pip_freeze_311.txt` est présent dans ton dossier mais Git ne le connaît pas encore (il n'a jamais été `add`). Sur la branche distante, ce même fichier existe déjà. Git refuse donc de continuer et affiche :

```
error: The following untracked working tree files would be overwritten by merge:
        .pip_freeze_311.txt
Please move or remove them before you merge.
Aborting
```

## Pourquoi `git restore --staged` ne marche pas ici

La commande `git restore --staged` sert uniquement à retirer un fichier de l'index (zone de staging). Comme `.pip_freeze_311.txt` n'a jamais été ajouté (`git add`), Git te répond logiquement qu'il ne trouve pas ce chemin dans l'index.

## Comment débloquer la situation

Tu as trois possibilités. Choisis celle qui correspond à ce que tu veux faire du fichier local :

### 1. Tu veux conserver ta version locale et l'ajouter à l'historique

```bash
git add .pip_freeze_311.txt
git commit -m "Capture locale de pip freeze"
git pull --ff-only
```

### 2. Tu veux simplement mettre de côté ta version locale sans la committer

```bash
git stash push --include-untracked --message "Sauvegarde pip freeze"
git pull --ff-only
git stash pop   # si tu veux récupérer ta version ensuite
```

### 3. Tu n'as pas besoin du fichier local (ou tu acceptes de repartir de la version distante)

```bash
rm .pip_freeze_311.txt
# ou, sous Windows PowerShell : Remove-Item .pip_freeze_311.txt

git pull --ff-only
```

> Astuce : si tu veux simplement déplacer temporairement le fichier, tu peux le renommer (`mv .pip_freeze_311.txt sauvegarde.txt`) avant de relancer le `pull`.

## Résumé rapide

- L'erreur vient d'un fichier non suivi localement qui existe aussi sur la branche distante.
- `git restore --staged` échoue car le fichier n'est pas dans l'index.
- Solution : soit tu ajoutes/commits le fichier, soit tu le mets de côté (`stash`), soit tu le supprimes/renommes avant de relancer `git pull`.

Ainsi, tu évites toute perte de données et tu peux synchroniser ton dépôt sans blocage.
