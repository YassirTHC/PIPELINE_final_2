# Comprendre l'erreur « Your local changes would be overwritten »

Lorsque `git pull` échoue avec un message du type :

```
error: Your local changes to the following files would be overwritten by merge:
    <fichier>
Please commit your changes or stash them before you merge.
```

cela signifie que Git détecte des modifications locales non enregistrées sur les mêmes
fichiers que ceux qui arrivent depuis la branche distante. Pour éviter de perdre ces
modifications, Git refuse d'appliquer le `pull`.

## Pourquoi cela arrive ?

1. Vous avez modifié (ou créé) un fichier en local, ici `.pip_freeze_311.txt`.
2. Pendant ce temps, le même fichier a été ajouté ou modifié dans `origin/main`.
3. Git ne peut pas fusionner proprement car votre version locale et la version distante
   divergent sans historique commun.

## Comment résoudre la situation ?

### Option A : enregistrer vos modifications locales

1. Inspectez ce que vous avez en attente : `git status`.
2. Ajoutez le fichier si vous souhaitez le garder : `git add .pip_freeze_311.txt`.
3. Créez un commit local : `git commit -m "Sauvegarde locale du freeze pip"`.
4. Relancez la mise à jour : `git pull --ff-only`.

### Option B : mettre de côté temporairement (stash)

Si vous voulez récupérer la version distante sans conserver vos modifications :

1. Sauvegardez vos changements dans un stash : `git stash push -m "freeze pip local"`.
2. Mettez à jour la branche : `git pull --ff-only`.
3. Réappliquez le stash si besoin : `git stash pop` (résoudrez les conflits le cas échéant).

### Option C : abandonner vos modifications locales

Si le fichier local ne vous intéresse pas :

1. Remettez le fichier dans l'état précédent : `git restore .pip_freeze_311.txt`.
2. Mettez à jour la branche : `git pull --ff-only`.

---

En résumé, Git protège vos changements locaux en bloquant le `pull`. Pour continuer,
il suffit soit de les enregistrer (commit), soit de les mettre de côté (stash), soit de
les abandonner avant de synchroniser avec `origin/main`.
