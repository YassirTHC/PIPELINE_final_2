# Audit du style Hormozi 1 et de la logique emojis

Ce rapport résume, à partir du code actuel (`hormozi_subtitles.py`) et de la documentation
`SUBTITLE_AND_EMOJI_WORKFLOW.md`, dans quelle mesure le rendu correspond au style Hormozi 1
célèbre sur TikTok, et si la logique d'emojis est cohérente.

## 1. Fidélité au style Hormozi 1

### Points conformes

* **Groupes de 2–3 mots avec bounce** – `parse_transcription_to_word_groups` regroupe
  les tokens par deux en appliquant un effet bounce/zoom sur l'apparition, ce qui
  rappelle bien les animations rapides des montages Hormozi.【F:hormozi_subtitles.py†L369-L454】【F:hormozi_subtitles.py†L812-L878】
* **Texte en majuscule, police Impact/Bold et contour noir** – les mots sont
  normalisés en majuscules, la classe cherche une police Impact/Liberation Bold,
  et applique un contour noir épais lors du rendu, ce qui rapproche le look des
  sous-titres viraux.【F:hormozi_subtitles.py†L339-L411】【F:hormozi_subtitles.py†L584-L612】【F:hormozi_subtitles.py†L878-L1008】
* **Position basse avec marge de sécurité** – la logique maintient les sous-titres
  centrés, en bas de l'image, avec une marge (~200 px) proche des standards Hormozi.【F:hormozi_subtitles.py†L776-L858】

### Écarts relevés

* **Palette incohérente** – la table `category_colors` surcharge plusieurs fois la clé
  `business` (or puis bleu) et ne définit pas la clé `finance` alors qu'elle est
  utilisée plus loin, ce qui provoque des retours en blanc et des erreurs de cohérence
  par rapport au combo jaune/vert très marqué du style Hormozi.【F:hormozi_subtitles.py†L112-L170】【F:hormozi_subtitles.py†L337-L374】
* **Pas de fond coloré derrière le mot clé** – `keyword_background` est à `False` et
  aucun rectangle de couleur n'est dessiné derrière les mots ; or le style Hormozi 1
  se caractérise justement par des blocs pleins jaunes/verts/rouges sous les mots
  clés. Le pipeline produit donc un texte coloré mais sans background, ce qui reste
  visuellement loin du rendu attendu.【F:hormozi_subtitles.py†L82-L140】【F:hormozi_subtitles.py†L878-L1008】
* **Couverture partielle des catégories** – des catégories ajoutées dans
  `_bootstrap_categories` (`sales`, `content`, `sports`, etc.) ne possèdent ni couleur
  ni emojis dédiés, de sorte que les mots correspondants ressortent en blanc, sans
  accent, contrairement à la direction artistique Hormozi qui exige une coloration
  quasi systématique des termes forts.【F:hormozi_subtitles.py†L660-L732】
* **Police non garantie** – si Impact/Arial Bold ne sont pas installées, le code
  retombe sur la police par défaut Pillow, ce qui casse l'identité visuelle. Un pack
  de police embarqué serait nécessaire pour garantir le style.【F:hormozi_subtitles.py†L584-L604】

> **Conclusion** : le moteur pose de bonnes bases (majuscule, contour, bounce), mais
> il manque la palette jaune/verte sur fond rectangulaire et une table de couleurs
> cohérente pour répliquer fidèlement le style Hormozi 1 viral.

## 2. Qualité du mapping et du rendu des emojis

### Points positifs

* **Catégories riches** – `category_emojis` propose une bonne variété par thèmes
  (finance, action, émotions, etc.) avec rotation pour limiter les répétitions.
  L'intensité est prise en compte via la ponctuation et certains mots clés.【F:hormozi_subtitles.py†L188-L256】【F:hormozi_subtitles.py†L768-L838】
* **Évitement des doublons immédiats** – `_choose_emoji_for_tokens` mémorise le dernier
  emoji pour éviter de ressortir exactement le même symbole sur deux groupes
  consécutifs.【F:hormozi_subtitles.py†L768-L824】【F:hormozi_subtitles.py†L824-L838】
* **Fallback PNG/Unicode** – lorsqu'aucun PNG Twemoji n'est disponible, le pipeline
  bascule sur un rendu Unicode en police couleur, évitant un trou dans l'affichage.【F:hormozi_subtitles.py†L612-L652】【F:hormozi_subtitles.py†L1008-L1040】

### Limites et incohérences

* **Couverture incomplète** – les catégories `sales`, `content`, `mobile`, `sports`,
  etc., introduites pour les mots clés, ne disposent d'aucune entrée dans
  `category_emojis`. Ces mots n'obtiennent donc jamais d'emoji dédié, ce qui nuit à la
  cohérence entre le discours et l'iconographie.【F:hormozi_subtitles.py†L660-L708】【F:hormozi_subtitles.py†L188-L256】
* **Priorité couleur → emoji cassée** – la construction de `self.keyword_colors`/`emoji_mapping`
  s'appuie sur `self.category_colors.get(cat, '#FFFFFF')`. Quand `cat` vaut `finance`
  ou `sales` (non définis dans la palette), le mot reste blanc et l'emoji mapping est
  ignoré. On perd ainsi les cas emblématiques (💰, ⚡) censés renforcer le style.【F:hormozi_subtitles.py†L248-L276】【F:hormozi_subtitles.py†L708-L736】
* **Densité aléatoire** – lorsque aucun score ne ressort, `_choose_emoji_for_tokens`
  force un fallback vers la catégorie `business`. On se retrouve facilement avec des
  💼/📊 hors contexte si les mots n'appartiennent à aucune catégorie connue.【F:hormozi_subtitles.py†L768-L824】
* **Pas d'association multi-mots** – le moteur raisonne groupe par groupe et ne gère
  pas d'emojis composés (ex : texte parlant d'"🔥 OFFER" devrait déclencher à la fois
  🔥 et 💰). Un enrichissement via `span_style_map` ou des big-emojis ciblés serait
  nécessaire pour atteindre la densité TikTok.【F:hormozi_subtitles.py†L188-L256】【F:hormozi_subtitles.py†L928-L1008】

> **Conclusion** : le mapping actuel suffit pour une première passe automatique, mais
> des trous de couverture et des couleurs incohérentes empêchent d'obtenir une
> narration emoji vraiment logique et alignée sur les mots clés.

## 3. Recommandations rapides

1. **Stabiliser la palette** : ajouter explicitement `finance`, `sales`, `content`, etc.
   dans `category_colors` et éviter les doublons de clés.
2. **Activer les fonds colorés** : dessiner des rectangles derrière les mots clés
   (`keyword_background=True`) pour coller à la charte Hormozi.
3. **Fournir une police embarquée** : inclure Impact (ou équivalent) dans le repo et
   la charger par défaut.
4. **Compléter `category_emojis`** : fournir des listes pour toutes les catégories
   créées dans `_bootstrap_categories`, et enrichir `emoji_alias` pour couvrir
   davantage de cas en français.
5. **Réduire le fallback générique** : ajuster `_choose_emoji_for_tokens` pour
   retourner `""` plutôt qu'un emoji hors sujet quand aucune catégorie ne ressort.

Ces ajustements rapprocheraient nettement le rendu du style Hormozi 1 viral et
rendraient les emojis plus cohérents avec le contenu des sous-titres.

## 4. Résolu / Mises à jour

* **Police Montserrat prioritaire** – les fichiers `Montserrat-ExtraBold.ttf` et `Montserrat-Bold.ttf` sont embarqués et chargés avant les polices système. Le chemin effectif est résolu dans la config typée (`SubtitleSettings.font_path`) et consigné dans le log startup.【F:MANIFEST.in†L1-L1】【F:video_pipeline/config/settings.py†L162-L202】
* **Config typée appliquée au burn-in** – `video_processor_clean` transmet désormais `get_settings().subtitles` au wrapper `add_hormozi_subtitles`, garantissant que marge, taille, fonds de mots-clés et émojis suivent la configuration centralisée.【F:video_processor_clean.py†L811-L818】
* **Palette stabilisée + fonds rectangulaires** – `self.category_colors` couvre explicitement `finance`, `sales`, `content`, `mobile`, `sports` et synonymes. Lors du rendu, un fond arrondi est peint derrière chaque mot-clé lorsque `keyword_background` est actif.【F:hormozi_subtitles.py†L147-L182】【F:hormozi_subtitles.py†L870-L924】
* **Mapping emoji cohérent** – `category_emojis` fournit une liste pour chaque catégorie, les alias héritent de la même base et le moteur anti-repeat sélectionne un symbole différent sur les appels successifs ou retourne `""` si aucune catégorie n'est pertinente.【F:hormozi_subtitles.py†L184-L210】【F:hormozi_subtitles.py†L770-L828】
