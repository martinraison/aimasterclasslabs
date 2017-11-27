# AI Masterclass Labs

Bienvenue dans l'AI Masterclass Labs. Le but de cette séance est d'aider l'association "Agir pour l'Ecole" à développer de nouveaux outils d'apprentissage de l'écriture pour les enfants.

Nous disposons de données contenant des caractères écrits par des enfants. Le but est de créer un modèle permettant de détecter automatiquement quel caractère a été écrit (ce qui peut ensuite être utilisé dans des applications pour tablette/smartphone/etc)

Ces données sont présentes en quantité très limitée pour l'instant (quelques centaines de caractères), c'est pourquoi il est préférable d'entraîner d'abord sur un plus gros dataset (EMNIST), et d'appliquer ensuite ce modèle sur les données de l'association.

### Prérequis

Pytorch et torchvision doivent être installés. Suivre les instructions d'installation ici http://pytorch.org/

Pour entraîner un modèle de reconnaissance de caractères sur EMNIST:

### Utilisation

```
python main.py
```

Pour évaluer le modèle sur des données fournies par l'association:

```
python eval.py
```

### Instructions

1. forker le repo
2. ajouter l'URL du fork sur [ce document](https://docs.google.com/spreadsheets/d/1Jqwg_Q6kx909itz1_gIsi1VB3kOwg3caD83KCnUD1yU/edit#gid=0)
3. coder! Le but est d'améliorer le score d'évaluation. Nous exécuterons un script d'évaluation similaire à `eval.py`, votre code doit donc contenir un fichier `predictor.py` avec une classe `Predictor`
4. lorsque vous mettez à jour le repo, nous lançons automatiquement l'évaluation et mettons à jour le [leaderboard](http://54.166.222.173/)

### Quelques pistes (non exhaustives)

1. Le modèle de base n'utilise pas de convolutions... ça peut être utile :)
2. Vous pouvez aussi utiliser des techniques de "data augmentation" (rotations/translations aléatoires des images de training, etc)
3. Les données d'entraînement (EMNIST) et d'évaluation (Agir pour l'Ecole) proviennent de sources différentes! Il peut être utile d'utiliser les données de validation fournies pour "fine-tuner" le modèle (mais pas n'importe comment...)

Bon courage!
