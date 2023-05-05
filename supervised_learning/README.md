# TMP README





## All error in the curriculum (In French)

### Overall

---

#### Description du problème: __VM Ubuntu 16.04__
Problemes
- VM plus supporteé sur certain hardware -- Avec le guide d'installation fournis en premier année
- Lenteur du au simulation de composant -- Peu adapter au ML
- Techno et savoir de moins en moins demandé et déjà vu en première année (Vu un bien grand mots tout de même), sur le marchée la techno demandé acctuellement est Docker, remplie une bonne parties des fonctionnalitées des VM et bien plus modulaire et plus efficace

Impact: 
- Perte de temps de developement
- Frustration quotidienne quand a son utilisation (si installée)
- Peu valorisé sur le marché du travail dans cette spé

---

#### Description du problème: __Python version 3.5__
Problemes
- Version plus téléchargable / Installable | __Surtout plus maintenu__ juste a voir les warning lors de l'installation  
- Pose des soucis sur l'ajout des bonnes pratiques (Typing, Statement, Écriture en général du code)
- Effet boule de neige sur les package (Sur les version)
- Très peu performante du a l'ancien compiler (20% de gain en moyenne par nouvelle verison de computation), impact fort sur le temps de run des scripts

Impact: 
- Impossible d'être a jour sur les technos maintenus
- Impossible d'être a jour sur les bonnes pratiques
- Impact sur le temps de run du code
- Frustration d'installation vu que plus dispo

---

#### Description du problème: __Numpy 1.15__
Problemes
- Impossiblité de l'installer même si requirement de python3.5 validé (Sur certaines machines)
- Deprecated version

Impact: 
- Pas de grand impact tant a la retro-compatibilité de la version
- Quelques exemples dans les projets qui me sont pas egaux si on utiliser une version plus récente -- Plus les projets en tête

---
#### Description du problème: __Les commentaires__
Problemes
- Perte de temps immense, les commetaires sont la pour expliquer pourquoi certain choix doivent etre fait, et non pas expliquer le code. On est des auteurs et on ecrit pour d'autre dev, ils savent lire le code.
- Alternative :
  - Bien nommer ses varibles
  - Le typing

Impact: 
- Perte de temps d'aller-retours entre le code et le checker
- Lisibilité peu accru du code

---
#### Description du problème: __Les ressources__
La majorité des ressources sont bien cepandant

Problemes
- Les ressources wikipedia sont indigeste a lire, surtout pour des personnes sans grand baguage mathematique

Impact: 
- Overwhelming
- Sensation de n'avoir rien compris

### Projet 1 `Linear Algebra`
Un peu rapide, je pense qu'on aurait pu voir beaucoup plus de numpy dans le temps imparti

Pas de probleme particulier a la relecture du projet qui me sont venu en tête

### Projet 2 `Plotting`
Plutot cool, assez complet, mais dommage de ne voir ça que en one shot et pas une grosse fois + étalé dans les semaines futures, l'impression d'avoir rien retenu

Pas de probleme particulier a la relecture du projet qui me sont venu en tête
### Projet 3 `Calculus`
Pareil que linear algebra, projet rapide et aurait peut-etre merité d'avoir une petite introduction a `SymPy` ou `JAX` afin de pouvoir ecrire des expression mathematique et avoir accès au savoir de dérivations et integrales via ordinateurs (En plus d'avoir les bases a la main)

Pas de probleme particulier a la relecture du projet qui me sont venu en tête
### Projet 4 `Proba`
Projet complet, maque une application ou deux réel des lois de proba pour bien comprendre ce qu'on fait

Pas de probleme particulier a la relecture du projet qui me sont venu en tête

### Projet 5 `Classification`
Projet très instructif

#### Description du problème: __Les contraintes sur les boucles for__
Problemes
- Autorisation de boucle que 1 fois sur des petites ranges (qui resteront petite, ie verification des layers).

Impact: 
- Perte en lisibilité (On ne peut pas faire des fonction check et tout reste dans le init, ce qui en terme de responsabilité ne doit pas être la, mais dans des abstractions)

#### Description du problème: __Methods qui ne se servent pas de la classe__
Problemes
- Soucis de SoC, on a des methods qui ne devrait pas en être car ne se servent pas de la classe
  - Exemple:
    - Function cost (Tache 3) 
    - Function cost (Tache 11)
    - Function cost (Tache 19)

Impact: 
- Perte en lisibilité et SoC -- Mauvaise façon d'apprendre les classes.
- Référence a Sklearn ou les metrics sont separé des models (Dans les structures de classes)

#### Description du problème: __Naming__
Problemes
- Mauvaise utilisation des naming entre scalaire, vecteur et matrix, des vecteurs de retrouve en majuscule alors que pas dans les conventions
  - Exemple:
    - Toutes les fois ou Y est utilisé dans les classification binaire

Impact: 
- Mauvaise façon d'apprendre

### Projet 6 `Tensorflow`
Projet inutile

#### Description du problème: __Tensorflow version__
Problemes:
- Utilisation d'une version plus maintenu (cf la doc officiel (version utilisé par Hbtn: 1.12, version maintenu: 2.12, une version majeur d'écart)) | Plus installable (Meme sur Colab -- Tensorflow crée par Google et maintenu par (Un hasard je ne crois pas))
- Une rétro-compat et présente entre la v1 et la v2, cependant elle oblige a changer les mains donnée + changer notre code entre la pratique et le checker (Chronophage et prone aux erreurs)
- Toutes les fonctions utilisé ont leurs équivalent dans la version 2 (L'argument pedagogique de commencer par le low level et donc faux), soit avec d'autre noms, soit a d'autre endroit de l'API
- Certaines fonctions n'existe plus dans la compat et ne nous permette pas de run le code tout court (`tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")`)
  - Impacte les taches: 1; 2; 3; 4; 5; 6; 7 (Tous le projet)

Impact: 
- Aucun apprentissage est possible durant ce projet, aucune pratique que du théorique sur les github des autres
- Apprentissage biaisé sur une techno legacy -- Rarement utilisé et maintenue dans les projets des entreprise actuelles. (Oui on a les bases, on pourra apprendre facilement sur la nouvelle version / un autre framework, mais ça ne nous aide pas a nous rendre plus recrutable)
- Nouvelle façon de fonctionnement (Par default) pas abordé (Eager Mode vs Session Flow)
- Très peu de docs sur internet sur l'ancienne version
- Aucune mise a jour de ces versions -- Avec le domaine qui prends litteralement feu depuis 2 ans et ne fait que s'expendre. Rejoins le point 2.
- Ajout d'un potentiel gros temps d'onboarding car utilisation d'une ancienne techno (Oui 5 ans c'est ancien a cette echelle) + des anciennes pratiques liées.


### Projet 7 `Optimisation`
Plutot cool, encore des soucis avec Tensorflow

#### Description du problème: __Tensorflow version__
Problemes:
- Cf voir au dessus

Impact:
- Impossiblité de finir le projet (deux dernière taches)
- Impossiblité d'avoir une total comprehension global de toutes les technique vu ensemble
- Impossiblité de pouvoir écrire un blog a 100% accurate avec les manque sur les deux dernières taches

#### Description du problème: __Naming__
Problemes:
- Des majuscules dans les fonction ie `create_Adam_op`, `create_RMSProp_op`
- Des majuscules dans les fichier ie `9-Adam`

Impact:
- Maivaise exemple lors de l'apprentissage

### Projet 8 `Error Analysis`
Good

### Projet 9 `Regularization`

#### Description du problème: __Tensorflow version__
Problemes:
- Cf voir au dessus

Impact:
- Impossiblité de finir le projet (Tache 3 | Tache 6)
- Impossiblité d'avoir une total comprehension des techniques sur tensorflow vu en low level

### Resultat du premier mois
Aucune pratique de tensorflow concrete.