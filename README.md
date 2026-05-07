# Compte-rendu My Movie Friend

Désolé pour le retard du README, le reste du projet a été rendu à l'heure. Uniquement le README a eu du retard.

## **Introduction**
Ce document synthétise le processus de développement du projet "My Movie Friend", un système de recommandation de films basé sur une architecture RAG (Retrieval-Augmented Generation). L'objectif était de transformer une base de données tabulaire (TMDB 5000) en une base de connaissances interrogeable en langage naturel via une interface utilisant Groq et ChromaDB.

**2. Décisions de Conception**
* **Transformation des données :** L'enjeu principal des données tabulaires dans un RAG est la perte de contexte. Pour y palier, j'ai développé une fonction de pré-traitement change les lignes pertinentes du CSV (Titre, Date, Note, Genres, Synopsis) en un paragraphe cohérent qui servira de chunk. Le parsing des JSON pour la colonne "genres" a permis d'enrichir considérablement la sémantique de ce texte.
* **Le Prompting :** La recommandation n'étant pas une science exacte, la qualité du résultat dépendait fortement du *System Prompt*. J'ai conçu des instructions spécifiques forçant le LLM à adopter un ton d'ami expert, à justifier ses choix en fonction de la requête de l'utilisateur, et à toujours citer la note et l'année pour garantir la fiabilité perçue.

**3. Difficultés Rencontrées et Solutions Apportées**

* **Défi A : La barrière de la langue**
  * *Problème :* Initialement, j'utilisais un modèle d'embedding basé sur LLaMA. Lors des tests, je me suis aperçu qu'il peinait à établir une correspondance sémantique entre les requêtes des utilisateurs (en français) et le contenu de la base de données vectorielle (synopsis originaux en anglais).
  * *Solution :* J'ai pivoté vers le modèle `paraphrase-multilingual-mpnet-base-v2`. Ce modèle est spécifiquement entraîné pour aligner les représentations vectorielles de textes de langues différentes ayant le même sens. Ce changement technique a drastiquement amélioré la pertinence des documents retournés par ChromaDB.

* **Défi B : Gestion du temps d'exécution et du Cache**
  * *Problème :* L'indexation de centaines de textes prend un temps significatif. Mon intention initiale était de développer de zéro une classe de gestion de cache, chargée de vérifier les *chunks* et de gérer la persistance des embeddings dans des cahces sur disques. 
  * *Solution :* Contraint par le temps imparti pour le TP, j'ai dû revoir mes priorités. Plutôt que de risquer de ne pas livrer un produit fonctionnel, j'ai abandonné l'idée de ce gestionnaire de cache sur-mesure au profit la solution de gestion de la Vector DB que tu as proposé. Cette approche, plus pragmatique à l'écosystème ChromaDB, permet de sauvegarder et charger l'index vectoriel directement sur le disque, réglant le problème de temps de chargement sans complexifier la base de code inutilement.
