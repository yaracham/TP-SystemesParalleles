# Travaux Pratiques – Systèmes Parallèles

Ce dépôt GitHub contient les **sujets des TP1 et TP2** ainsi que **mon rendu pour le TP2**, réalisés dans le cadre du cours de Systèmes Parallèles.

## Organisation du dépôt

Le dépôt est structuré de la manière suivante :


### TP1
Le dossier **TP1** contient le sujet du TP1 ainsi que mon travail **à partir de la partie 2 uniquement**, conformément aux consignes.

Les exercices portent notamment sur :
- la parallélisation de calculs (OpenMP / MPI),
- la circulation de messages entre processus,
- des problématiques classiques de calcul distribué.

### TP2
Le dossier **TP2** contient :
- le sujet complet du TP2,
- mon implémentation et mon rendu,
- un mini-rapport présentant les **résultats expérimentaux** et leur **analyse**.

Le TP2 est structuré autour de trois parties principales :
1. **Parallélisation de l’ensemble de Mandelbrot**
2. **Produit matrice–vecteur**
3. **Entraînement pour l’examen écrit (lois d’Amdahl et de Gustafson)**

## Résultats principaux du TP2

### 1. Ensemble de Mandelbrot (MPI – 4 processus, image 1024×1024)

| Méthode                    | Temps d’exécution (s) |
|---------------------------|------------------------|
| Répartition par blocs     | 1.061                  |
| Répartition cyclique      | 1.051                  |
| Stratégie maître-esclave  | 1.376                  |

**Analyse :**
- La répartition cyclique est légèrement plus performante grâce à un meilleur équilibrage de charge.
- La répartition par blocs donne des performances proches, le déséquilibre restant limité.
- La stratégie maître-esclave est pénalisée par le surcoût des communications MPI lorsque les tâches sont fines.

👉 Pour cette configuration, une **répartition statique bien choisie** est plus efficace qu’une stratégie dynamique.

### 2. Produit matrice–vecteur (MPI, np = 4)

Temps mesurés :
- Découpage par colonnes : `Tcols = 0.027088 s`
- Découpage par lignes : `Trows = 0.012808 s`

Le découpage par lignes est environ **2.1× plus rapide** que le découpage par colonnes, principalement à cause du coût des communications collectives (Allreduce) dans la version par colonnes.

### 3. Loi d’Amdahl et de Gustafson

- Fraction parallélisable : `p = 0.9`
- Speedup maximal théorique (Amdahl) :  
  **Smax = 10**
- Un nombre de nœuds raisonnable est d’environ **6 à 10**, au-delà duquel l’efficacité chute fortement.
- En doublant la taille des données, la loi de **Gustafson** prédit un speedup plus favorable (≈ 5.74 pour 6 nœuds).

## Remarques

- Tous les résultats présentés ont été obtenus expérimentalement.
- Les analyses mettent en évidence l’impact du **choix de la stratégie de parallélisation** et du **coût des communications**.

---

**Auteur :**  
Yara EL CHAM  
Élève ingénieur – ENSTA Paris  
Institut Polytechnique de Paris
