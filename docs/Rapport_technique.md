

---

## **Rapport Technique – Système RAG pour la Recommandation d’Événements Culturels**
**Auteur** : Yves Dangel
**Date** : 22 février 2026
**Version** : 1.0

---

## 1. Introduction
### 1.1 Contexte
Ce rapport présente le Proof of Concept (POC) d’un système RAG (Retrieval-Augmented Generation) développé pour **Puls-Events**, une plateforme de gestion d’événements culturels. L’objectif est de démontrer la faisabilité d’un chatbot capable de recommander des événements culturels en temps réel, en s’appuyant sur une base de données vectorielle et un modèle de langage (LLM).

### 1.2 Objectifs
- **Automatiser** la collecte, le traitement et l’indexation des événements culturels.
- **Augmenter** les réponses du chatbot avec des données contextuelles précises.
- **Évaluer** la pertinence et la robustesse du système via des tests unitaires et un jeu de données annoté.
- **Documenter** les choix techniques et proposer des recommandations pour une version finale scalable.

---

## 2. Architecture du Système

### 2.1 Diagramme d’Architecture Globale
```
┌─────────────────┐
│  Open Agenda    │  ← Collecte événements (API REST)
│      API        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Data Fetch    │  ← Script: src/vector/data_fetcher.py
│   + Snapshot    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │  ← Script: src/vector/preprocessing.py
│  (Cleaning)     │     - Standardisation
└────────┬────────┘     - Déduplication
         │              - Validation
         ▼
┌─────────────────┐
│ Vectorization   │  ← Script: src/vector/vectorization.py
│   + Chunking    │     - Chunking (500 chars)
│   + Embedding   │     - Embedding (Mistral/OpenAI/Gemini)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Faiss Index    │  ← Index: IndexIVFFlat
│   (Retrieval)   │     - Recherche par similarité
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RAG Engine    │  ← Modules: src/rag/
│  (LangChain +   │     - Query parsing
│   LLM Chat)     │     - Context retrieval
└────────┬────────┘     - Answer generation
         │
         ▼
┌─────────────────┐
│   Streamlit     │  ← Interface: main.py + src/ui/
│      UI         │     - Recherche simple
└─────────────────┘     - Chat interactif
                        - Analytics
```

### 2.2 Composants Principaux
- **Data Fetch** : Récupération des événements via l’API Open Agenda, avec filtres géographiques (ex: Occitanie) et temporels (< 1 an).
- **Preprocessing** : Nettoyage, déduplication, validation des données.
- **Vectorization** : Découpage en chunks (500 caractères), embedding via Mistral/OpenAI/Gemini.
- **Faiss Index** : Indexation optimisée avec IVFFlat pour une recherche rapide par similarité.
- **RAG Engine** : Orchestration via LangChain pour la récupération de contexte et la génération de réponses.
- **Streamlit UI** : Interface utilisateur multipage (accueil, recherche, chat).

---

## 3. Choix Techniques

### 3.1 Base de Données Vectorielle : Faiss
- **Pourquoi Faiss ?**
  - Optimisé pour 1K-100K vecteurs (cas d’usage : 2K-10K événements).
  - Backend CPU suffisant pour le POC.
  - Recherche exacte L2 avec clustering.
- **Limites** : Scalabilité limitée au-delà de 1M vecteurs (nécessiterait IndexHNSW ou GPU).

### 3.2 Orchestration : LangChain
- **Avantages** :
  - Abstraction haut niveau pour le pipeline RAG.
  - Gestion automatique du prompt engineering.
  - Support natif multi-LLM.
  - Chaînes réutilisables et testables.

### 3.3 Modèles de Langage : Multi-LLM
- **Flexibilité** : Intégration de Mistral, OpenAI, et Gemini via un Factory Pattern.
- **Fallback** : Possibilité de basculer entre providers en cas de quota dépassé.

### 3.4 Stack Technique Complète
| Couche          | Technologie      | Version  | Rôle                          |
|-----------------|------------------|----------|-------------------------------|
| Interface       | Streamlit        | 1.35+    | UI multipage                  |
| Orchestration   | LangChain        | 1.2+     | Chaînes RAG, retrieval        |
| LLM Chat        | Mistral/OpenAI   | Latest   | Génération réponses          |
| LLM Embed       | Mistral Embed    | Latest   | Vectorisation texte           |
| Vector DB       | Faiss (CPU)      | 1.13+    | Indexation + recherche        |
| Data Processing | Pandas           | 3.0+     | Manipulation DataFrames      |
| API Data        | Open Agenda      | v3       | Source événements            |

---

## 4. Résultats du POC

### 4.1 Métriques de Performance
- **Temps d’exécution** : 2.155905 secondes pour une session de chat (exemple : recherche d’activités à Toulouse le 14/03/2026).
- **Distance moyenne** : 0.43767847418785094 (métrique de similarité entre requête et résultats).
- **Précision** : 100% des événements indexés correspondent au périmètre géographique et temporel défini.

### 4.2 Exemples de Requêtes
| Requête Utilisateur               | Date Extraite       | Ville      | Département |
|-----------------------------------|---------------------|------------|-------------|
| "Concerts ce weekend à Toulouse" | Sam-Dim prochain     | Toulouse   | -           |
| "Expositions en février à Montpellier" | 01-28/02/2026 | Montpellier | -           |

---

## 5. Recommandations pour la Version Finale

### 5.1 Améliorations Techniques
- **Scalabilité** :
  - Passer à un backend GPU pour Faiss si >1M vecteurs.
  - Utiliser IndexHNSW pour des recherches encore plus rapides.
- **Robustesse** :
  - Implémenter un système de fallback automatique entre LLM providers.
  - Ajouter un historique de conversation pour un contexte plus riche.
- **Maintenance** :
  - Automatiser la mise à jour de l’index Faiss (ex: cron job quotidien).
  - Monitorer les métriques de performance (latence, recall@k, coût).

### 5.2 Points de Vigilance
- **Données** : Vérifier rigoureusement les filtres temporels pour éviter les événements >1 an.
- **Index Faiss** : Scripts reproductibles pour la sauvegarde/chargement.
- **LangChain** : Conserver les métadonnées lors du retrieval pour un contexte riche.
- **Mistral** : Surveiller les quotas API et budgéter l’usage.

---

## 6. Conclusion
Ce POC a démontré la faisabilité d’un système RAG pour la recommandation d’événements culturels, avec une architecture modulaire, des performances satisfaisantes et une interface utilisateur intuitive. Les recommandations proposées visent à préparer une version finale scalable, robuste et maintenable, adaptée aux besoins de Puls-Events.

---
**Annexes** :
- Exemples de code (data_fetcher.py, preprocessing.py, vectorization.py).
- Résultats des tests unitaires.
- Capture d’écran de l’interface Streamlit.

---
**Prochaines Étapes** :
- Finaliser la présentation PowerPoint et la démo live.
- Préparer la soutenance en anticipant les questions techniques (optimisation Faiss, choix LangChain, etc.).

---
