
# 🎭 Puls-Events - RAG Cultural Events Assistant

**Projet OpenClassrooms P11** - Système RAG (Retrieval-Augmented Generation) pour la recommandation d'événements culturels en Occitanie

---

## 📋 Table des matières

- [Présentation](#présentation)
- [Objectifs](#objectifs)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Pipeline de données](#pipeline-de-données)
- [Technologies](#technologies)
- [Tests](#tests)
- [Configuration](#configuration)
- [Auteur](#auteur)

---

## 🎯 Présentation

**Puls-Events** est un Proof of Concept (POC) d'assistant intelligent basé sur un système RAG pour recommander des événements culturels en région Occitanie[file:5]. L'application permet aux utilisateurs de :

- **Rechercher** des événements via des questions en langage naturel
- **Converser** avec un chatbot intelligent pour affiner leurs recherches
- **Explorer** les événements avec filtrage temporel et géographique automatique
- **Visualiser** les statistiques d'utilisation (tokens, temps d'exécution)

Le système utilise **LangChain**, **Faiss** et plusieurs LLM (**Mistral AI**, **OpenAI**, **Google Gemini**) pour fournir des recommandations contextualisées à partir de données Open Agenda[file:5].

---

## 🎯 Objectifs

Ce projet répond aux exigences suivantes[file:3][file:5] :

1. ✅ **Environnement reproductible** : Gestion des dépendances, documentation complète
2. ✅ **Pipeline de données automatisé** : Fetch → Preprocessing → Vectorization → Indexation
3. ✅ **Base vectorielle Faiss** : Indexation optimisée avec IVFFlat
4. ✅ **Système RAG fonctionnel** : Intégration LangChain + LLM multi-providers
5. ✅ **Interface utilisateur** : Application Streamlit avec 3 modes (Accueil, Recherche, Chat)
6. ✅ **Tests unitaires** : Validation des données et des composants
7. ✅ **Traçabilité** : Comptabilité des tokens et métriques de performance

---

## 🏗️ Architecture

### Architecture globale

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

### Composants principaux

| Composant | Fichier(s) | Rôle |
|-----------|-----------|------|
| **Data Fetcher** | `src/vector/data_fetcher.py` | Récupération des événements depuis Open Agenda |
| **Preprocessor** | `src/vector/preprocessing.py` | Nettoyage, validation, structuration des données |
| **Vectorizer** | `src/vector/vectorization.py` | Chunking, embedding, indexation Faiss |
| **RAG Engine** | `src/rag/engine.py` | Retrieval + génération de réponses |
| **SearchBot** | `src/rag/searchbot.py` | Mode recherche simple (question → réponse) |
| **ChatBot** | `src/rag/chatbot.py` | Mode conversationnel avec historique |
| **Query Parser** | `src/rag/query_parser.py` | Extraction contraintes (date, ville, département) |
| **LLM Factory** | `src/llm/factory.py` | Abstraction multi-LLM (Mistral/OpenAI/Gemini) |
| **SeekEngine** | `src/core/seek_engine.py` | Interface unifiée pour recherche et chat |
| **Streamlit UI** | `main.py`, `src/ui/` | Interface utilisateur multipage |

---

## 📦 Prérequis

- **Python 3.10+** (testé sur Python 3.10)
- **Linux/MacOS/Windows** (développé sur Debian)
- **API Keys** :
  - Mistral AI OU OpenAI OU Google Gemini (au moins un)
  
---

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone <URL_DU_REPO>
cd OCDE-P11
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt`** :
```
langchain>=1.2.7
mistralai>=0.0.12
langchain-mistralai>=0.0.3
openai>=1.0.0
langchain-openai>=0.1.0
google-genai>=0.1.0
langchain-google-genai>=0.1.0
faiss-cpu>=1.13.2
pandas>=3.0.0
python-dotenv>=0.9.9
streamlit>=1.35.0
streamlit-extras>=0.4.0
```

### 4. Configuration des API Keys

Créer un fichier `.env` à la racine du projet[file:6] :

```bash
# Choisir au moins un provider
MISTRAL_API_KEY=your_mistral_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Configuration (optionnel)
REGION=Occitanie
DAYS_BACK=365
MAX_PAGES=100
```

---

## 📁 Structure du projet

```
OCDE-P11/
│
├── main.py                      # Point d'entrée Streamlit
├── config.py                    # Configuration centralisée
├── requirements.txt             # Dépendances Python
├── .env                         # Variables d'environnement (API keys)
├── README.md                    # Ce fichier
│
├── data/                        # Données (gitignored)
│   ├── raw/                     # Snapshots bruts Open Agenda
│   ├── processed/               # Événements nettoyés
│   ├── indexes/                 # Index Faiss + metadata
│   └── token_logs/              # Logs de consommation tokens
│
├── src/
│   ├── core/
│   │   └── seek_engine.py       # Interface unifiée Search/Chat
│   │
│   ├── llm/
│   │   ├── factory.py           # Factory multi-LLM
│   │   ├── base.py              # Classe abstraite LLM
│   │   ├── mistral_llm.py       # Implémentation Mistral
│   │   ├── openai_llm.py        # Implémentation OpenAI
│   │   └── gemini_llm.py        # Implémentation Gemini
│   │
│   ├── rag/
│   │   ├── engine.py            # Moteur RAG (retrieval + LLM)
│   │   ├── searchbot.py         # Bot recherche simple
│   │   ├── chatbot.py           # Bot conversationnel
│   │   ├── query_parser.py      # Parsing contraintes (date/ville)
│   │   └── langchain_bridge.py  # Pont LangChain
│   │
│   ├── ui/
│   │   ├── components.py        # Composants Streamlit réutilisables
│   │   └── pages/
│   │       ├── home.py          # Page d'accueil
│   │       ├── search.py        # Page recherche
│   │       ├── chat.py          # Page chat
│   │       └── analytics.py     # Page analytics
│   │
│   ├── utils/
│   │   ├── utils.py             # Utilitaires généraux
│   │   └── token_accounting.py # Comptabilité tokens
│   │
│   └── vector/
│       ├── data_fetcher.py      # Récupération Open Agenda
│       ├── preprocessing.py     # Nettoyage données
│       ├── vectorization.py     # Embedding + Faiss
│       └── run_pipeline.py      # Orchestrateur pipeline
│
├── tests/
│   └── test_*.py                # Tests unitaires
│
└── scripts/
    ├── fetch.sh                 # Lancer pipeline données
    └── app.sh                   # Lancer application Streamlit
```

---

## 🎮 Utilisation

### Pipeline de données (une seule fois)

**Étape 1** : Collecter les événements Open Agenda[file:6]

```bash
python src/vector/run_pipeline.py
```

Ce script exécute automatiquement :
1. **Fetch** : Téléchargement des événements (région Occitanie, derniers 365 jours)
2. **Preprocessing** : Nettoyage, déduplication, validation
3. **Vectorization** : Chunking + Embedding + Indexation Faiss

Options disponibles :
```bash
# Forcer une date de snapshot spécifique
python src/vector/run_pipeline.py --snapshot-date 2026-01-25

# Limiter le nombre de pages
python src/vector/run_pipeline.py --max-pages 50

# Vectoriser uniquement (si preprocessing déjà fait)
python src/vector/run_pipeline.py --vectorize-only --snapshot-date 2026-01-25

# Lister les snapshots disponibles
python src/vector/run_pipeline.py --list-indexes
```

**Durée estimée** : 10-30 minutes selon le nombre d'événements (≈2000-5000 événements)[file:6]

---

### Lancer l'application Streamlit

```bash
streamlit run main.py
```

Ou via le script shell :
```bash
bash app.sh
```

L'application s'ouvre sur **http://localhost:8501**[file:6]

---

### Modes d'utilisation

#### 🏠 **Page Accueil**
- Vue d'ensemble du système
- Statistiques du dataset
- Informations techniques

#### 🔍 **Recherche Simple**
- Poser une question en langage naturel
- Exemples :
  - *"Concerts ce weekend à Toulouse"*
  - *"Expositions en février à Montpellier"*
  - *"Spectacles pour enfants demain"*
- Résultats : Réponse + sources + statistiques

#### 💬 **Chat Interactif**
- Conversation continue avec l'assistant
- Historique de conversation
- Affinage progressif des recherches

#### 📊 **Analytics**
- Consommation de tokens (vectorization + recherches)
- Statistiques de session
- Monitoring des performances

---

## 🔧 Pipeline de données

### Architecture du pipeline[file:6]

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATEUR                         │
│              src/vector/run_pipeline.py                  │
└────┬────────────────────┬─────────────────────┬─────────┘
     │                    │                     │
     ▼                    ▼                     ▼
┌─────────┐        ┌──────────────┐     ┌──────────────┐
│ STEP 1  │        │   STEP 2     │     │   STEP 3     │
│  Fetch  │   →    │ Preprocess   │  →  │ Vectorize    │
└─────────┘        └──────────────┘     └──────────────┘
     │                    │                     │
     ▼                    ▼                     ▼
  raw JSON          processed JSON    Faiss Index + Metadata
```

### Détail des étapes[file:6]

#### **STEP 1 : Data Fetching** (`data_fetcher.py`)

```python
class OpenAgendaFetcher:
    def fetch_all(limit=100, max_pages=100):
        # Pagination automatique
        # Filtre : région + derniers 365 jours
        # Retry avec backoff exponentiel
        # → data/raw/raw_snapshot_YYYY-MM-DD.json
```

**Paramètres API** :
- `location_region='Occitanie'`
- `lastdate_begin >= (today - 365 days)`
- Champs sélectionnés : `title`, `description`, `longdescription_fr`, `location_city`, `timings`, etc.[file:6]

**Sortie** :
```json
{
  "metadata": {
    "fetch_date": "2026-02-10T14:00:00",
    "region": "Occitanie",
    "total_events": 4532
  },
  "events": [ {...}, {...}, ... ]
}
```

---

#### **STEP 2 : Preprocessing** (`preprocessing.py`)

```python
class EventPreprocessor:
    def run_full_pipeline():
        standardize_columns()      # Uniformisation
        drop_duplicates()          # Déduplication par UID
        handle_missing_values()    # Gestion valeurs manquantes
        validate_text_content()    # Texte minimum 10 chars
        sort_and_reset_index()     # Tri par UID
```

**Validations appliquées**[file:6] :
- ✅ Champs requis : `uid`, `title`, `timings`, `location_city`
- ✅ Filtrage temporel : événements < 365 jours
- ✅ Nettoyage HTML : suppression balises
- ✅ Géolocalisation : exclusion événements sans ville

**Sortie** :
```json
[
  {
    "uid": "evt_12345",
    "title": "Festival Jazz à Toulouse",
    "description_fr": "...",
    "longdescription_fr": "...",
    "location_city": "Toulouse",
    "location_department": "Haute-Garonne",
    "timings": [...],
    "first_date": "2026-03-15T20:00:00",
    ...
  }
]
```

---

#### **STEP 3 : Vectorization** (`vectorization.py`)

```python
class EventVectorizer:
    def run_full_vectorization_pipeline():
        chunks = chunk_events(events, chunk_size=500)
        embeddings = vectorize_chunks(chunks, batch_size=100)
        index = create_faiss_index(embeddings, n_vectors)
        save_index(index, snapshot_date)
        save_metadata(chunks, events, snapshot_date)
```

**Chunking intelligent**[file:6] :
- Taille max : 500 caractères
- Séparation progressive : `. ` → `, ` → ` `
- Préservation du contexte événement

**Embedding**[file:6] :
- Batch size : 100 chunks/appel API
- Dimension : 1024 (Mistral) ou 1536 (OpenAI) ou 768 (Gemini)
- Retry avec backoff si échec

**Indexation Faiss** :
```python
# IndexIVFFlat : optimisé pour 1K-100K vecteurs
nlist = min(100, max(10, n_vectors // 100))
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(embeddings)
index.add(embeddings)
```

**Sorties** :
- `data/indexes/{provider}_faiss_index_YYYY-MM-DD.bin` : Index Faiss
- `data/indexes/{provider}_metadata_YYYY-MM-DD.json` : Métadonnées (chunk_id → event_id, text, title, city, dates, url)

---

### Gestion des snapshots

Le système supporte **plusieurs snapshots** avec dates différentes[file:6] :

```python
# config.py
DEV_SNAPSHOT_DATE = "2026-01-25"  # Snapshot figé pour développement
```

**Snapshots disponibles** :
```bash
python src/vector/run_pipeline.py --list-indexes
```

**Changement de snapshot** : Sélection dans le sidebar Streamlit[file:6]

---

## 🤖 Technologies

### Stack technique

| Couche | Technologie | Version | Rôle |
|--------|-------------|---------|------|
| **Interface** | Streamlit | 1.35+ | UI multipage |
| **Orchestration RAG** | LangChain | 1.2+ | Chaînes RAG, retrieval |
| **LLM Chat** | Mistral AI / OpenAI / Gemini | Latest | Génération réponses |
| **LLM Embed** | Mistral Embed / OpenAI Ada / Gemini Embed | Latest | Vectorisation texte |
| **Vector DB** | Faiss (CPU) | 1.13+ | Indexation + recherche |
| **Data Processing** | Pandas | 3.0+ | Manipulation DataFrames |
| **API Data** | Open Agenda API | v3 | Source événements |
| **Env Management** | python-dotenv | 0.9+ | Variables environnement |

### Choix techniques justifiés

#### **Faiss (IndexIVFFlat)**
- ✅ Rapide pour 1K-100K vecteurs (notre use case : 2K-10K)
- ✅ Backend CPU suffisant pour POC
- ✅ Recherche exacte L2 avec clustering
- ⚠️ Limite : >1M vecteurs nécessiterait IndexHNSW ou GPU

#### **LangChain**
- ✅ Abstraction haut niveau pour RAG
- ✅ Gestion automatique du prompt engineering
- ✅ Support multi-LLM natif
- ✅ Chaînes réutilisables

#### **Multi-LLM**
- ✅ Flexibilité : tester plusieurs providers
- ✅ Abstraction via Factory Pattern (`src/llm/factory.py`)
- ✅ Fallback possible si quota dépassé

---

## 🧪 Tests

### Tests unitaires[file:6]

```bash
# Tester l'environnement
python tests/test_environment.py

# Tester la vectorisation
python tests/test_vectorization.py

# Tester le RAG
python tests/test_rag.py
```

### Tests couverts

| Module | Tests | Fichier |
|--------|-------|---------|
| **Environnement** | Imports, Faiss fonctionnel | `test_environment.py` |
| **Vectorization** | Chunking, embedding, indexation | `test_vectorization.py` |
| **RAG Engine** | Query parsing, retrieval, answer generation | `test_rag.py` |
| **Query Parser** | Extraction date, ville, département | `test_rag.py` |

---

## ⚙️ Configuration

### Variables d'environnement (`.env`)

```bash
# === API KEYS (au moins une) ===
MISTRAL_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# === DATA PIPELINE ===
REGION=Occitanie           # Région géographique
DAYS_BACK=365              # Historique (jours)
MAX_PAGES=100              # Limite pagination API

# === RAG PARAMETERS ===
LLM_PROVIDER=mistral       # mistral | openai | gemini
CHUNK_SIZE=500             # Taille chunks (caractères)
TOP_K=5                    # Nombre résultats retrieval
TEMPERATURE=0.7            # Température LLM (0.0-1.0)
```

### Configuration avancée (`config.py`)

```python
class Config:
    # Chemins
    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    INDEXES_DIR = DATA_DIR / "indexes"
    
    # API Open Agenda
    BASE_URL = "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    
    # Champs sélectionnés
    SELECTED_FIELDS = ['uid', 'title', 'description_fr', ...]
    REQUIRED_FIELDS = ['uid', 'title', 'timings', 'location_city']
    CHUNK_FIELDS = ['title', 'description_fr', 'longdescription_fr']
    
    # LLM Models
    LLM_MODELS = {
        'mistral': {'chat': 'mistral-large-latest', 'embed': 'mistral-embed'},
        'openai': {'chat': 'gpt-4o-mini', 'embed': 'text-embedding-3-small'},
        'gemini': {'chat': 'gemini-2.0-flash-exp', 'embed': 'text-embedding-004'}
    }
```

---

## 📊 Exemples de requêtes

### Extraction automatique de contraintes[file:6]

Le système extrait automatiquement les **contraintes temporelles** et **géographiques** :

| Requête utilisateur | Date extraite | Ville | Département |
|---------------------|---------------|-------|-------------|
| *"Concerts ce weekend à Toulouse"* | Sam-Dim prochain | Toulouse | - |
| *"Expositions en février à Montpellier"* | 01-28/02/2026 | Montpellier | - |
| *"Spectacles demain"* | 11/02/2026 | - | - |
| *"Événements dans le Gard"* | Prochains 30j | - | Gard |

Lorsque l'on utilise une date de snapshot référencé 🔒 DEV dans le sélecteur de snapshot,
la date du snapshot est considéré comme "aujourd'hui" et sert de référence au calcul de constraintes
permettant la repdocutibilité des questions/réponses.

### Parsing date intelligent[file:6]

```python
# src/rag/query_parser.py
QueryParser.parse_date("concerts ce weekend")
# → (datetime(2026, 02, 15), 1)  # Samedi + 1 jour

QueryParser.parse_date("en mars")
# → (datetime(2026, 03, 01), 30)  # 01-31 mars

QueryParser.parse_date("le 14/07")
# → (datetime(2026, 07, 14), 0)  # Date exacte
```

---

## 👤 Auteur

**Projet OpenClassrooms P11 - Data Engineer**  
Formation : Ingénieur Data  
Date : Février 2026

---

## 📝 Licence

Ce projet est un POC éducatif réalisé dans le cadre d'une formation OpenClassrooms[file:3][file:5].

---

## 🔗 Ressources

- [Open Agenda API](https://data.opendatasoft.com/api/explore/v2.1/console)
- [LangChain Documentation](https://python.langchain.com/)
- [Faiss Documentation](https://github.com/facebookresearch/faiss)
- [Mistral AI](https://docs.mistral.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🐛 Dépannage

### Problème : "Index not found"
**Solution** : Lancer le pipeline de données
```bash
python src/vector/run_pipeline.py
```

### Problème : "API Key not found"
**Solution** : Vérifier le fichier `.env` et les variables
```bash
echo $MISTRAL_API_KEY  # Doit afficher votre clé
```

### Problème : Erreur Faiss "dimension mismatch"
**Solution** : Régénérer l'index avec le bon provider
```bash
python src/vector/run_pipeline.py --vectorize-only --snapshot-date YYYY-MM-DD
```

### Problème : Streamlit ne démarre pas
**Solution** : Vérifier les dépendances
```bash
pip install --upgrade streamlit
streamlit hello  # Test Streamlit
```

---

## ✅ Checklist production

- [ ] Tests unitaires passent
- [ ] Variables `.env` configurées
- [ ] Pipeline exécuté (index + metadata présents)
- [ ] Application Streamlit démarre sans erreur
- [ ] Recherche simple fonctionne
- [ ] Chat interactif fonctionne
- [ ] Analytics affiche les tokens
- [ ] Documentation à jour

---

**🎭 Puls-Events - Votre assistant culturel intelligent en Occitanie**
```

***

Ce README couvre tous les aspects requis dans les livrables : [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_cae1b248-9511-4e7b-9c4e-85caeb63e5af/074e7c59-fe78-4716-bcd9-f9f2c859ff2c/Mission-livrables.pdf)
- ✅ Présentation et objectifs clairs
- ✅ Instructions de reproduction complètes
- ✅ Description détaillée de la structure
- ✅ Documentation des scripts (preprocessing, vectorisation)
- ✅ Architecture technique justifiée
- ✅ Guide d'utilisation pas-à-pas
- ✅ Configuration des dépendances
- ✅ Tests unitaires mentionnés
- ✅ Dépannage et ressources

Le fichier est prêt à être placé à la racine du projet sous `README.md`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_cae1b248-9511-4e7b-9c4e-85caeb63e5af/6457e3a0-5c07-4df3-bf0f-940808134ebc/OCDE-P11.txt)