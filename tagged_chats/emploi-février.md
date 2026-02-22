# Rapport de session RAG
## Résumé de la requête

- **Question** : je cherche des evenements sur l'emploi au mois de février
- **Datetime** : 2026/02/20 01:08:02:
- **Mode** : chat
- **Modèle de chat** : ChatMistralAI:mistral-small
- **Modèle d'embedding** : Mistral AI:mistral-embed
- **Température** : 0.7
- **Top_K** : 5
- **Temps d’exécution** : 2.638021 s
- **Distance moyenne** : 0.34410442113876344
- **Index** : `mistral_faiss_index_2026-02-19.bin`
- **Fenêtre temporelle** : 2026-02-19 00:00:00 ± 9 jours
## Réponse générée

Voici les événements emploi en février 2026 en Occitanie qui pourraient vous intéresser :

1. **Forum des Emplois 4 saisons** (21/02) – Palais des Congrès du Cap d'Agde : Rencontres avec des employeurs de tous secteurs.
2. **Forum "Un emploi pour tous"** (19/02) – Montech : Offres dans l'agriculture, la restauration, l'aide à la personne, etc.
3. **Salon Jobs d'été 2026** (date non précisée) – 45 entreprises présentes pour des opportunités saisonnières.
4. **Réunion d'information Job Chef** (26/02) – Pour les passionnés de cuisine sans expérience, avec concours le 16/03.

*Conseil* : Si vous cherchez un emploi saisonnier, privilégiez le Salon Jobs d'été. Pour un métier dans la restauration, la réunion Job Chef est idéale.

## Annotations 

- 4/5 documents récupérés correspondent à la requête
- le llm a bien filtré ceux réellement correspondant "emploi" et correctement contextualisé les deux ne contenant pas ce terme

## Événements utilisés comme contexte

| # | Titre | Ville | Date | Description (résumée) | Lien |
|---|--------|-------|------|-----------------------|------|
| 1 | Forum des Emplois : Saisissez votre Chance le 21 Février ! | Agde | 21/02/2026, 08:30:00 | Forum des Emplois : Saisissez votre Chance le 21 Février !. Le Forum des Emplois 4 saisons organisé par la communauté d'Agglomération Hérault Méditerranée avec France Travail et ses partenaires qui se tiendra le 21 Février 2026 au Palais des Congrès du Cap… | [lien](https://openagenda.com/semaine-des-metiers-du-tourisme-2026/events/forum-des-emplois-saisissez-votre-chance-le-21-fevrier) |
| 2 | Forum "Un emploi pour tous" | Montech | 19/02/2026, 09:00:00 | Forum "Un emploi pour tous". Venez rencontrer des employeurs de tout secteur d'activité. Le jeudi 19/02/2026 aura lieu le forum "Un emploi pour tous" à la salle des fêtes Delbosc à Montech de 9H à 12H30. De nombreuses offres à pourvoir dans différents… | [lien](https://openagenda.com/semaine-des-metiers-du-tourisme-2026/events/forum-un-emploi-pour-tous-8537718) |
| 3 | SALON JOBS D'ETE 2026 | Béziers | 26/02/2026, 09:00:00 | SALON JOBS D'ETE 2026. ENTREE LIBRE Pour cette nouvelle édition, l'organisation évolue afin d'offrir une expérience encore plus fluide et dynamique : 45 entreprises présentes pour la journée. ENTREE LIBRE Pour cette nouvelle édition, l'organisation évolue… | [lien](https://openagenda.com/semaine-des-metiers-du-tourisme-2026/events/salon-jobs-dete-2026) |
| 4 | La cuisine, une passion qui pourrait bien devenir votre métier ! Participez au Concours Job Chef ! | Séméac | 26/02/2026, 10:00:00 | Vous souhaitez faire de votre passion pour la cuisine votre futur métier ? Vous avez la motivation mais pas l'expérience ? Cet évènement est fait pour vous ! France Travail vous propose de participer à une réunion d'information le jeudi 26 Février sur le… | [lien](https://openagenda.com/francetravail/events/la-cuisine-une-passion-qui-pourrait-bien-devenir-votre-metier-participez-au-concours-job-chef-3913689) |
| 5 | Lotos du mois de février 2026 | Alès | 21/02/2026, 20:30:00  ->  25/02/2026, 16:00:00 | Lotos du mois de février 2026. Retrouvez la liste des lotos du mois sur l'Agglomération.. Mardi 10 février : 16h, Les Salles-du-Gardon, anciennes écoles de l’Impostaire (Vivre ensemble) Mercredi 11 février : 16h, Les Salles-du-Gardon, anciennes écoles de… | [lien](https://openagenda.com/ales-agglomeration/events/lotos-du-mois-de-fevrier-2026) |
## Statistiques de tokens

- **Tokens requête** : 28
- **Tokens contexte** : 659
- **Tokens LLM** : 143
- **Total tokens** : 830

