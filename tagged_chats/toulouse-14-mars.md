# Rapport de session RAG
## Résumé de la requête

- **Question** : Que faire à  Toulouse le 14 mars ?
- **Datetime** : 2026/02/22 01:05:04:
- **Mode** : chat
- **Modèle de chat** : ChatMistralAI:mistral-small
- **Modèle d'embedding** : Mistral AI:mistral-embed
- **Température** : 0.7
- **Top_K** : 5
- **Temps d’exécution** : 2.012098 s
- **Distance moyenne** : 0.44022822976112364
- **Index** : `mistral_faiss_index_2026-02-19.bin`
- **Date ciblée** : 2026-03-14 00:00:00
- **Ville ciblée** : TOULOUSE
## Réponse générée

Le **14 mars**, à Toulouse, je te recommande :
- **"Regards croisés sur l'accrochage du musée"** (21 rue de Metz) – Une visite guidée passionnante pour découvrir les coulisses des expositions (1h30, 9€/7€).

Si tu cherches une activité créative, l'atelier de **sculpture en papier** (même adresse) pourrait aussi t'intéresser, mais vérifie les dates exactes.

*Conseil* : Réserve si nécessaire, car certains événements sont limités en places.

## Annotations 

- Requête 100% efficace, 5/5 correspondent à la requête
- Double contrainte date+lieu efficace 

## Événements utilisés comme contexte

| # | Titre | Ville | Date | Description (résumée) | Lien |
|---|--------|-------|------|-----------------------|------|
| 1 | Le ciel nous vengera. Nicolas Daubanes. | Toulouse | 14/03/2026, 11:00:00 | Vernissage le samedi 7 mars 2026, à 17h à la Maison Salvan et à 19h au Castelet. Entrée libre \| visite guidée : 3€. Toulouse. 18 bis Grande Rue Saint-Michel 31400 Toulouse, France. Haute-Garonne. | [lien](https://openagenda.com/monuments-toulouse/events/le-ciel-nous-vengera-nicolas-daubanes) |
| 2 | Lecture | Toulouse | 14/03/2026, 10:30:00 | Lecture. pour les 0-3 ans. Lecture pour les 0-3 ans. Tous les mercredis et samedis 10h30. Toulouse. 125 Avenue Jean Rieux, 31400 Toulouse, France. Haute-Garonne. | [lien](https://openagenda.com/bibliotheques-de-toulouse/events/lecture-5422864) |
| 3 | Journée Portes Ouvertes du Conservatoire | Toulouse | 14/03/2026, 09:00:00 | Venez, écoutez, apprenez, posez des questions et peut-être même, trouvez votre voie artistique au Conservatoire de Toulouse. RENDEZ-VOUS : DANSE : Site Espace DANSE - 12 Place Saint-Pierre de 9H À 16H MUSIQUE : au Conservatoire - 17 rue Larrey de 9h À 18h… | [lien](https://openagenda.com/conservatoire-de-toulouse/events/journee-portes-ouvertes-du-conservatoire-3329456) |
| 4 | Atelier P'tits Artistes 6-12 ans | Toulouse | 14/03/2026, 10:30:00  ->  14/03/2026, 14:00:00 | sculpture en papier) &gt; Durée : 2h &gt; 10€ &gt; Pas de parent accompagnateur pour cette activité, Toulouse. 21 rue de Metz 31000 Toulouse. Haute-Garonne. | [lien](https://openagenda.com/musee-des-augustins-toulouse/events/atelier-ptits-artistes-6-12-ans) |
| 5 | Conversation autour de l'art | Toulouse | 14/03/2026, 16:30:00 | 14/03 : Regards croisés sur l'accrochage du musée &gt; Durée : 1h30 &gt; Tarif : 9€ tarif plein /7€ tarif jeunes et étudiants / 7€ tarif réduit PSH, minimas sociaux, Amis du musée. Toulouse. 21 rue de Metz 31000 Toulouse. Haute-Garonne. | [lien](https://openagenda.com/musee-des-augustins-toulouse/events/conversation-autour-de-lart-8682418) |
## Statistiques de tokens

- **Tokens requête** : 19
- **Tokens contexte** : 490
- **Tokens LLM** : 81
- **Total tokens** : 590

