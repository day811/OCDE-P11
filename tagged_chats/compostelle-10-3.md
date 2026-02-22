# Rapport de session RAG
## Résumé de la requête

- **Question** : trouve moi des informations sur  l'avant-première du film "compostelle" le 10 mars
- **Datetime** : 2026/02/21 01:04:04:
- **Mode** : chat
- **Modèle de chat** : ChatMistralAI:mistral-small
- **Modèle d'embedding** : Mistral AI:mistral-embed
- **Température** : 0.7
- **Top_K** : 4
- **Temps d’exécution** : 7.579976 s
- **Distance moyenne** : 0.37438950687646866
- **Index** : `mistral_faiss_index_2026-02-19.bin`
- **Date ciblée** : 2026-03-10 00:00:00
## Réponse générée

Je n'ai pas trouvé d'information sur une avant-première du film *"Compostelle"* le 10 mars 2026 dans les événements listés. En revanche, voici une recommandation personnalisée basée sur les autres événements :

**Pour une soirée culturelle et engagée** :
- **10 mars 18h30** : Café Euclide (Quai des Savoirs) – Idéal pour un débat intellectuel dans un cadre convivial.
- **12 mars 12h** : Repas-débat *"Luttes féministes et sororité"* (Maison des chômeurs) – Pour échanger autour d’un repas solidaire.

**Pour une immersion dans l’histoire des femmes** :
- **3-14 mars** : Exposition *"L’effet Matilda"* (Médiathèque de Saint-Jory) – Gratuit et accessible.
- **8 mars 14h30** : Table ronde sur Marie Marvingt (Envol des Pionniers) – Pour découvrir une figure inspirante.

**Pour une projection cinématographique** :
- **8 mars 20h30** : *"Girls for Tomorrow"* (VO) – Un documentaire poignant sur les femmes et leur résilience.

Si vous cherchez spécifiquement *"Compostelle"*, je vous conseille de vérifier les sites des cinémas locaux ou des festivals en Occitanie. 😊

## Annotations 

- L'événement demandé n'apparait pas dans la liste bien qu'existant. Le chat ne cherche que dans les documents fournis
- Donner une précision sur le lieu (dept ou ville) permet de trouver l'événement
- Si le retriever (bug dev) ne renvoie aucun document, le llm trouve les informations seul sur le web.

## Événements utilisés comme contexte

| # | Titre | Ville | Date | Description (résumée) | Lien |
|---|--------|-------|------|-----------------------|------|
| 1 | L'effet Matilda, exposition et rencontre | Fenouillet | 10/03/2026, 14:30:00 | Au Café Euclide du Quai des Savoirs le 10 mars 2026 à 18h30 - plus d'infos Un repas-débat : "Luttes féministes et sororité" , proposé à à la Maison des chômeurs Partage Faourette de Toulouse le 12 mars 2026 à 12h - plus d'infos L'exposition L'effet Matilda :… | [lien](https://openagenda.com/planlibre/events/exposition-leffet-matilda-1744646) |
| 2 | L'effet Matilda, exposition et rencontre | Saint-Jory | 10/03/2026, 15:00:00 | Au Café Euclide du Quai des Savoirs le 10 mars 2026 à 18h30 - plus d'infos Un repas-débat : "Luttes féministes et sororité" , proposé à à la Maison des chômeurs Partage Faourette de Toulouse le 12 mars 2026 à 12h - plus d'infos L'exposition L'effet Matilda :… | [lien](https://openagenda.com/planlibre/events/exposition-leffet-matilda-4587539) |
| 3 | Dans les coulisses d’un métier pas comme les autres, instructrice d’astronautes ! | Toulouse | 10/03/2026, 18:30:00 | retrouvez également : Une table ronde : " Meet her in space ! À la rencontre des femmes du spatial", à la Cité de l’espace le 5 mars 2026 à 18h30 - www.cite-espace.com Une table ronde : " L’aviation sanitaire, de Marie Marvingt aux convoyeuses de l’air… | [lien](https://openagenda.com/planlibre/events/dans-les-coulisses-dun-metier-pas-comme-les-autres-instructrice-dastronautes) |
| 4 | Droits des femmes | Castanet-Tolosan | 10/03/2026, 09:00:00 | confiance et image de soi avec une socio-esthéticienne * Mercredi 1 avril 18h-21h → Atelier de self défense entre femmes avec Faire Face* • MJC - CINE 113 Tarifs : 4€ pour les femmes toute la journée / 6€ pour les hommes www.mjc-castanet-tolosan.fr/cinema-… | [lien](https://openagenda.com/mervilla/events/droits-des-femmes-5267861) |
## Statistiques de tokens

- **Tokens requête** : 24
- **Tokens contexte** : 577
- **Tokens LLM** : 201
- **Total tokens** : 802

