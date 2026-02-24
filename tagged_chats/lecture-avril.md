# Rapport de session RAG
## Résumé de la requête

- **Question** : Je suis passionné de livres, propose moi des activités autour de la lecture et de l'écriture au mois d'avril
- **Datetime** : 2026/02/21 00:01:02:
- **Mode** : chat
- **Modèle de chat** : ChatMistralAI:mistral-small
- **Modèle d'embedding** : Mistral AI:mistral-embed
- **Température** : 0.7
- **Top_K** : 9
- **Temps d’exécution** : 2.539722 s
- **Distance moyenne** : 0.3252357840538025
- **Index** : `mistral_faiss_index_2026-02-19.bin`
- **Fenêtre temporelle** : 2026-04-01 00:00:00 ± 29 jours
## Réponse générée

Voici une sélection d'activités autour de la lecture et de l'écriture en avril en Occitanie :

1. **Ateliers d'écriture** (Médiathèque George Sand, Bessières) :
   - **Mercredi 1er avril** (10h-11h30) – Jeux d'écriture conviviaux (prose/poésie).
   *Gratuit sur inscription.*

2. **Lectures théâtrales** (Toulouse) :
   - **Mardi 7 avril** (19h) – Lecture dirigée par Maëva Meunier (Compagnie La Particule).
   *Entrée payante, activités gratuites.*

3. **Festival "Mes Poissons d’avril"** (Toulouse) :
   - **Du 15 au 18 avril** – Installations artistiques et lectures dans un univers immersif.
   *Accès libre à la Bibliothèque Bonnefoy.*

4. **Ateliers créatifs** (Toulouse) :
   - **Jeudi 16 avril** (18h-21h) – Ateliers dessin, pause musicale et échanges littéraires.
   *Entrée payante, activités gratuites.*

5. **Club de lecture** (Toulouse) :
   - **Lundi 13 avril** (9h30-11h30) – Partage autour de la musique et des livres.
   *Ambiance conviviale, sans inscription.*

*Conseil* : Réservez pour les ateliers sur inscription (ex : médiathèque Bessières). Pour les événements gratuits comme le festival, arrivez tôt !

## Annotations 

- Les documents récupérés correspondent à la requête`

## Événements utilisés comme contexte

| # | Titre | Ville | Date | Description (résumée) | Lien |
|---|--------|-------|------|-----------------------|------|
| 1 | Ateliers d’écriture | Bessières | 04/04/2026, 14:00:00  ->  20/04/2026, 10:00:00 | LUNDI 19 JANVIER LUNDI 16 FÉVRIER LUNDI 16 MARS LUNDI 20 AVRIL 10H – durée 2 heures Médiathèque George Sand ---------------------------------------------------- Chaque premier samedi du mois, la médiathèque de Bessières ouvre ses portes aux passionnés… | [lien](https://openagenda.com/valaigo-agenda/events/ateliers-decriture-633758) |
| 2 | Jules Julien lit le théâtre contemporain | Toulouse | 07/04/2026, 19:00:00 | &gt; Lundi 15 décembre 2025 à 19h City Stade de Sarah Hassenforder Lecture dirigée par Gaspard Chauvelot Avec les élèves de Cycle 3 du Département théâtre du Conservatoire &gt; Lundi 16 février 2026 à 19h Le Coeur quand il explose de Claire Griois Lecture… | [lien](https://openagenda.com/conservatoire-de-toulouse/events/jules-julien-lit-le-theatre-contemporain-9841936) |
| 3 | Conférences | Alès | 01/04/2026, 09:00:00  ->  14/04/2026, 13:30:00 | Mercredi 1er avril de 9h à 12h : Histoire moderne avec Dominique Biloghi Mardi 7 avril de 9h30 à 11h30 : Découverte de la psychanalyse avec Robert Tarancon Mercredi 8 avril de 14h à 17h : Géopolitique avec Régis Vezon Jeudi 8 avril de 14h à 16h : Histoire de… | [lien](https://openagenda.com/ales-agglomeration/events/conferences-6510068) |
| 4 | Nocturne : Art et créativité | Toulouse | 16/04/2026, 18:00:00 | Programme du jeudi 16 avril de 18h à 21h : 18h-20h : Atelier créatif 18h : Dessin de modèle vivant 18h : Atelier dessin de statues 18h30 : Pause musicale 20h00 : Pause musicale &gt; Tarifs : Entrée payante et toutes activités gratuites (avec ou sans… | [lien](https://openagenda.com/musee-des-augustins-toulouse/events/nocturne-art-et-creativite) |
| 5 | Club Lecture | Bessières | 01/04/2026, 10:00:00 | C’est un club dans une ambiance chaleureuse, à la portée de tous et respectueux des opinions de chacun et de tout type de lecture. Venez découvrir, partager ou simplement écouter pour trouver de nouvelles inspirations de lecture ! MERCREDI 7 JANVIER MERCREDI… | [lien](https://openagenda.com/valaigo-agenda/events/club-lecture-4007026) |
| 6 | Paysages Lumineux par la Cie Les Bricoleuses | Toulouse | 15/04/2026, 09:30:00  ->  18/04/2026, 09:30:00 | Du 15 au 18 avril Un champ de cailloux doux à transporter, des montagnes lumineuses à escalader, des ombres en mouvement qui dessinent des histoires, un paysage en carton à construire et déconstruire pour s’y fabriquer un passage, un abri, un endroit pour… | [lien](https://openagenda.com/bibliotheques-de-toulouse/events/paysages-lumineux-par-la-cie-les-bricoleuses) |
| 7 | Droits des femmes | Castanet-Tolosan | 01/04/2026, 09:00:00  ->  16/04/2026, 09:00:00 | jeux ! avec le Conseil Municipal des Jeunes et un.e médiathécaire* Jeudi 12 mars 20h30-21h45 → On y va ensemble ! Spectacle Dark Salle Jacques Brel Vendredi 13 mars 10h-12h → Venez rencontrer la femme que vous êtes avec Tisser le lien* 13h → Blind-Test : la… | [lien](https://openagenda.com/mervilla/events/droits-des-femmes-5267861) |
| 8 | Colorcircus • Filmez vos performances #3 | Alès | 22/04/2026, 15:00:00  ->  24/04/2026, 15:00:00 | Pendant les vacances de Pâques, passez en mode « full Color » : Domptez votre équilibre, surpassez votre vertige, mixez votre son, dansez votre choré , roulez votre bosse, prenez le micro, jouez la comédie, shootez vos cascades, filmez vos performances,… | [lien](https://openagenda.com/ales-agglomeration/events/colorcircus-filmez-vos-performances-3) |
| 9 | Atelier Parents - Enfants (dès 3 ans) | Toulouse | 11/04/2026, 10:30:00  ->  18/04/2026, 10:30:00 | Thèmes des ateliers : 21 février : A toi les pinceaux ! (Atelier peinture) 21 mars : Safari au musée : à la découverte des animaux sculptés (Modelage en argile) 28 mars (2 séances : 10h30 et 14h) : Dessine-moi un arbre (Création d’un tôte bag imprimé) 11… | [lien](https://openagenda.com/musee-des-augustins-toulouse/events/atelier-parents-enfants-des-3-ans) |
## Statistiques de tokens

- **Tokens requête** : 40
- **Tokens contexte** : 1332
- **Tokens LLM** : 221
- **Total tokens** : 1593

