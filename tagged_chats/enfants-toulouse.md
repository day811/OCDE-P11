# Rapport de session RAG
## Résumé de la requête

- **Question** : Je cherche des activités pour les enfants sur toulouse
- **Datetime** : 2026/02/20 01:08:03:
- **Mode** : chat
- **Modèle de chat** : ChatMistralAI:mistral-small
- **Modèle d'embedding** : Mistral AI:mistral-embed
- **Température** : 0.7
- **Top_K** : 5
- **Temps d’exécution** : 3.727654 s
- **Distance moyenne** : 0.34689611196517944
- **Index** : `mistral_faiss_index_2026-02-19.bin`
- **Fenêtre temporelle** : 2026-02-19 00:00:00 ± 30 jours
- **Ville ciblée** : TOULOUSE
## Réponse générée

Voici une sélection d'activités adaptées aux enfants à Toulouse :

1. **Pour les tout-petits (0-3 ans)** :
   - *"Les bibliothécaires racontent..."* : Lectures interactives les samedis (10h-10h30) à la bibliothèque de Micoulaud.
   - *"Les Bébés bouquinent"* : Lectures les mercredis (horaires variables) à la bibliothèque de Reynerie (inscription requise).

2. **Pour les 2-7 ans** :
   - *Le Quai des Petits* : Espace d'exploration ludique et scientifique (réouverture le 21 février, réservation en ligne).

3. **Pour les 8-12 ans** :
   - *Graines de lecteurs* : Club de lecture les mercredis (15h) à la bibliothèque de la Dépêche.

4. **Expositions & événements** :
   - *L'effet Matilda* : Exposition interactive sur les femmes scientifiques (3-14 mars 2026) dans les médiathèques de Fenouillet et Saint-Jory.

**Conseil** : Réservez à l'avance pour les activités nécessitant inscription (ex : *Les Bébés bouquinent*). Profitez des samedis et mercredis pour alterner entre lectures et exploration scientifique !

*Plus d'infos* : Vérifiez les liens fournis pour les détails pratiques.

## Annotations 

- Les documents récupérés correspondent à la requête`
- Le chat a présenté les informations regroupées par age !

## Événements utilisés comme contexte

| # | Titre | Ville | Date | Description (résumée) | Lien |
|---|--------|-------|------|-----------------------|------|
| 1 | Les bibliothécaire racontent... aussi le samedi | Toulouse | 21/02/2026, 10:00:00  ->  21/03/2026, 10:00:00 | Les bibliothécaire racontent... aussi le samedi. Lectures pour les tout petits (0-3 ans). Lectures pour les tout petits (0-3 ans) Samedi 10 janvier samedi 24 janvier samedi 7 février samedi 21 février samedi 7 mars samedi 21 mars 10h à 10h30. Toulouse. 1… | [lien](https://openagenda.com/bibliotheques-de-toulouse/events/les-bibliothecaire-racontent-aussi-le-samedi) |
| 2 | Les Bébés bouquinent | Toulouse | 11/03/2026, 10:30:00 | Les Bébés bouquinent. lectures pour les 0-3 ans. Les Bébés bouquinent Inscription au 05 81 91 79 40 Mercredi 7 janvier Mercredi 21 janvier Mercredi 4 février Mercredi 18 février Mercredi 11 mars Mercredi 25 mars. Toulouse. 37 Avenue de Reynerie, 31100… | [lien](https://openagenda.com/bibliotheques-de-toulouse/events/les-bebes-bouquinent-4817171) |
| 3 | Graines de lecteurs | Toulouse | 18/03/2026, 15:00:00 | Graines de lecteurs. Pour les 8-12 ans.. Mercredi 28 janvier - 15h Mercredi 18 février - 15h Mercredi 18 mars - 15h Club de lecture Graines de lecteurs Pour les 8-12 ans. Toulouse. 3 Avenue de la Dépêche, 31100 Toulouse, France. Haute-Garonne. | [lien](https://openagenda.com/bibliotheques-de-toulouse/events/graines-de-lecteurs-9418860) |
| 4 | Dans les coulisses d’un métier pas comme les autres, instructrice d’astronautes ! | Toulouse | 10/03/2026, 18:30:00 | proposé à à la Maison des chômeurs Partage Faourette de Toulouse le 12 mars 2026 à 12h - plus d'infos L'exposition L'effet Matilda : dans les médiathèques de Fenouillet ( plus d'infos ) et de Saint-Jory ( plus d'infos ), ainsi que des rencontres proposées… | [lien](https://openagenda.com/planlibre/events/dans-les-coulisses-dun-metier-pas-comme-les-autres-instructrice-dastronautes) |
| 5 | S'inspirer du vivant : ateliers des vacances d'hiver | Toulouse | 21/02/2026, 14:30:00  ->  08/03/2026, 14:30:00 | Les propositions s'exposeront dans une création collective au fil des jours, pour montrer au public ces nouvelles créations inspirantes ! Le Quai des Petits revient aussi à partir du 21 février ! Venez en famille au Quai des Petits, l’espace d’exploration… | [lien](https://openagenda.com/planlibre/events/sinspirer-du-vivant) |
## Statistiques de tokens

- **Tokens requête** : 27
- **Tokens contexte** : 674
- **Tokens LLM** : 213
- **Total tokens** : 914

