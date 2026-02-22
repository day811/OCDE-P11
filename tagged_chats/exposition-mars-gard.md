# Rapport de session RAG
## Résumé de la requête

- **Question** : je cherche des **expostions** dans le gard en mars
- **Datetime** : 2026/02/21 01:04:03:
- **Mode** : chat
- **Modèle de chat** : ChatMistralAI:mistral-small
- **Modèle d'embedding** : Mistral AI:mistral-embed
- **Température** : 0.7
- **Top_K** : 5
- **Temps d’exécution** : 2.886035 s
- **Distance moyenne** : 0.3469139814376831
- **Index** : `mistral_faiss_index_2026-02-19.bin`
- **Fenêtre temporelle** : 2026-03-01 00:00:00 ± 30 jours
- **Département ciblé** : Gard
## Réponse générée

Voici deux expositions à ne pas manquer dans le Gard en mars :

1. **"Lumières Nocturnes"** – Découvrez les clichés de paysages, orages et animaux de David Guerre.
   **Où ?** 574 route d’Uzès, Alès.
   **Quand ?** Vernissage le 5 mars à 18h30.

2. **"Graines d'avenir"** – Une exposition gratuite à Biosphera (inscription conseillée).
   **Où ?** 155 rue du Faubourg de Rochebelle, Alès.
   **Contact** : 04 66 52 61 38 ou contact@cpiegard.fr.

*Conseil* : Réservez à l’avance pour les deux expositions, surtout pour "Graines d’avenir".

## Annotations 

- 1 faute d'orthographe dans expositions
- 2/5 de documents récupérés correspondent à la requête, filtrés correctement dans la réponse sur le mot exposition orthographe corrigé
- en corrigeant l'orthographe dans la requete, on obtient 1 réponse correcte de plus

## Événements utilisés comme contexte

| # | Titre | Ville | Date | Description (résumée) | Lien |
|---|--------|-------|------|-----------------------|------|
| 1 | Lotos du mois de mars 2026 | Alès | 01/03/2026, 14:00:00  ->  31/03/2026, 16:00:00 | 06 61 07 89 71) 15h, Cruviers-Lascours, salle du Parc (Model’s) 15h, Salindres, salle Becmil (APE) 15h, Sénéchas, salle polyvalente (Comité de Restauration de l’église de Sénéchas) 16h, La Grand-Combe, rue du Gouffre (Les Joyeux Mineurs) Mardi 31 mars : 16h,… | [lien](https://openagenda.com/ales-agglomeration/events/lotos-du-mois-de-mars-2026) |
| 2 | Les Mercredis du Pôle " Décore ton jardin " | Alès | 18/03/2026, 14:00:00 | Les Mercredis du Pôle " Décore ton jardin ". Les Mercredis au Jardin. Atelier " Décore ton jardin " Mercredi 18 mars 2026 de 14h à 16h Après une visite au jardin, chaque participant laissera libre cours à son imagination pour concevoir une décoration unique… | [lien](https://openagenda.com/ales-agglomeration/events/les-mercredis-du-pole-decore-ton-jardin) |
| 3 | Exposition “Lumières Nocturnes” | Alès | 02/03/2026, 09:00:00  ->  31/03/2026, 09:00:00 | Exposition “Lumières Nocturnes”. Découvrez l’univers du photographe David Guerre. Clichés de paysages, traque orageuse, photographie animalière, … Les thèmes sont variés.. Vernissage jeudi 5 mars à 18h30. Alès. 574 route d’Uzès, 30100 Alès. Gard. | [lien](https://openagenda.com/ales-agglomeration/events/exposition-lumieres-nocturnes) |
| 4 | Echanges techniques : diversifier sa production | Le Vigan | 05/03/2026, 14:00:00 | Pour faciliter l'organisation il est préférable de s'inscrire.. Le Vigan. 30 route du Pont de la Croix, Le Vigan. Gard. | [lien](https://openagenda.com/toutes-les-chambres-agriculture-agregees/events/echanges-techniques-diversifier-sa-production) |
| 5 | La course aux graines | Alès | 04/03/2026, 14:00:00 | Pour compléter cette animation, n'hésitez pas à visiter l'exposition "Graines d'avenir" à Biosphera !. Gratuit. Inscription conseillée au 04 66 52 61 38 ou contact@cpiegard.fr. Alès. 155 rue du faubourg de Rochebelle 30100 Alès. Gard. | [lien](https://openagenda.com/ales-agglomeration/events/la-course-aux-graines) |
## Statistiques de tokens

- **Tokens requête** : 27
- **Tokens contexte** : 741
- **Tokens LLM** : 115
- **Total tokens** : 883

