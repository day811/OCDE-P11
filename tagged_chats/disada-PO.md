# Rapport de session RAG
## Résumé de la requête

- **Question** : Sais tu ou se trouve le concert de DISADA et DJ VKO dans les pyrénees orientales
- **Datetime** : 2026/02/19 20:01:02:
- **Mode** : chat
- **Modèle de chat** : ChatMistralAI:mistral-small
- **Modèle d'embedding** : Mistral AI:mistral-embed
- **Température** : 0.7
- **Top_K** : 9
- **Temps d’exécution** : 2.01008 s
- **Distance moyenne** : 0.3767637759447098
- **Index** : `mistral_faiss_index_2026-02-19.bin`
- **Fenêtre temporelle** : 2026-02-19 00:00:00 ± 30 jours
- **Département ciblé** : Pyrénées-Orientales
## Réponse générée

Le concert de **Disada et DJ VKO** (reggae, ragga, dancehall) a lieu **au Nautilus**, situé **20 rue Jules Verne, 66100 Perpignan** (Pyrénées-Orientales).

**Infos pratiques** :
- **Date/heure** : 21h00 à 2h00
- **Tarif** : 5€ (2€ pour les adhérents)
- **Ambiance** : Bar sur place, ambiance festive et dansante, hommage à l'héritage reggae.

**Recommandation** :
Si tu aimes les sons reggae locaux et une ambiance conviviale, c'est un bon plan. Pense à vérifier les horaires et les consignes éventuelles avant de te déplacer.

*Bonus* : Si tu cherches une alternative plus intimiste, le concert commenté de **Nilco Moogin** (guitare/voix) au même lieu pourrait aussi t'intéresser.

## Annotations 

- Un des documents récupérés correspond bien à la requête`
- Le filtre de département est efficace pour récupérer le bon document (sans le département, la bonne réponse n'est pas trouvée)

## Événements utilisés comme contexte

| # | Titre | Ville | Date | Description (résumée) | Lien |
|---|--------|-------|------|-----------------------|------|
| 1 | CONCERT DE DISADA et DJ VKO | Perpignan | 28/02/2026, 21:00:00  ->  01/03/2026, 00:00:00 | CONCERT DE DISADA et DJ VKO. CONCERT RAGGA, REGGAE, DANCEHALL, DISADA + VKO DJ , ENTREE 5€ et 2€ pour les ADHERENTS AU NAUTILUS 20 RUE JULES VERNE 66100 PERPIGNAN. CONCERT DE DISADA + DJ VKO CONCERT REGGAE, RAGGA, DANCEHALL 21H00 à 2H00 AU NAUTILUS 20 RUE… | [lien](https://openagenda.com/le-nautilus-perpignan/events/concert-de-disada-et-dj-vko) |
| 2 | ANNULE ! Nathan Roche + Harlan T. Bobo + Elgun Stone | Perpignan | 11/03/2026, 19:00:00 | Guitare, voix, récit : un moment intimiste, sincère, fait pour écouter et voyager. https://youtu.be/ML88Aftz1zk?si=9ITMHNx8KDNrjPaW. 10 €. Perpignan. 20 rue jules verne 66000 Perpignan. Pyrénées-Orientales. | [lien](https://openagenda.com/le-nautilus-perpignan/events/nathan-roche-harlan-t-bobo-elgun-stone) |
| 3 | Fête de l'Ours de Saint Laurent de Cerdans | Saint-Laurent-de-Cerdans | 01/03/2026, 10:00:00 | Fête participative dont il est vivement conseillé de prendre des informations en amont et de respecter les consignes de sécurité.. Saint-Laurent-de-Cerdans. Rue de la Sort, Saint-Laurent-de-Cerdans. Pyrénées-Orientales. | [lien](https://openagenda.com/pci-occitanie/events/fete-de-lours-6367377) |
| 4 | Exposition peintures ELIKIA act.3 de Sarah PAUL | Perpignan | 19/02/2026, 10:00:00  ->  19/02/2026, 18:30:00 | 19h30 Concert Commenté de @Nilco Moogin Dans une forme acoustique, sans fards ni artifices, face au public textes en main, l'artiste compositeur, guitariste, interprète nous invite à un temps d'échange, directement de l'artiste au "consommacteur". Un moment… | [lien](https://openagenda.com/tac-troupuscule/events/vernissage-exposition-elikia-act3-de-sarah-paul-concert-commente-de-nilco-moogin) |
## Statistiques de tokens

- **Tokens requête** : 36
- **Tokens contexte** : 458
- **Tokens LLM** : 127
- **Total tokens** : 621

