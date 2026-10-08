# Ablations Jean-Zay

Hello Paul, tout devrait tourner avec `bash scripts/jean_zay/reparameterization.sh`.
Il devrait y en avoir pour ~90h de GPU si tout va bien, sinon le script s'arrete en cas d'erreur.

`bash scripts/jean_zay/reparameterization.sh status` pour verifier l'avancement. Quand c'est fini (`status` devrait afficher « Archive prête »), il faudrait récupèrer `.cache/reparameterization/fineqcomp_reparameterization_results.tar.gz` et me l'envoyer. Garde le dossier `.cache/reparameterization/runs` (environ 10 Go) tant que je n'ai pas vérifié les résultats.

Si besoin, pour tout arrêter : `bash scripts/jean_zay/reparameterization.sh cancel`.
