# Ablations Jean-Zay : mode d'emploi

Salut Paul, et merci ! Tout passe par une seule commande. Elle vérifie le compte, les GPU et l'accès aux modèles avant de soumettre quoi que ce soit, puis enchaîne tous les jobs toute seule. Rien ne touche à mes fichiers : tout s'écrit dans ton clone.

1. Clone le dépôt dans ton `$WORK` depuis une frontale : `cd $WORK && git clone https://github.com/joey-david/fineqcomp.git && cd fineqcomp`. Si GitHub ne répond pas depuis Jean-Zay, clone sur ta machine puis copie le dossier avec `rsync`.

2. Vérifie ton projet avec `idrproj`. Si le projet par défaut n'est pas celui qui a des heures GPU, fais `export FQ_PROJECT=xxx` avec ses trois lettres.

3. Lance `bash scripts/jean_zay/reparameterization.sh`. En une minute environ, le script choisit les H100 (ou les A100 si ton projet n'a pas d'heures H100), soumet la chaîne et rend la main. Si un modèle de `$DSDIR` t'est interdit (Llama, typiquement), il le signale et lance le reste sans lui.

4. Suis l'avancement quand tu veux avec `bash scripts/jean_zay/reparameterization.sh status`. Il faut compter une demi-journée à une journée selon la file d'attente, pour environ 65 heures H100 en tout. Un premier job court teste toute la chaîne sur GPU ; s'il échoue, tout le reste est annulé avant de consommer des heures.

5. Quand `status` affiche « Archive prête », récupère `.cache/reparameterization/fineqcomp_reparameterization_results.tar.gz` (quelques Mo) avec `scp` et envoie-le-moi. Garde le dossier `.cache/reparameterization/runs` (environ 10 Go) tant que je n'ai pas vérifié les résultats.

6. Si quelque chose casse ou dépasse son temps, relance simplement la commande du point 3 : elle ne soumet que ce qui n'est pas fini. L'archive est produite même en cas d'échec, avec les journaux, donc envoie-la-moi aussi dans ce cas. Pour tout arrêter : `bash scripts/jean_zay/reparameterization.sh cancel`.
