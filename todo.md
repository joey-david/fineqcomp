[ ] - passe-bande SVD, corrupted vs uncorrupted
[ ] - train a corrupted adapter on GSM8K (e.g. Mistral)
[ ] - train uncorrupted
[ ] - run baseline
test all on other datasets and see generalizations

## Jean-Zay short-job routing

`qos_gpu_h100-dev` is a QoS, not a partition. For an H100 job that is expected
to finish within two hours, use `--partition=gpu_p6`, `--constraint=h100`,
`--qos=qos_gpu_h100-dev`, and `--time` no greater than `02:00:00`. The dev QoS
has a two-hour wall-time cap and a 32-GPU per-job/user limit; its priority is
intended for short development and smoke jobs. Longer jobs keep
`qos_gpu_h100-t3`.

`scripts/jean_zay_submit.sh` now reads a batch script's declared `#SBATCH
--time` and routes qualifying H100/A100 jobs to the matching `*-dev` QoS. Set
`JZ_AUTO_DEV=0` to disable this for an exceptional submission, or set
`JZ_EXPECTED_TIME=HH:MM:SS` when the declared script time is not the expected
runtime. The wrapper changes the QoS only; it does not change the requested
partition or GPU constraint.
