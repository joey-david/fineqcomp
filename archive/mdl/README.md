# Archived: adaptive MDL allocation

This is the Lagrangian minimum-description-length allocator and its launchers.
It chose a bit width per LoRA row to minimise weight reconstruction error under
a rate budget, and it was the intended headline method of the study.

**It is archived because it does not work.** Measured on Mistral-7B-v0.1 with a
rank-16 LoRA trained on MetaMathQA and evaluated on the full GSM8K test set over
three seeds, it never beat a plain zero-free uniform quantizer at a matched file
rate, and at two bits the uniform code won outright. Allowed to drop rows it
collapsed to zero retained gain; forbidden from dropping, it reproduced the
uniform code exactly. The evidence is in
`results/rmse_mdl_lora_vs_quantized_lora/`.

The code is kept rather than deleted because the negative result is a finding
and it should stay reproducible, and because the file format and the
rate-targeting bisection would be reusable if the allocation objective is ever
replaced with a behaviour-aware one (calibration-NLL sensitivity, Fisher
weighting, or direct loss). The failure was the *objective*, not the machinery.

To run it again:

```bash
PYTHONPATH=src:archive/mdl python -m mdl <run_dir> --config <config> [--no-drop]
```

`fineqcomp.information_scaling` used to call into this module for its minimum
adapter description length; it now sweeps the uniform ladder instead, which is
both cheaper and, per the result above, at least as good.
