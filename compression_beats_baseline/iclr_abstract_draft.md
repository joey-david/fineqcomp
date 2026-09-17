# Compression Filters Behavior: Receiver-Relative information in LLM Finetuning and Reasoning

How much information must a language model receive for a learned behavior to survive? We argue this is not a property of a dataset, an adapter, or a reasoning trace, but of the receiver and the intended use. We make it measurable for fine-tuning by defining the rate of an update as the exact serialized size of a compressed, reloadable LoRA adapter at a fixed level of retained held-out behavior. A model-relative spectrum of the corrections the data requests predicts rate across 22 model–dataset conditions (Spearman ρ = 0.76 (try to improve this), selection-adjusted p < 0.001). _We also show that adapters trained on chain-of-thought rationales attached to the wrong problems lose most of their accuracy, yet compressing them to rank one at one bit recovers above the base model in three model families, 13–59 points above the uncompressed fine-tune_. Though shrinkage restores the base model, the corruption sits in the dominant singular directions, and what survives is a chain-of-thought format prior that in-context examples can reproduce: discarding just the top four directions beats the base model by 8 points. Explicit state interfaces need enough capacity, one stable name per state, and reliable repeated updates. Together, these results recast adaptation and reasoning as communication with a specific receiver.

.... Compression leads to more generalized behavior?
.... Compression undoes overfitting?

^ claims too strong for now
