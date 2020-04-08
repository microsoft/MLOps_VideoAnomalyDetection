# Fine Tuning

Checkout the script `train.py`. You will notice several blocks that check whether we want to perform fine tuning.  If that is the case, the script will look for a registered model in the AML Model Registry, instead of training from scratch.

Note the additional input argument `freeze_layers` to the script.  This allows us to explore which layers get to learn during fine-tuning.
