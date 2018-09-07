# scone
Code coming soon!

## Data
All data is available in the `data/` directory, whose subdirectories correspond
to the three domains in SCONE (Alchemy, Scene, and Tangrams). Each subdirectory
contains JSON files storing the processed train, dev, and test data. We also
provide a script that generates action sequences for supervised learning
(`action_sequences.py`).

Structure of the JSON files:
* One line per file, containing a single JSON object with a list of examples.
* Each example is a dictionary with a few properties:
    * `identifier`: A unique identifier for the interaction.
    * `initial_env`: The initial world state for the interaction.
    * `utterances`: Sequences of instructions. The `instruction` is the natural language
       instruction; `after_env` is the gold environment after executing the instruction
       according to the original data; and `actions` is a sequence of low-level actions
       generated from `action_sequences.py` and only used during the supervised learning
       setting.

## Model
See `model/`. There are a few necessary steps to start running the code.

### Requirements
* DyNet (see [install guide](http://dynet.readthedocs.io/en/latest/python.html) for Python)
* Crayon (see [install guide](https://github.com/clab/dynet/tree/master/examples/tensorboard) for use with DyNet)
* Docker

If you don't want to use Crayon for logging, you can disable it in the code.
