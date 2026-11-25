"""
Model loader for the two generator arms.

Singletons so each checkpoint is read once per process:

* the **MSA Transformer** (`esm_msa1_t12_100M_UR50S`) -- the published method;
* **ESM-2 650M** (`esm2_t33_650M_UR50D`) -- the single-sequence baseline for
  Reviewer 2's major comment #1.

Two things this module exists to get right.

**The alphabet is reachable without any checkpoint.** `Alphabet.from_architecture`
reads no weights (esm/data.py:143-174), and the MSA Transformer's and ESM-2's
alphabets are identical on every id this repo uses -- 4-23 (canonical residues),
30 ('-'), 32 ('<mask>'), plus cls 0, pad 1, eos 2. They differ only in
`append_eos`. So `get_alphabet()` can tokenise an alignment for either arm
without loading a 2.4 GiB file, which is what lets the ESM-2 arm run on a
machine that has no MSA-Transformer checkpoint at all.

**`token_dropout` must be off for ESM-2.** `esm/esm2.py:88-92` rescales the input
embeddings by (1 - 0.15*0.8) / (1 - observed mask ratio). The observed ratio is
counted per forward pass, and this loop masks 0 positions at iteration 1 and
about P_MASK afterwards, so the rescale would vary from step to step (0.88 at
iteration 1, ~0.926 later). It also weakens the BERT-style copying of unmasked
tokens that keeps a pathway from wandering at columns nobody masked. It is a
plain attribute read at `esm2.py:86`, so assigning after load is enough -- but
the assignment is checked rather than assumed.
"""
import torch
from esm import pretrained
from esm.data import Alphabet
from config.settings import (
    DEVICE,
    GENERATOR_MODEL_NAME,
    GENERATOR_BACKEND,
    ESM2_MODEL_NAME,
    ESM2_TOKEN_DROPOUT,
)

#: The architecture whose alphabet protmixy's token layout is built on. Both
#: arms use the same ids; only `append_eos` differs, and the ESM-2 arm never
#: routes through ESM-2's batch converter for exactly that reason.
ALIGNMENT_ARCHITECTURE = "msa_transformer"


class ModelLoader:
    """Cached access to the two checkpoints and to the weightless alphabet."""

    _model = None
    _alphabet = None
    _esm2_model = None
    _esm2_alphabet = None
    _shared_alphabet = None

    @staticmethod
    def get_alphabet():
        """The alignment alphabet, with NO checkpoint read.

        Its `get_batch_converter()` is the MSA converter, which emits the
        `[cls] + one token per column` layout (length L+1) that this repo
        assumes everywhere -- `helpers.py:443`, `helpers.py:133`,
        `msat_beam_evolver.py:68`.
        """
        if ModelLoader._shared_alphabet is None:
            ModelLoader._shared_alphabet = Alphabet.from_architecture(
                ALIGNMENT_ARCHITECTURE
            )
        return ModelLoader._shared_alphabet

    @staticmethod
    def get_msa_model():
        """The MSA Transformer, loaded on first use."""
        if ModelLoader._model is None or ModelLoader._alphabet is None:
            print(f"Loading MSA-Transformer model: {GENERATOR_MODEL_NAME}...")
            ModelLoader._model, ModelLoader._alphabet = pretrained.load_model_and_alphabet(
                GENERATOR_MODEL_NAME
            )
            ModelLoader._model.eval()
            ModelLoader._model.to(DEVICE)
            print(f"Model loaded successfully on device: {DEVICE}")

        return ModelLoader._model, ModelLoader._alphabet

    @staticmethod
    def get_esm2_model():
        """ESM-2, loaded on first use, with `token_dropout` forced off."""
        if ModelLoader._esm2_model is None or ModelLoader._esm2_alphabet is None:
            print(f"Loading ESM-2 model: {ESM2_MODEL_NAME}...")
            model, alphabet = pretrained.load_model_and_alphabet(ESM2_MODEL_NAME)

            model.token_dropout = ESM2_TOKEN_DROPOUT
            if model.token_dropout is not ESM2_TOKEN_DROPOUT:
                # Report the assertion and the values. The cause is the
                # reader's to establish: a guard that volunteers a diagnosis it
                # has not tested sends the next person hunting the wrong bug,
                # which is what the benchmark array's skip message did on
                # 2026-11-19.
                raise RuntimeError(
                    f"token_dropout did not take on the loaded ESM-2 module: "
                    f"assigned {ESM2_TOKEN_DROPOUT!r}, reads back "
                    f"{model.token_dropout!r} on {ESM2_MODEL_NAME}"
                )

            model.eval()
            model.to(DEVICE)
            ModelLoader._esm2_model, ModelLoader._esm2_alphabet = model, alphabet
            print(
                f"ESM-2 loaded on device: {DEVICE}, "
                f"num_layers={model.num_layers}, embed_dim={model.embed_dim}, "
                f"token_dropout={model.token_dropout}"
            )

        return ModelLoader._esm2_model, ModelLoader._esm2_alphabet

    @staticmethod
    def get_model(backend=None):
        """The generator for this arm. Defaults to `GENERATOR_BACKEND`."""
        backend = (backend or GENERATOR_BACKEND).lower()
        if backend == "esm2":
            return ModelLoader.get_esm2_model()
        return ModelLoader.get_msa_model()

    @staticmethod
    def esm2_repr_layer():
        """The final layer of the loaded ESM-2, read from the module.

        Not hard-coded: 33 at 650M, 12 at 35M, 6 at 8M, and the smoke runs on
        the small ones.
        """
        model, _ = ModelLoader.get_esm2_model()
        return model.num_layers

    @staticmethod
    def esm2_embed_dim():
        model, _ = ModelLoader.get_esm2_model()
        return model.embed_dim
