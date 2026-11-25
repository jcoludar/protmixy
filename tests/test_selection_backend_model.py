"""The MSA selection path must read the MSA Transformer, not the generator arm.

Why this file exists. LRZ job 5929646 (the GPU smoke, 2026-11-17) failed inside
`msat_output.get_current_state_with_ll` with

    ValueError: too many values to unpack (expected 2)   [esm/data.py:265]

on the **strict one-factor variant**: generator ESM-2, selection MSA Transformer
(`PROTMIXY_BACKEND=esm2 PROTMIXY_SELECTION_BACKEND=msa`). The full swap
(selection=esm2) had just passed, so the fault was not in the ESM-2 backend.

The cause was one call. `get_current_state_with_ll` is the published MSA path and
asked for its model with `ModelLoader.get_model()`, which dispatches on
`GENERATOR_BACKEND`. With the generator set to ESM-2 it handed back ESM-2 and
**ESM-2's alphabet**, whose `get_batch_converter()` is a plain `BatchConverter`,
not an `MSABatchConverter`. A plain converter reads `[(label, seq), ...]`; the
MSA path passes `[[(label, seq), ...]]`, so `zip(*raw_batch)` unpacked the
alignment's rows instead of a label and a sequence. The exception therefore fires
only when the two backends disagree, which no unit test and no single-arm run
ever did.

These tests pass no weights. The MSA Transformer's alphabet comes from
`Alphabet.from_architecture`, which reads no checkpoint, and the forward pass is
a stub that returns tensors of the shapes the real module returns.
"""
import os
import sys
import types

import numpy as np
import pytest
import torch
from esm.data import Alphabet, MSABatchConverter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEPTH = 4        # candidate + target + 2 context rows
WIDTH = 12       # alignment columns
N_TOK = 33       # the shared vocabulary size
EMBED = 768      # the MSA Transformer's embedding dimension
LAYER = 12       # the layer msat_output reads


@pytest.fixture
def aln(tmp_path):
    """A 3-row conditioning alignment; the candidate is prepended at call time."""
    rows = [
        (">target", "ACDEFGH-IKLM"),
        (">ctx1", "ACDEFG--IKLM"),
        (">ctx2", "A-DEFGHWIKLM"),
    ]
    path = tmp_path / "cond_context.aln"
    path.write_text("".join(f"{h}\n{s}\n" for h, s in rows))
    return str(path)


class _StubMSATransformer:
    """Returns what `esm.MSATransformer.forward` returns, at the real shapes."""

    def __init__(self):
        self.calls = []

    def __call__(self, tokens, repr_layers=None, return_contacts=False,
                 need_head_weights=False):
        self.calls.append(tuple(tokens.shape))
        batch, depth, length = tokens.shape
        g = torch.Generator().manual_seed(0)
        return {
            "representations": {
                LAYER: torch.rand(batch, depth, length, EMBED, generator=g)
            },
            "logits": torch.rand(batch, depth, length, N_TOK, generator=g),
        }


def _msa_arm():
    """(stub model, real MSA-Transformer alphabet) with no checkpoint read."""
    return _StubMSATransformer(), Alphabet.from_architecture("msa_transformer")


def _esm2_arm():
    """(stub model, real ESM-2 alphabet) with no checkpoint read."""
    return _StubMSATransformer(), Alphabet.from_architecture("ESM-1b")


class TestTheAlphabetsDifferInTheirConverter:
    """The precondition that made the bug possible, stated as a fact."""

    def test_the_msa_alphabet_gives_an_msa_converter(self):
        conv = Alphabet.from_architecture("msa_transformer").get_batch_converter()
        assert isinstance(conv, MSABatchConverter)

    def test_the_esm2_alphabet_does_not(self):
        conv = Alphabet.from_architecture("ESM-1b").get_batch_converter()
        assert not isinstance(conv, MSABatchConverter)


class TestTheMsaSelectionPathUnderAnEsm2Generator:
    """`BACKEND=esm2`, `SELECTION_BACKEND=msa` -- the strict one-factor variant."""

    def test_get_current_state_with_ll_runs_with_an_esm2_generator(
        self, aln, monkeypatch
    ):
        from src.utils import msat_output
        from src.utils.model_loader import ModelLoader

        monkeypatch.setattr(msat_output, "SELECTION_BACKEND", "msa")
        monkeypatch.setattr(ModelLoader, "get_msa_model", staticmethod(_msa_arm))
        monkeypatch.setattr(
            ModelLoader, "get_esm2_model",
            staticmethod(lambda: pytest.fail(
                "the MSA selection path loaded the ESM-2 checkpoint"
            )),
        )
        # The generator arm is ESM-2, so a dispatching get_model() returns ESM-2.
        monkeypatch.setattr(
            ModelLoader, "get_model", staticmethod(lambda backend=None: _esm2_arm())
        )

        out = msat_output.get_current_state("ACDEFGHWIKLM", aln)

        (query_emb, target_emb, query_pos, target_pos,
         pos_entropy, avg_ll) = out
        assert query_emb.shape == (EMBED,)
        assert target_emb.shape == (EMBED,)
        assert query_pos.shape == (WIDTH + 1, EMBED)
        assert target_pos.shape == (WIDTH + 1, EMBED)
        assert pos_entropy.shape == (WIDTH + 1,)
        assert np.isfinite(avg_ll)

    def test_the_selection_pass_sees_the_whole_alignment_depth(
        self, aln, monkeypatch
    ):
        """A plain converter would have collapsed the rows; an MSA one keeps them."""
        from src.utils import msat_output
        from src.utils.model_loader import ModelLoader

        seen = {}

        def _recording_msa_arm():
            model, alphabet = _msa_arm()
            seen["model"] = model
            return model, alphabet

        monkeypatch.setattr(msat_output, "SELECTION_BACKEND", "msa")
        monkeypatch.setattr(
            ModelLoader, "get_msa_model", staticmethod(_recording_msa_arm)
        )
        monkeypatch.setattr(
            ModelLoader, "get_model", staticmethod(lambda backend=None: _esm2_arm())
        )

        msat_output.get_current_state("ACDEFGHWIKLM", aln)

        assert seen["model"].calls == [(1, DEPTH, WIDTH + 1)]


class TestTheReferenceRunIsUnchanged:
    """With both backends on 'msa' the path must behave exactly as before."""

    def test_the_msa_only_run_takes_the_same_route(self, aln, monkeypatch):
        from src.utils import msat_output
        from src.utils.model_loader import ModelLoader

        monkeypatch.setattr(msat_output, "SELECTION_BACKEND", "msa")
        monkeypatch.setattr(ModelLoader, "get_msa_model", staticmethod(_msa_arm))
        monkeypatch.setattr(
            ModelLoader, "get_model", staticmethod(lambda backend=None: _msa_arm())
        )

        out = msat_output.get_current_state("ACDEFGHWIKLM", aln)
        assert out[0].shape == (EMBED,)
