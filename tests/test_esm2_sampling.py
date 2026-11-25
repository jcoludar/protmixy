"""Tests for ESM-2 candidate sampling.

The behaviour these pin down comes from a measured defect in the MSA-Transformer
path that the ESM-2 arm would inherit.

`top_k_samples()` (src/utils/helpers.py:60-83) keeps a running *joint*
probability as a plain product over every alignment column
(`new_prob = seq_prob * prob_matrix[pos, val]`, :71). Measured over 733
multiplications (work/check_underflow_threshold.py): the product reaches exactly
0.0 whenever the mean per-position probability falls below 0.362 in float64
(0.500 in float32, which is what numpy >= 2.0 would use for this expression).
Once every beam entry is 0.0, `heapq.nlargest` ranks ties by insertion order and
:69 inserts in token-id order, so each candidate's tail fills with token 0 --
'<cls>' -- which `decode_token_to_aa` (:133) renders into the sequence string.
Measured on 734 columns of low-confidence logits
(work/check_topk_tail_content.py): the tail was 344/344 '<cls>', and agreement
with the model's own per-position argmax fell from 99.7 % to 1.5 %.

The MSA Transformer is reconstructing a sequence present in its own input
context, so its per-position probabilities are high and the published runs very
likely stayed above the floor. ESM-2 650M has no MSA conditioning and a UniRef
pseudo-perplexity around 8-10, i.e. a mean per-position probability near 0.1 --
squarely in the underflowing regime. Hence log space, and hence these tests.
"""

import numpy as np
import pytest
import torch

from src.utils import esm2_backend as backend

VALID_TOKENS = set(backend.VALID_AA_TOKENS)
VOCAB_SIZE = 33
CLS_IDX = 0
REAL_ALIGNMENT_LENGTH = 734


def confident_logits(length, vocab=VOCAB_SIZE, seed=0):
    """Logits with a clear per-position winner among the allowed residues."""
    rng = np.random.default_rng(seed)
    logits = torch.full((length + 1, vocab), -20.0)
    for position in range(length + 1):
        for token in backend.VALID_AA_TOKENS:
            logits[position, token] = float(rng.normal())
        winner = int(rng.choice(backend.VALID_AA_TOKENS))
        logits[position, winner] = 10.0
    return logits


def flat_logits(length, vocab=VOCAB_SIZE):
    """Near-uniform logits: the low-confidence regime that underflows."""
    return torch.zeros(length + 1, vocab)


class TestSampleCandidatesContract:
    def test_returns_the_requested_number_of_candidates(self):
        candidates = backend.sample_candidates(confident_logits(20), n_candidates=5)

        assert len(candidates) == 5

    def test_every_candidate_starts_with_the_cls_token(self):
        candidates = backend.sample_candidates(confident_logits(20), n_candidates=5)

        for candidate in candidates:
            assert int(candidate[0]) == CLS_IDX

    def test_candidate_length_is_one_token_per_column_plus_cls(self):
        candidates = backend.sample_candidates(confident_logits(20), n_candidates=3)

        for candidate in candidates:
            assert len(candidate) == 20 + 1

    def test_candidates_are_distinct(self):
        candidates = backend.sample_candidates(confident_logits(20), n_candidates=5)

        rendered = {tuple(int(t) for t in c) for c in candidates}
        assert len(rendered) == 5

    def test_returns_torch_tensors(self):
        candidates = backend.sample_candidates(confident_logits(10), n_candidates=2)

        for candidate in candidates:
            assert isinstance(candidate, torch.Tensor)


class TestSampleCandidatesVocabulary:
    def test_no_candidate_contains_a_disallowed_token(self):
        candidates = backend.sample_candidates(flat_logits(50), n_candidates=5)

        for candidate in candidates:
            for token in candidate[1:]:
                assert int(token) in VALID_TOKENS

    def test_a_dominant_mask_token_is_never_sampled(self):
        logits = flat_logits(30)
        logits[:, 32] = 100.0

        candidates = backend.sample_candidates(logits, n_candidates=3)

        for candidate in candidates:
            assert 32 not in [int(t) for t in candidate[1:]]

    def test_the_cls_token_never_appears_after_position_zero(self):
        """The signature of the underflow defect: tails filled with '<cls>'."""
        candidates = backend.sample_candidates(flat_logits(200), n_candidates=5)

        for candidate in candidates:
            assert CLS_IDX not in [int(t) for t in candidate[1:]]


class TestSampleCandidatesNumerics:
    def test_top_candidate_is_the_per_position_argmax_of_allowed_tokens(self):
        """Positions are independent, so the most likely sequence is the argmax."""
        logits = confident_logits(40)
        expected = torch.argmax(backend.restrict_logits_to_valid_aa(logits), dim=-1)

        best = backend.sample_candidates(logits, n_candidates=3)[0]

        assert [int(t) for t in best[1:]] == [int(t) for t in expected[1:]]

    def test_survives_a_real_alignment_length_at_low_confidence(self):
        """733 multiplications of ~1/21 underflow any product in probability space."""
        logits = flat_logits(REAL_ALIGNMENT_LENGTH)

        candidates = backend.sample_candidates(logits, n_candidates=5)

        assert len(candidates) == 5
        for candidate in candidates:
            assert len(candidate) == REAL_ALIGNMENT_LENGTH + 1
            assert all(int(t) in VALID_TOKENS for t in candidate[1:])

    def test_tail_still_tracks_the_argmax_at_a_real_alignment_length(self):
        """The measured regression: agreement collapsed to 1.5 % past position 390."""
        logits = confident_logits(REAL_ALIGNMENT_LENGTH)
        expected = torch.argmax(backend.restrict_logits_to_valid_aa(logits), dim=-1)

        best = backend.sample_candidates(logits, n_candidates=5)[0]

        tail = slice(400, REAL_ALIGNMENT_LENGTH + 1)
        agreement = sum(
            int(a) == int(b) for a, b in zip(best[tail], expected[tail])
        ) / len(range(400, REAL_ALIGNMENT_LENGTH + 1))
        assert agreement == pytest.approx(1.0)

    def test_scores_are_finite_log_probabilities_at_a_real_alignment_length(self):
        logits = flat_logits(REAL_ALIGNMENT_LENGTH)

        _, scores = backend.sample_candidates(
            logits, n_candidates=5, return_scores=True
        )

        for score in scores:
            assert np.isfinite(score)
            assert score < 0.0

    def test_candidates_are_ordered_by_descending_log_probability(self):
        logits = confident_logits(60)

        _, scores = backend.sample_candidates(
            logits, n_candidates=5, return_scores=True
        )

        assert scores == sorted(scores, reverse=True)
