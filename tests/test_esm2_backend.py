"""Tests for the ESM-2 generator arm.

These pin the four things that would silently corrupt a pathway if the
generator were swapped naively, each found by reading the MSA-Transformer code
it has to stay compatible with:

1. ESM-2's alphabet sets append_eos=True while the MSA Transformer's does not.
   protmixy assumes tokens are [cls] + residues of length L+1
   (src/utils/helpers.py:443, src/msat_beam_evolver.py:68) and decodes with
   `seq_tokens[1:]` (src/utils/helpers.py:133), so a trailing EOS would be
   decoded into the sequence string and shift every position index.
2. `top_k_samples` (src/utils/helpers.py:60-83) ranges over all 33 tokens, so
   B/Z/X/U/O/'.'/<mask>/<pad>/<eos> are reachable. protmixy already defines the
   correct allowed set as `valid_aa_vals` (src/utils/helpers.py:32-33) but never
   references it. The restriction must filter, not post-patch.
3. ESM-2 was pretrained on ungapped UniRef, so gapped alignment columns are out
   of distribution. The query is run ungapped and gaps re-inserted by column
   map; the round trip must be exact.
4. Degenerate iterative filling is the main scientific risk, so per-step
   diversity and pseudo-perplexity are instrumented rather than assumed.
"""

import numpy as np
import pytest
import torch

from src.utils import esm2_backend as backend

# The 20 canonical residues + gap, as protmixy already lists them.
VALID_TOKENS = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 30
]
VOCAB_SIZE = 33
CLS_IDX = 0
MASK_IDX = 32


class TestEncodeQuery:
    def test_encodes_to_cls_plus_one_token_per_residue(self):
        tokens = backend.encode_query("LAG")

        assert tokens.tolist() == [CLS_IDX, 4, 5, 6]

    def test_appends_no_eos_token(self):
        """ESM-2's own alphabet would append EOS (id 2); protmixy's layout must not."""
        tokens = backend.encode_query("LAG")

        assert 2 not in tokens.tolist()
        assert len(tokens) == len("LAG") + 1

    def test_length_matches_the_mask_length_protmixy_builds(self):
        """helpers.py:443 builds `torch.ones(size_align_seq + 1)`; lengths must agree."""
        sequence = "LAGVSERTID"

        tokens = backend.encode_query(sequence)
        mask = torch.ones(len(sequence) + 1, dtype=torch.uint8)

        assert len(tokens) == len(mask)

    def test_encodes_gap_characters_to_the_gap_token(self):
        tokens = backend.encode_query("L-G")

        assert tokens.tolist() == [CLS_IDX, 4, 30, 6]


class TestRestrictLogits:
    def test_disallowed_tokens_become_negative_infinity(self):
        logits = torch.zeros(5, VOCAB_SIZE)

        restricted = backend.restrict_logits_to_valid_aa(logits)

        disallowed = [t for t in range(VOCAB_SIZE) if t not in VALID_TOKENS]
        assert torch.isneginf(restricted[:, disallowed]).all()

    def test_allowed_tokens_keep_their_values(self):
        logits = torch.arange(VOCAB_SIZE, dtype=torch.float32).repeat(5, 1)

        restricted = backend.restrict_logits_to_valid_aa(logits)

        for token in VALID_TOKENS:
            assert torch.equal(restricted[:, token], logits[:, token])

    def test_does_not_mutate_the_input(self):
        logits = torch.zeros(2, VOCAB_SIZE)

        backend.restrict_logits_to_valid_aa(logits)

        assert torch.equal(logits, torch.zeros(2, VOCAB_SIZE))

    def test_softmax_of_restricted_logits_puts_zero_mass_on_disallowed_tokens(self):
        """The filter has to survive the softmax that follows it in helpers.py:101."""
        logits = torch.zeros(4, VOCAB_SIZE)

        probs = torch.softmax(backend.restrict_logits_to_valid_aa(logits), dim=-1)

        disallowed = [t for t in range(VOCAB_SIZE) if t not in VALID_TOKENS]
        assert probs[:, disallowed].sum().item() == pytest.approx(0.0)
        assert probs.sum(dim=-1).numpy() == pytest.approx(np.ones(4))

    def test_mask_token_is_never_sampleable(self):
        """<mask> (32) scoring highest must not be allowed to fill a position."""
        logits = torch.full((3, VOCAB_SIZE), -10.0)
        logits[:, MASK_IDX] = 100.0

        restricted = backend.restrict_logits_to_valid_aa(logits)

        assert torch.argmax(restricted, dim=-1).tolist() != [MASK_IDX] * 3
        assert all(int(t) in VALID_TOKENS for t in torch.argmax(restricted, dim=-1))


class TestGapColumnMap:
    @pytest.mark.parametrize(
        "aligned",
        ["LAG", "--LAG--", "L-A-G", "---", "L", ""],
    )
    def test_ungap_then_regap_round_trips_exactly(self, aligned):
        ungapped, column_map = backend.ungap(aligned)

        assert backend.regap(ungapped, column_map, len(aligned)) == aligned

    def test_ungap_removes_every_gap(self):
        ungapped, _ = backend.ungap("--LA-G-")

        assert ungapped == "LAG"

    def test_column_map_gives_the_alignment_column_of_each_residue(self):
        _, column_map = backend.ungap("-LA--G")

        assert column_map == [1, 2, 5]

    def test_regap_places_substituted_residues_in_their_original_columns(self):
        """A filled sequence must land back on the columns it came from."""
        _, column_map = backend.ungap("-LA--G")

        assert backend.regap("WWW", column_map, 6) == "-WW--W"

    def test_regap_rejects_a_length_mismatch(self):
        _, column_map = backend.ungap("-LA--G")

        with pytest.raises(ValueError):
            backend.regap("WW", column_map, 6)


class TestInstrumentation:
    def test_identical_sequences_have_zero_diversity(self):
        assert backend.step_diversity(["LAG", "LAG", "LAG"]) == 0.0

    def test_fully_distinct_sequences_have_diversity_one(self):
        assert backend.step_diversity(["AAA", "LLL"]) == 1.0

    def test_diversity_is_the_mean_pairwise_hamming_fraction(self):
        # LAG vs LAA differ at 1 of 3; LAG vs LLL at 2 of 3; LAA vs LLL at 2 of 3.
        expected = (1 / 3 + 2 / 3 + 2 / 3) / 3

        assert backend.step_diversity(["LAG", "LAA", "LLL"]) == pytest.approx(expected)

    def test_a_single_sequence_has_no_diversity_to_report(self):
        assert backend.step_diversity(["LAG"]) == 0.0

    def test_perplexity_of_a_confident_uniform_prediction_is_one(self):
        log_probs = torch.zeros(4)  # log(1.0)

        assert backend.pseudo_perplexity(log_probs) == pytest.approx(1.0)

    def test_perplexity_of_a_uniform_distribution_over_21_residues_is_21(self):
        log_probs = torch.full((6,), float(np.log(1 / 21)))

        assert backend.pseudo_perplexity(log_probs) == pytest.approx(21.0)

    def test_perplexity_rises_as_the_model_grows_less_certain(self):
        confident = torch.full((5,), float(np.log(0.9)))
        unsure = torch.full((5,), float(np.log(0.2)))

        assert backend.pseudo_perplexity(confident) < backend.pseudo_perplexity(unsure)
