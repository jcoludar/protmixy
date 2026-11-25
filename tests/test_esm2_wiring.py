"""Tests for the parts that put ESM-2 inside protmixy's loop.

`test_esm2_backend.py` and `test_esm2_sampling.py` pin the pieces. These pin the
joins, which is where this port can fail silently rather than loudly:

1. **The fast sampler must be the reference sampler.** `sample_candidates` is
   the specification -- exact, readable, and quadratic in alignment length
   because it rebuilds every prefix list at every column. `sample_candidates_fast`
   is what actually runs 891 times per pathway. They are checked against each
   other rather than each against a hand-written expectation.
2. **The gap column map must never drop a masked column.** `mask_sequence`
   (src/utils/helpers.py:190) writes `<mask>` over selected columns, including
   columns whose current residue is a gap -- and such a column was selected
   precisely because the target has a residue there. Dropping it would make the
   pathway unable to fill it, silently.
3. **A dropped column must come back as a gap**, so every alignment index
   downstream -- `tgt_pos_diff_aa`'s `pos+1`, the entropy percentile, the
   position-wise cosine distance -- keeps its meaning.
4. **The three forward-pass shapes** that the port was written against by
   reading fair-esm's source, and that §8 of the baseline README lists as never
   observed.
"""

import numpy as np
import pytest
import torch

from src.utils import esm2_backend as backend

VALID_TOKENS = set(backend.VALID_AA_TOKENS)
VOCAB_SIZE = backend.VOCAB_SIZE
CLS = backend.CLS_TOKEN
GAP = backend.GAP_TOKEN
MASK = backend.MASK_TOKEN


def random_logits(length, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(length + 1, VOCAB_SIZE, generator=generator)


class TestFastSamplerMatchesTheReference:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_fast_sampler_matches_the_reference_implementation(self, seed):
        logits = random_logits(40, seed=seed)

        reference = backend.sample_candidates(logits, n_candidates=5)
        fast = backend.sample_candidates_fast(logits, n_candidates=5)

        assert [c.tolist() for c in fast] == [c.tolist() for c in reference]

    def test_fast_sampler_matches_the_reference_scores(self):
        logits = random_logits(25, seed=7)

        _, reference = backend.sample_candidates(
            logits, n_candidates=4, return_scores=True)
        _, fast = backend.sample_candidates_fast(
            logits, n_candidates=4, return_scores=True)

        assert fast == pytest.approx(reference)

    def test_fast_sampler_matches_at_a_real_alignment_width(self):
        """733 columns is the committed conditioning context's width."""
        logits = random_logits(200, seed=11)

        reference = backend.sample_candidates(logits, n_candidates=3)
        fast = backend.sample_candidates_fast(logits, n_candidates=3)

        assert [c.tolist() for c in fast] == [c.tolist() for c in reference]

    def test_fast_sampler_emits_no_disallowed_token(self):
        logits = torch.zeros(300 + 1, VOCAB_SIZE)
        logits[:, MASK] = 50.0

        for candidate in backend.sample_candidates_fast(logits, n_candidates=5):
            assert all(int(t) in VALID_TOKENS for t in candidate[1:])
            assert CLS not in [int(t) for t in candidate[1:]]


class TestColumnIndexMap:
    def test_unmasked_gap_columns_are_dropped(self):
        tokens = torch.tensor([CLS, 4, GAP, 5, GAP, 6])

        model_tokens, col_to_input = backend.column_index_map(tokens)

        assert model_tokens.tolist() == [CLS, 4, 5, 6]
        assert col_to_input == [0, 1, -1, 2, -1, 3]

    def test_a_masked_column_is_kept_even_where_the_residue_was_a_gap(self):
        """mask_sequence writes <mask> over a gap column whose target differs."""
        tokens = torch.tensor([CLS, 4, MASK, GAP, 5])

        model_tokens, col_to_input = backend.column_index_map(tokens)

        assert MASK in model_tokens.tolist()
        assert col_to_input[2] == 2
        assert col_to_input[3] == -1

    def test_the_cls_slot_is_required(self):
        with pytest.raises(ValueError):
            backend.column_index_map(torch.tensor([4, 5, 6]))

    def test_an_all_gap_row_leaves_only_the_cls_token(self):
        model_tokens, col_to_input = backend.column_index_map(
            torch.tensor([CLS, GAP, GAP, GAP]))

        assert model_tokens.tolist() == [CLS]
        assert col_to_input == [0, -1, -1, -1]

    def test_the_map_is_as_long_as_the_alignment_row(self):
        tokens = torch.tensor([CLS] + [GAP, 4] * 30)

        _, col_to_input = backend.column_index_map(tokens)

        assert len(col_to_input) == len(tokens)


class TestScatterLogitsToColumns:
    def test_kept_columns_get_their_model_row_back(self):
        model_logits = torch.arange(4 * VOCAB_SIZE, dtype=torch.float32).reshape(4, VOCAB_SIZE)
        col_to_input = [0, 1, -1, 2, 3]

        columns = backend.scatter_logits_to_columns(model_logits, col_to_input)

        assert torch.equal(columns[0], model_logits[0])
        assert torch.equal(columns[1], model_logits[1])
        assert torch.equal(columns[3], model_logits[2])
        assert torch.equal(columns[4], model_logits[3])

    def test_a_dropped_column_is_pinned_to_the_gap_token(self):
        model_logits = torch.zeros(2, VOCAB_SIZE)
        col_to_input = [0, -1, 1]

        columns = backend.scatter_logits_to_columns(model_logits, col_to_input)

        assert int(torch.argmax(columns[1])) == GAP
        others = [t for t in range(VOCAB_SIZE) if t != GAP]
        assert torch.isneginf(columns[1, others]).all()

    def test_round_trip_reproduces_the_gap_pattern_exactly(self):
        """An identity model must give back the row it was handed."""
        row = "MA--KY-C"
        tokens = backend.encode_query(row)
        model_tokens, col_to_input = backend.column_index_map(tokens)

        identity = torch.full((len(model_tokens), VOCAB_SIZE), -30.0)
        for index, token in enumerate(model_tokens):
            identity[index, int(token)] = 10.0

        columns = backend.scatter_logits_to_columns(identity, col_to_input)
        rebuilt = backend.sample_candidates_fast(columns, n_candidates=1)[0]

        assert backend.decode(rebuilt) == row

    def test_rejects_a_map_pointing_past_the_model_output(self):
        with pytest.raises(ValueError):
            backend.scatter_logits_to_columns(torch.zeros(2, VOCAB_SIZE), [0, 1, 2])

    def test_rejects_unsliced_msa_transformer_logits(self):
        """The MSA Transformer's logits are (1, N, L, 33); ours must be (T, 33)."""
        with pytest.raises(ValueError):
            backend.scatter_logits_to_columns(torch.zeros(1, 3, 5, VOCAB_SIZE), [0, 1])


class TestFillCandidates:
    def test_the_argmax_is_the_top_candidate_and_dedups_away(self):
        """Positions are independent, so the argmax IS the top-1 joint sequence.

        `msa_query_sample` appends it anyway and lets the set dedup it, which is
        why A4 §5 counts n_cand as 5 rather than 6; the same holds here, and it
        is worth pinning because a 6th candidate would change the per-pathway
        forward-pass count the compute estimate rests on.
        """
        logits = random_logits(30, seed=3)

        candidates = backend.fill_candidates(logits, n_candidates=5)

        assert len(candidates) == 5
        argmax = torch.argmax(backend.restrict_logits_to_valid_aa(logits), dim=-1)
        assert candidates[0].tolist()[1:] == argmax.tolist()[1:]

    def test_deduplicates_the_argmax_against_the_top_candidate(self):
        """The argmax IS the top-1 by joint log probability, so it dedups away."""
        logits = torch.full((10 + 1, VOCAB_SIZE), -30.0)
        for position in range(11):
            for token in backend.VALID_AA_TOKENS:
                logits[position, token] = -5.0
            logits[position, 4] = 10.0

        candidates = backend.fill_candidates(logits, n_candidates=1)

        assert len(candidates) == 1

    def test_every_candidate_is_in_protmixys_layout(self):
        candidates = backend.fill_candidates(random_logits(20, seed=5), n_candidates=3)

        for candidate in candidates:
            assert int(candidate[0]) == CLS
            assert len(candidate) == 21
            assert candidate.dtype == torch.int64
            assert all(int(t) in VALID_TOKENS for t in candidate[1:])

    def test_order_is_deterministic_across_calls(self):
        logits = random_logits(15, seed=9)

        first = [c.tolist() for c in backend.fill_candidates(logits, 4)]
        second = [c.tolist() for c in backend.fill_candidates(logits, 4)]

        assert first == second


class TestForwardShapeAssertions:
    def good_output(self, n_tokens=7, layer=6, dim=320):
        return {
            "logits": torch.zeros(1, n_tokens, VOCAB_SIZE),
            "representations": {layer: torch.zeros(1, n_tokens, dim)},
            "attentions": torch.zeros(1, layer, 20, n_tokens, n_tokens),
        }

    def test_accepts_a_well_formed_forward_pass(self):
        backend.assert_forward_shapes(self.good_output(), 7, 6, 320)

    def test_rejects_a_logits_tensor_with_the_msa_depth_axis(self):
        out = self.good_output()
        out["logits"] = torch.zeros(1, 3, 7, VOCAB_SIZE)

        with pytest.raises(AssertionError):
            backend.assert_forward_shapes(out, 7, 6, 320)

    def test_rejects_the_wrong_embedding_dimension(self):
        """A smoke on ESM-2 8M sees 320, not the 650M's 1280."""
        with pytest.raises(AssertionError):
            backend.assert_forward_shapes(self.good_output(dim=320), 7, 6, 1280)

    def test_rejects_the_msa_transformers_attention_key(self):
        out = self.good_output()
        out["row_attentions"] = out.pop("attentions")

        with pytest.raises(AssertionError):
            backend.assert_forward_shapes(out, 7, 6, 320)


class TestDegeneracyInstrumentation:
    def test_homopolymer_run_counts_the_longest_stretch(self):
        assert backend.max_homopolymer_run("MAAAKY") == 3

    def test_homopolymer_run_is_not_bridged_by_a_gap(self):
        assert backend.max_homopolymer_run("AA--AAA") == 3

    def test_homopolymer_run_of_an_all_gap_row_is_zero(self):
        assert backend.max_homopolymer_run("-----") == 0

    def test_repeat_kmer_fraction_is_zero_for_unique_kmers(self):
        assert backend.repeat_kmer_fraction("MAKYCWF") == 0.0

    def test_repeat_kmer_fraction_approaches_one_for_a_collapsed_sequence(self):
        assert backend.repeat_kmer_fraction("A" * 60) == pytest.approx(1 - 1 / 58)

    def test_repeat_kmer_fraction_ignores_gaps(self):
        assert backend.repeat_kmer_fraction("A-A-A-A-A-A") == backend.repeat_kmer_fraction("AAAAAA")

    def test_a_collapsed_step_reads_worse_on_every_column_but_perplexity(self):
        """The trap: a collapsed beam scores BETTER on perplexity."""
        collapsed = ["AAAAAAAAAAAA"] * 3
        healthy = ["MAKYCWFTVLIG", "GILVTFWCYKAM", "CWFTVMAKYLIG"]

        assert backend.step_diversity(collapsed) < backend.step_diversity(healthy)
        assert max(map(backend.max_homopolymer_run, collapsed)) > \
            max(map(backend.max_homopolymer_run, healthy))
        assert max(map(backend.repeat_kmer_fraction, collapsed)) > \
            max(map(backend.repeat_kmer_fraction, healthy))


class TestDecode:
    def test_decode_drops_the_cls_token_like_protmixy_does(self):
        """`decode_token_to_aa` (src/utils/helpers.py:133) does `seq_tokens[1:]`."""
        assert backend.decode(torch.tensor([CLS, 4, GAP, 5])) == "L-A"

    def test_encode_then_decode_round_trips_an_alignment_row(self):
        row = "MA--KYC-WF"

        assert backend.decode(backend.encode_query(row)) == row
