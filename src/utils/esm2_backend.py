"""ESM-2 generator arm for protmixy.

Swaps the MSA Transformer for ESM-2 650M as the *generator* only. The endpoint
constraint is untouched: `distance_to_probabilities()` (src/utils/helpers.py:280)
zeroes every position that already matches the target, so only differing
positions are ever masked and the model only chooses which residue fills a
masked column.

This module holds the parts that have to stay compatible with the
MSA-Transformer code path, each of which differs in a way that would corrupt a
pathway silently:

* **Token layout.** ESM-2's alphabet sets `append_eos=True`; the MSA
  Transformer's does not. protmixy assumes `[cls] + residues` of length L+1
  (src/utils/helpers.py:443) and decodes with `seq_tokens[1:]`
  (src/utils/helpers.py:133), so `encode_query` builds that layout directly
  rather than going through ESM-2's batch converter.
* **Vocabulary.** `top_k_samples()` (src/utils/helpers.py:60-83) ranges over all
  33 tokens. protmixy already lists the correct allowed set as `valid_aa_vals`
  (src/utils/helpers.py:32-33) but never references it, so B/Z/X/U/O/'.' and
  even `<mask>` are reachable. We filter the logits before the softmax instead
  of patching the sampled sequence afterwards.
* **Gaps.** ESM-2 was pretrained on ungapped UniRef, so gapped alignment columns
  are out of distribution. The query is scored ungapped and gaps are restored by
  column map.
* **Degeneracy.** Iterative masked-pLM filling can collapse to repetitive
  sequence. Per-step diversity and pseudo-perplexity are reported beside the
  headline so that a degenerate result reads as a mechanism, not as a second
  weak comparator.
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
from esm.data import Alphabet

# ESM-2 uses the ESM-1b alphabet. Verified identical to the MSA Transformer's on
# token ids 4-23 (the canonical residues), 30 ('-') and 32 ('<mask>'); the two
# differ only in `append_eos`.
ESM2_ARCHITECTURE = "ESM-1b"

#: The 20 canonical residues plus the gap, exactly as protmixy's unused
#: `valid_aa_vals` lists them (src/utils/helpers.py:32-33).
VALID_AA_TOKENS: Tuple[int, ...] = (
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 30,
)

GAP_CHARACTER = "-"
CLS_TOKEN = 0
GAP_TOKEN = 30
MASK_TOKEN = 32
VOCAB_SIZE = 33

_alphabet = Alphabet.from_architecture(ESM2_ARCHITECTURE)


def encode_query(sequence: str) -> torch.Tensor:
    """Tokenise a query into protmixy's layout: ``[cls] + one token per residue``.

    Deliberately not ESM-2's batch converter, which would append an EOS token
    that `decode_token_to_aa` (src/utils/helpers.py:133) would decode into the
    sequence string and that would shift every downstream position index.
    """
    tokens = [_alphabet.cls_idx]
    tokens.extend(_alphabet.tok_to_idx[residue] for residue in sequence)
    return torch.tensor(tokens, dtype=torch.int64)


def decode(tokens: torch.Tensor) -> str:
    """Tokens back to an alignment row, dropping the leading ``<cls>``.

    The same convention as `decode_token_to_aa` (src/utils/helpers.py:122-136),
    repeated here so the backend can be tested without importing `helpers`,
    which pulls in pysam, h5py and Biopython.
    """
    return "".join(
        _alphabet.all_toks[int(token)] for token in torch.as_tensor(tokens).reshape(-1)[1:]
    )


def restrict_logits_to_valid_aa(logits: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``logits`` with every non-residue token set to ``-inf``.

    Applied before the softmax, so disallowed tokens carry exactly zero
    probability mass and can never be sampled. Filtering, not post-patching:
    replacing an out-of-vocabulary residue after the fact (as Wang et al. did
    with lysine) changes the sequence the model actually scored.
    """
    restricted = torch.full_like(logits, float("-inf"))
    allowed = torch.tensor(VALID_AA_TOKENS, dtype=torch.int64, device=logits.device)
    restricted[..., allowed] = logits[..., allowed]
    return restricted


def ungap(aligned_sequence: str) -> Tuple[str, List[int]]:
    """Strip gaps, returning the ungapped sequence and each residue's column."""
    residues = []
    column_map = []
    for column, character in enumerate(aligned_sequence):
        if character != GAP_CHARACTER:
            residues.append(character)
            column_map.append(column)
    return "".join(residues), column_map


def regap(ungapped_sequence: str, column_map: Sequence[int], aligned_length: int) -> str:
    """Place residues back on their original alignment columns."""
    if len(ungapped_sequence) != len(column_map):
        raise ValueError(
            f"expected {len(column_map)} residues to match the column map, "
            f"got {len(ungapped_sequence)}"
        )
    columns = [GAP_CHARACTER] * aligned_length
    for residue, column in zip(ungapped_sequence, column_map):
        columns[column] = residue
    return "".join(columns)


def step_diversity(sequences: Sequence[str]) -> float:
    """Mean pairwise Hamming distance across a step's sequences, as a fraction.

    0.0 means every sequence at this step is identical -- the signature of
    degenerate iterative filling.
    """
    if len(sequences) < 2:
        return 0.0

    fractions = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            first, second = sequences[i], sequences[j]
            length = min(len(first), len(second))
            if length == 0:
                continue
            mismatches = sum(a != b for a, b in zip(first[:length], second[:length]))
            fractions.append(mismatches / length)

    if not fractions:
        return 0.0
    return sum(fractions) / len(fractions)


def max_homopolymer_run(sequence: str) -> int:
    """Longest run of one residue, gaps excluded.

    The ESM3 repetition result (Zhang et al., ICLR 2026, arXiv:2602.00782)
    reports "long-run homopolymer collapse unique to masked models", with
    temperature and entropy sampling raising it further -- and the top-k draw
    here is in that family. Their scope was unconditional and prefix generation,
    not masked infilling at pre-selected columns, so this is instrumentation for
    a risk rather than a prediction of one.
    """
    longest = run = 0
    previous = None
    for residue in sequence:
        if residue == GAP_CHARACTER:
            run, previous = 0, None
            continue
        run = run + 1 if residue == previous else 1
        previous = residue
        longest = max(longest, run)
    return longest


def repeat_kmer_fraction(sequence: str, k: int = 3) -> float:
    """Fraction of k-mers that are not that k-mer's first appearance.

    0.0 means every k-mer is unique; a sequence that has collapsed to a repeated
    motif approaches 1.0. Gaps are removed first so an alignment row is scored
    on its residues.
    """
    ungapped = sequence.replace(GAP_CHARACTER, "")
    if len(ungapped) < k:
        return 0.0
    kmers = [ungapped[i:i + k] for i in range(len(ungapped) - k + 1)]
    return 1.0 - len(set(kmers)) / len(kmers)


def pseudo_perplexity(log_probs: torch.Tensor) -> float:
    """Exponentiated mean negative log-likelihood over the scored positions."""
    if log_probs.numel() == 0:
        return float("nan")
    return math.exp(-float(torch.as_tensor(log_probs).mean()))


def sample_candidates(
    logits: torch.Tensor,
    n_candidates: int,
    return_scores: bool = False,
):
    """Return the ``n_candidates`` most likely fills, in protmixy's token layout.

    Replaces `top_k_samples()` (src/utils/helpers.py:60-83), which accumulates a
    *product* of per-position probabilities. Measured over the 733
    multiplications of a real alignment, that product reaches exactly 0.0 as
    soon as the mean per-position probability drops below 0.362 (float64) or
    0.500 (float32); `heapq.nlargest` then ranks ties by insertion order and
    fills every candidate's tail with token 0, '<cls>'. The MSA Transformer is
    confident enough to stay above the floor, but ESM-2 650M without MSA
    conditioning sits near a mean probability of 0.1 and would underflow around
    column 324. Accumulating log probabilities removes the floor entirely.

    Because the positions are conditionally independent given one forward pass,
    keeping the best ``n_candidates`` prefixes is exact: the most likely full
    sequence is the per-position argmax, and the next best differ from it by the
    cheapest single substitutions.

    Args:
        logits: ``(L+1, vocab)`` logits for the query, row 0 being the cls
            position, as produced by one ESM-2 forward pass.
        n_candidates: how many candidates to return.
        return_scores: also return each candidate's total log probability.

    Returns:
        A list of ``(L+1,)`` int64 tensors ordered by descending log
        probability, each beginning with the cls token; or a
        ``(candidates, scores)`` pair when ``return_scores`` is set.
    """
    log_probs = torch.log_softmax(restrict_logits_to_valid_aa(logits), dim=-1)

    # Only the allowed residues carry any mass, so the rest are never explored.
    allowed = list(VALID_AA_TOKENS)

    beam: List[Tuple[List[int], float]] = [([], 0.0)]
    for position in range(1, log_probs.shape[0]):
        position_log_probs = log_probs[position]
        extended: List[Tuple[List[int], float]] = []
        for tokens, score in beam:
            for token in allowed:
                extended.append((tokens + [token], score + float(position_log_probs[token])))
        extended.sort(key=lambda item: item[1], reverse=True)
        beam = extended[:n_candidates]

    candidates = [
        torch.tensor([_alphabet.cls_idx] + tokens, dtype=torch.int64)
        for tokens, _ in beam
    ]
    if return_scores:
        return candidates, [score for _, score in beam]
    return candidates


def sample_candidates_fast(
    logits: torch.Tensor,
    n_candidates: int,
    return_scores: bool = False,
):
    """`sample_candidates` without the quadratic list copying. Same answer.

    The reference implementation above rebuilds every prefix list at every
    column (``tokens + [token]``), which costs about k*V*L^2/2 element copies --
    roughly 28 million per call at the 733-column alignments this repo ships,
    and it is called 9 times per iteration for 99 iterations. This version keeps
    back-pointers instead of prefixes and only considers each column's top
    ``n_candidates`` allowed tokens, which is exact: a beam entry that reached
    for a worse token would be beaten by the same prefix holding any of the
    better ones.

    `test_fast_sampler_matches_the_reference_implementation` pins the two
    together on random logits, so the reference stays the specification.
    """
    log_probs = torch.log_softmax(restrict_logits_to_valid_aa(logits), dim=-1)
    allowed = torch.tensor(VALID_AA_TOKENS, dtype=torch.int64, device=log_probs.device)
    width = min(n_candidates, len(VALID_AA_TOKENS))

    scores: List[float] = [0.0]
    back: List[List[Tuple[int, int]]] = []
    for position in range(1, log_probs.shape[0]):
        row = log_probs[position].index_select(0, allowed)
        top_values, top_slots = torch.topk(row, k=width)
        top_values = [float(v) for v in top_values]
        top_tokens = [int(t) for t in allowed.index_select(0, top_slots)]

        pool: List[Tuple[float, int, int]] = []
        for slot, prefix_score in enumerate(scores):
            for value, token in zip(top_values, top_tokens):
                pool.append((prefix_score + value, slot, token))
        pool.sort(key=lambda item: item[0], reverse=True)
        pool = pool[:n_candidates]

        scores = [item[0] for item in pool]
        back.append([(item[1], item[2]) for item in pool])

    candidates = []
    for slot in range(len(scores)):
        tokens: List[int] = []
        cursor = slot
        for position in range(len(back) - 1, -1, -1):
            parent, token = back[position][cursor]
            tokens.append(token)
            cursor = parent
        tokens.reverse()
        candidates.append(torch.tensor([CLS_TOKEN] + tokens, dtype=torch.int64))

    if return_scores:
        return candidates, scores
    return candidates


def column_index_map(tokens: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    """Drop unmasked gap columns from what ESM-2 is asked to read.

    ESM-2 was pretrained on ungapped UniRef50 and the conditioning contexts this
    repo ships are 54-68 % gap by character, so feeding the alignment row
    verbatim is a large distribution shift. Masked columns are never dropped:
    `mask_sequence` (src/utils/helpers.py:190) overwrites them with `<mask>`
    (32), including columns whose current residue is a gap, and such a column
    must stay fillable because `distance_to_probabilities` selected it.

    Args:
        tokens: ``(L+1,)`` in protmixy's layout, index 0 being ``<cls>``.

    Returns:
        ``(model_tokens, col_to_input)``: the tokens ESM-2 actually reads, and
        for each alignment column the index it occupies in them, or ``-1`` if
        the column was dropped.
    """
    tokens = torch.as_tensor(tokens).reshape(-1)
    if tokens.numel() == 0 or int(tokens[0]) != CLS_TOKEN:
        raise ValueError("expected protmixy's layout: index 0 must be the <cls> token")

    col_to_input = [0] + [-1] * (tokens.numel() - 1)
    kept = [CLS_TOKEN]
    for column in range(1, tokens.numel()):
        token = int(tokens[column])
        if token == GAP_TOKEN:
            continue
        col_to_input[column] = len(kept)
        kept.append(token)
    return torch.tensor(kept, dtype=torch.int64), col_to_input


def scatter_logits_to_columns(
    model_logits: torch.Tensor, col_to_input: Sequence[int]
) -> torch.Tensor:
    """Lift ESM-2's output back onto the alignment grid.

    A dropped column gets a one-hot-at-gap row, so the beam reproduces its gap
    deterministically and every downstream alignment index keeps its meaning.
    """
    if model_logits.dim() != 2:
        raise ValueError(f"expected (T, vocab) logits, got {tuple(model_logits.shape)}")

    columns = torch.full(
        (len(col_to_input), model_logits.shape[-1]),
        float("-inf"),
        dtype=model_logits.dtype,
    )
    for column, index in enumerate(col_to_input):
        if index < 0:
            columns[column, GAP_TOKEN] = 0.0
            continue
        if index >= model_logits.shape[0]:
            raise ValueError(
                f"column map points at model index {index} but the model "
                f"returned only {model_logits.shape[0]} positions"
            )
        columns[column] = model_logits[index]
    return columns


def query_logits(
    model,
    tokens: torch.Tensor,
    device: str = "cpu",
    ungapped: bool = True,
) -> torch.Tensor:
    """One ESM-2 forward pass over a masked alignment row.

    Returns ``(L+1, 33)`` logits on the *alignment* grid, which is the shape
    `top_k_samples` / `sample_candidates` expect.
    """
    tokens = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
    if ungapped:
        model_tokens, col_to_input = column_index_map(tokens)
    else:
        model_tokens, col_to_input = tokens, list(range(tokens.numel()))

    with torch.no_grad():
        out = model(model_tokens.unsqueeze(0).to(device), repr_layers=[], return_contacts=False)
    logits = out["logits"][0].to("cpu").float()
    return scatter_logits_to_columns(logits, col_to_input)


def fill_candidates(logits: torch.Tensor, n_candidates: int) -> List[torch.Tensor]:
    """Candidates in the contract `msa_query_sample` returns: top-k then argmax.

    Mirrors `msa_query_sample` (src/utils/helpers.py:85-120) — the k best
    sequences plus the per-position argmax, deduplicated — so the two arms hand
    the beam the same kind of list. Order is kept here rather than taken from a
    set, which makes the arm deterministic given its seed.
    """
    candidates = sample_candidates_fast(logits, n_candidates)

    argmax = torch.argmax(restrict_logits_to_valid_aa(logits), dim=-1).to(torch.int64).clone()
    argmax[0] = CLS_TOKEN
    candidates.append(argmax)

    unique: List[torch.Tensor] = []
    seen = set()
    for candidate in candidates:
        key = tuple(int(token) for token in candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def forward_grid(
    model,
    token_rows: Sequence[torch.Tensor],
    repr_layer: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Representations and logits for alignment rows, on the GAPPED grid.

    Used by the mask-*selection* path only. Selection needs a vector at every
    alignment column so that the candidate's and the target's position-wise
    cosine distances are comparable column for column, and a dropped column has
    no embedding to compare — so unlike the fill path this one does not use the
    column map and is knowingly out of distribution on gaps. That is the reason
    `PROTMIXY_SELECTION_BACKEND=msa` is the primary strict variant.

    Returns ``(representations (B, T, dim), logits (B, T, 33))``.
    """
    batch = torch.stack(
        [torch.as_tensor(row).reshape(-1).to(torch.int64) for row in token_rows]
    )
    with torch.no_grad():
        out = model(batch.to(device), repr_layers=[repr_layer], return_contacts=False)
    representations = out["representations"][repr_layer].to("cpu").float()
    logits = out["logits"].to("cpu").float()
    return representations, logits


def assert_forward_shapes(
    out: dict,
    n_tokens: int,
    repr_layer: int,
    embed_dim: Optional[int] = None,
) -> None:
    """Check the three shapes the port was written against but never observed.

    §8 of the baseline README lists them as read from the fair-esm source rather
    than measured: logits ``(1, T, 33)``, representations ``(1, T, dim)`` and an
    ``attentions`` key. `embed_dim` is a parameter because it is model-specific
    -- 1280 at 650M, 480 at 35M, 320 at 8M -- so a smoke on a small checkpoint
    still asserts something real.
    """
    if tuple(out["logits"].shape) != (1, n_tokens, VOCAB_SIZE):
        raise AssertionError(
            f"logits {tuple(out['logits'].shape)} != (1, {n_tokens}, {VOCAB_SIZE})"
        )
    representations = out["representations"][repr_layer]
    if representations.shape[:2] != (1, n_tokens):
        raise AssertionError(
            f"representations {tuple(representations.shape)} do not start (1, {n_tokens})"
        )
    if embed_dim is not None and representations.shape[-1] != embed_dim:
        raise AssertionError(
            f"representations dim {representations.shape[-1]} != {embed_dim}"
        )
    if "attentions" not in out:
        raise AssertionError(
            "no 'attentions' key: the MSA Transformer's is 'row_attentions', and "
            "the APC branch reads it"
        )
