"""
MSA output utilities for extracting embeddings and statistics from MSA-Transformer.

This module provides functions to compute embeddings, entropy, and log-likelihood
from MSA-Transformer for candidate sequences in the context of an MSA.
"""
import torch
import os
import numpy as np
import tempfile
from src.utils import helpers
from src.utils import esm2_backend
from src.utils.model_loader import ModelLoader
from config.settings import DEVICE, SELECTION_BACKEND

#: Target rows are constant for a pair, so the ESM-2 selection path reads each
#: context file once. Keyed by path; the MSA path cannot cache because the
#: target's representation there depends on the candidate sharing its forward
#: pass (tied row attention), which is exactly the conditioning under test.
_esm2_target_cache = {}

def get_embedding(context_msa_file):
    """
    Compute the MSA embedding for a candidate sequence using a pretrained MSA Transformer model.

    Args:
        context_msa_file (str): Path to the context MSA file.

    Returns:
        np.ndarray: Embedding of the candidate sequence.
    """
    # The MSA Transformer explicitly, not the generator arm: this function reads
    # a whole alignment, so it needs the alphabet whose batch converter is an
    # MSABatchConverter. See get_current_state_with_ll below.
    model, alphabet = ModelLoader.get_msa_model()
    msa_batch_converter = alphabet.get_batch_converter()

    if not os.path.exists(context_msa_file):
        raise FileNotFoundError(f"Context MSA file '{context_msa_file}' not found.")
    
    total_seqs, size_align_seq = helpers.check_msa_stats(context_msa_file)
    msa_data = [helpers.read_msa(context_msa_file, total_seqs)]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    
    with torch.no_grad():
        out = model(msa_batch_tokens.to(DEVICE), repr_layers=[12], need_head_weights=False)
        token_representations_wo_cls = out["representations"][12][0,:,1:,:] # 1st position is CLS # 1,3,733,768
        msa_sequence_embedding    = token_representations_wo_cls.mean(dim = 1).to('cpu').squeeze(0).numpy() # 3,768
        
    return msa_batch_labels, msa_sequence_embedding

def get_current_state_with_ll(candidate_sequence, context_msa_file):
  
    """
    Compute embeddings for the current MSA. Order of sequences in the MSA file should be:
    1: Query/Candidate, 2: Source, 3: Target, followed by context sequences

    Returns:
        tuple: (query_embedding, source_embedding, target_embedding, 
                query_pos_embedding, source_pos_embedding, target_pos_embedding)
    """

    # The MSA Transformer explicitly. `get_model()` dispatches on
    # GENERATOR_BACKEND, so under the strict one-factor variant (generator
    # ESM-2, selection MSA) it returned ESM-2 and ESM-2's alphabet, whose batch
    # converter is a plain BatchConverter. That converter reads
    # [(label, seq), ...]; this path passes [[(label, seq), ...]], so it
    # unpacked the alignment's rows and raised
    # "too many values to unpack (expected 2)" -- LRZ job 5929646.
    # Behaviour of the published msa/msa run is unchanged: there the dispatch
    # returned this same model.
    model, alphabet = ModelLoader.get_msa_model()
    msa_batch_converter = alphabet.get_batch_converter()

    # add candidate sequence to the context  
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as temp_file:
        temp_msa_file_name = temp_file.name
        helpers.prepend_sequence_to_fasta(candidate_sequence, 'candidate_sequence', context_msa_file, temp_msa_file_name)


    total_seqs, size_align_seq = helpers.check_msa_stats(temp_msa_file_name)
    msa_data = [helpers.read_msa(temp_msa_file_name, total_seqs)]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    
    with torch.no_grad():
        out = model(msa_batch_tokens.to(DEVICE), repr_layers=[12], return_contacts=False, need_head_weights=False)
        
        all_representations = out["representations"][12][0]  # batch size is 1
        model_logits = out["logits"]
        
        # Extract sequence embeddings (mean across positions, excluding CLS token) - For the first 2 sequences: source, target
        query_embedding  = all_representations[0, 1:, :].mean(dim=0).cpu().numpy()
        target_embedding = all_representations[1, 1:, :].mean(dim=0).cpu().numpy()
        
        # Extract position-wise embeddings (keep as tensors)
        query_pos_embedding  = all_representations[0, :, :].cpu()
        target_pos_embedding = all_representations[1, :, :].cpu()

        # check shape of query and target position embeddings else error
        if query_embedding.shape != target_embedding.shape:
            raise ValueError("Query and Target embeddings have different shapes")

        # check shape of query and target position embeddings else error
        if query_pos_embedding.shape != target_pos_embedding.shape:
            raise ValueError("Query and Target position embeddings have different shapes")

        query_token_probs = torch.log_softmax(model_logits[:,0,:,:], dim=-1).squeeze(0) # 734, 33
        query_token_probs = query_token_probs[1:] # avoid cls token
        query_str = msa_batch_strs[0][0] # msa_batch_strs is list of list
        idx = [alphabet.tok_to_idx[q] for q in query_str]
        query_ll_sum = np.sum(np.diag(query_token_probs[:,idx].cpu().numpy()))
        query_avg_ll = query_ll_sum  #/ len(query_token_probs)
     
        # query sequence - entropy calculation
        query_pos_probs = torch.softmax(model_logits[:,0,:,:], dim=2) # 1, 734, 33
        epsilon = 1e-10
        log_probs   = torch.log(query_pos_probs + epsilon)
        query_pos_entropy = -torch.sum(query_pos_probs * log_probs, dim=2)
        query_pos_entropy = query_pos_entropy.reshape(-1).cpu().numpy()

        return query_embedding, target_embedding, query_pos_embedding, target_pos_embedding, query_pos_entropy, query_avg_ll


def _esm2_target_row(context_msa_file):
    """Row 0 of the context file -- the target -- tokenised and forwarded once.

    `get_current_state_with_ll` prepends the candidate, which makes index 0 the
    candidate and index 1 the target; so in the file itself the target is row 0.
    """
    if context_msa_file in _esm2_target_cache:
        return _esm2_target_cache[context_msa_file]

    if not os.path.exists(context_msa_file):
        raise FileNotFoundError(f"Context MSA file '{context_msa_file}' not found.")

    target_name, target_sequence = helpers.read_msa(context_msa_file, 1)[0]
    model, _ = ModelLoader.get_esm2_model()
    representations, _ = esm2_backend.forward_grid(
        model,
        [esm2_backend.encode_query(target_sequence)],
        ModelLoader.esm2_repr_layer(),
        device=DEVICE,
    )
    target_pos_embedding = representations[0]
    target_embedding = target_pos_embedding[1:, :].mean(dim=0).numpy()

    _esm2_target_cache[context_msa_file] = (
        target_name, target_sequence, target_embedding, target_pos_embedding
    )
    return _esm2_target_cache[context_msa_file]


def get_current_state_esm2(candidate_sequence, context_msa_file):
    """The MSA-free version of `get_current_state_with_ll`.

    Returns the same six values on the same grid -- length L+1, index 0 the cls
    position -- so `PathwayEvaluator.scorer`, `distance_to_probabilities` and
    the `pos+1` convention in `eval_candidate_pathway` all keep their meaning.

    🛑 Run on the **gapped** alignment row on purpose, unlike the fill path.
    Mask selection compares the candidate's and the target's position-wise
    embeddings column for column, and a column dropped by the ungapped map has
    no embedding to compare. ESM-2 is therefore knowingly out of distribution on
    gaps here, which is why `PROTMIXY_SELECTION_BACKEND=msa` -- fill only -- is
    the strict variant worth reporting beside the full swap.
    """
    _, _, target_embedding, target_pos_embedding = _esm2_target_row(context_msa_file)

    model, _ = ModelLoader.get_esm2_model()
    representations, logits = esm2_backend.forward_grid(
        model,
        [esm2_backend.encode_query(candidate_sequence)],
        ModelLoader.esm2_repr_layer(),
        device=DEVICE,
    )

    query_pos_embedding = representations[0]
    query_embedding = query_pos_embedding[1:, :].mean(dim=0).numpy()

    if query_embedding.shape != target_embedding.shape:
        raise ValueError("Query and Target embeddings have different shapes")
    if query_pos_embedding.shape != target_pos_embedding.shape:
        raise ValueError("Query and Target position embeddings have different shapes")

    # Log-likelihood of the candidate's own residues, cls excluded -- the same
    # quantity `accept_or_reject_iteration_candidates` ranks the beam by.
    query_token_probs = torch.log_softmax(logits[0], dim=-1)[1:]
    alphabet = ModelLoader.get_alphabet()
    idx = [alphabet.tok_to_idx[q] for q in candidate_sequence]
    query_avg_ll = float(
        np.sum(np.diag(query_token_probs[:, idx].numpy()))
    )

    # Position-wise entropy over the full grid, cls included, exactly as the
    # MSA path leaves it (:96-100) -- `distance_to_probabilities` takes the
    # ENTROPY_THRESHOLD_FILTER percentile of this array.
    query_pos_probs = torch.softmax(logits[0], dim=-1)
    epsilon = 1e-10
    log_probs = torch.log(query_pos_probs + epsilon)
    query_pos_entropy = (
        -torch.sum(query_pos_probs * log_probs, dim=-1).reshape(-1).numpy()
    )

    return (query_embedding, target_embedding, query_pos_embedding,
            target_pos_embedding, query_pos_entropy, query_avg_ll)


def get_current_state(candidate_sequence, context_msa_file):
    """Dispatch on SELECTION_BACKEND. 'msa' keeps the published path untouched."""
    if SELECTION_BACKEND == 'esm2':
        return get_current_state_esm2(candidate_sequence, context_msa_file)
    return get_current_state_with_ll(candidate_sequence, context_msa_file)