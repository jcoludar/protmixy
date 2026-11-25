"""
Protein Evolution using MSA-Transformer.

This module implements iterative sampling from MSA-Transformer for generating
evolutionary paths between protein sequences. Two methods are supported:

1. IRS (Iterative Refinement Sampling): Uses cosine distance-based position sampling
2. APC (Attention-based Positional Coupling): Uses attention weights for coupled position sampling

The algorithm maintains a beam search with simulated annealing to explore the
sequence space between a starting and ending protein sequence.
"""

import torch
import numpy as np
import random
from tqdm import tqdm
from pysam import FastaFile
from src.utils.model_loader import ModelLoader
from src.utils import esm2_backend
from config.settings import ANNEAL_TEMP, TEMP_DECAY, ANNEAL_TEMP_MIN,\
      STOP_TOL_FACTOR, DEVICE, N_BEAM, N_TOSS, GENERATOR_METHOD, P_MASK,\
      GENERATOR_BACKEND, SELECTION_BACKEND, ESM2_MODEL_NAME, GENERATOR_MODEL_NAME,\
      hyperparameters_matched, hyperparameter_deviations
from src.utils.helpers import (
    create_msa_for_iterative_sampling, msa_query_sample,
    mask_sequence, eval_candidate_pathway, decode_token_to_aa,
    generate_pathway_mask, tokens_changed, accept_or_reject_beam_candidates,
    accept_or_reject_iteration_candidates, assemble_paths, get_row_col_attention, apply_apc, validate_path_consistency)
import pickle

# The alphabet only -- no checkpoint is read here. The generator loads lazily
# inside `msa_query_sample`, which is what lets the ESM-2 arm run without an
# MSA-Transformer checkpoint on disk.
msa_alphabet = ModelLoader.get_alphabet()
msa_batch_converter = msa_alphabet.get_batch_converter()
idx_list = msa_alphabet.tok_to_idx # reference dict for model token
aa_list  = {v: k for k,v in idx_list.items()}
valid_aa_vals = torch.tensor([ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                             18, 19, 20, 21, 22, 23, 30],dtype=torch.int64)

#: Columns of the per-step degeneracy table. Perplexity alone cannot read
#: degeneracy -- a collapsed beam scores BETTER on it -- so diversity, the
#: longest homopolymer run and the repeated-3mer fraction are reported beside it.
DEGENERACY_COLUMNS = (
    'iteration', 'beam', 'n_candidates', 'diversity', 'max_homopolymer_run',
    'repeat_3mer_fraction', 'n_masked', 'n_changed', 'n_changed_unmasked',
)


def _degeneracy_row(iteration, beam_idx, sequences, n_masked, n_changed, n_changed_unmasked):
    if not sequences:
        return None
    return (
        iteration, beam_idx, len(sequences),
        round(esm2_backend.step_diversity(sequences), 6),
        max(esm2_backend.max_homopolymer_run(s) for s in sequences),
        round(max(esm2_backend.repeat_kmer_fraction(s) for s in sequences), 6),
        n_masked, n_changed, n_changed_unmasked,
    )

def iterative_sampling(full_context_file, starting_seq_name, ending_seq_name, context_msa_file, random_seed, n_iter, p_mask, output_file_path):

    generator_name = ESM2_MODEL_NAME if GENERATOR_BACKEND == 'esm2' else GENERATOR_MODEL_NAME
    print(f"🧪 ARM: generator={GENERATOR_BACKEND} ({generator_name}), selection={SELECTION_BACKEND}")
    if not hyperparameters_matched():
        print(f"🚨 NOT THE MATCHED BENCHMARK: {hyperparameter_deviations()}")

    if GENERATOR_BACKEND == 'esm2' and GENERATOR_METHOD == 'apc':
        # ESM-2's `attentions` is single-sequence self-attention; the MSA
        # Transformer's `row_attentions` is tied attention pooled over MSA
        # depth. They are dimensionally swappable and semantically different,
        # so an ESM-2 APC arm is an ablation to design, not a port to assume.
        raise NotImplementedError(
            "GENERATOR_METHOD='apc' with the ESM-2 backend is not implemented: "
            "row_attentions (tied, over MSA depth) and attentions (single-sequence "
            "self-attention) are not the same quantity. Use GENERATOR_METHOD='irs'."
        )

    # Seed the sampler
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get the starting and ending sequence
    global_fasta_file_lkp = FastaFile(full_context_file)
    starting_seq = global_fasta_file_lkp.fetch(starting_seq_name)
    ending_seq   = global_fasta_file_lkp.fetch(ending_seq_name)

    # compare source sequence to the target sequence
    current_score, current_aa_tgt, current_aa_src, \
    current_pos_wise_dist_to_tgt, current_tgt_pos_diff_aa, current_pos_entropy, current_ll\
    = eval_candidate_pathway(starting_seq, starting_seq, ending_seq, context_msa_file)

    # total distance to travel for the source sequence
    total_dist_to_target, total_aa_to_target = current_score, current_aa_tgt

    # Add starting sequence to the MSA
    msa_data, total_seqs, size_align_seq = create_msa_for_iterative_sampling(context_msa_file, starting_seq)
    print(f"Total sequences in MSA {total_seqs} and size of alignment {size_align_seq}")

    # for iterating
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    current_sequence_token      = msa_batch_tokens[0,0,:].clone()

    # current state
    current_beam = [{
        'id': 0,
        'parent_id': 0,
        'name': starting_seq_name,
        'sequence': starting_seq,
        'sequence_token': current_sequence_token,
        'no_positions_masked': 0,
        'no_positions_changed': 0,
        'positions_masked': [],
        'positions_changed': [],
        'score': current_score,
        'aa_dist_target': current_aa_tgt,
        'aa_dist_source': current_aa_src,
        'pos_wise_dist_to_tgt': current_pos_wise_dist_to_tgt,
        'pos_entropy': current_pos_entropy,
        'llm_ll': current_ll,
        'tgt_pos_diff_aa': current_tgt_pos_diff_aa,
        'converged': False,
        'status': 'START' }]

    # Initialise variables
    t = ANNEAL_TEMP
    path_history = [current_beam[0]]
    num_exit_candidates = 0

    # stop tolerance
    stop_tol = STOP_TOL_FACTOR * total_dist_to_target
    print(f" 📊 Cosine distance to target is {total_dist_to_target} and stop tolerance is set to {stop_tol}.")

    # Per-step degeneracy table. Iterative masked-pLM filling collapsing to
    # repetitive sequence is the main scientific risk of this arm, and a
    # collapsed run that is only reported as a worse score reads as a second
    # weak comparator rather than as the mechanism it is.
    degeneracy_file = f"{output_file_path}degeneracy_{GENERATOR_BACKEND}_{random_seed}.tsv"
    degeneracy_rows = []

    # Main iterative search loop
    for it in tqdm(range(1, n_iter), desc="Processing", unit="step"): 
        print(f"🏹 Starting Iteration: {it}.")
        print(f"🏹 Number of exit candidates so far: {num_exit_candidates}")

        this_iteration_candidates = []
        for beam_idx, current_state in enumerate(current_beam): # 3 beams
            this_beam_candidates = []
            
            current_state_id             = current_state['id']
            current_sequence_token       = current_state['sequence_token']
            current_pos_wise_dist_to_tgt = current_state['pos_wise_dist_to_tgt']
            current_pos_entropy          = current_state['pos_entropy']
            current_score                = current_state['score']
            current_tgt_pos_diff_aa      = current_state['tgt_pos_diff_aa']
            current_aa_dist_target       = current_state['aa_dist_target']
            current_aa_dist_source       = current_state['aa_dist_source']

            print(f"🔦 Beam {beam_idx} with id: {current_state_id} and score {current_score} ")
            print(f"🔦 AA distance to target {current_aa_dist_target} and source {current_aa_dist_source}")
        
            if GENERATOR_METHOD == 'apc':
                msa_batch_tokens[0, 0, :] = current_sequence_token
                avg_row_attn = get_row_col_attention(msa_batch_tokens)
                apc_avg_row_attn = apply_apc(avg_row_attn)
            else:
                apc_avg_row_attn = None
        
            for toss in range(N_TOSS): # 3 tossess
                #print(f"🎲 Toss {toss}.")

                # generate mask
                mask, num_pos_mask = generate_pathway_mask(it, size_align_seq, \
                    current_pos_wise_dist_to_tgt, current_pos_entropy, current_tgt_pos_diff_aa, apc_avg_row_attn)

                msa_batch_tokens[0, 0, :], target_pos_change = mask_sequence(current_sequence_token, mask)

                # get candidates in a list
                cand_sequence_token_lst  = msa_query_sample(msa_batch_tokens)

                masked_positions = {int(p) for p in target_pos_change}
                toss_sequences = []
                toss_changed = 0
                toss_leaked = 0

                # evaluate candidate
                for candidate_idx in range(len(cand_sequence_token_lst)): # 5 candidates
                    candidate_state_id   = str(current_state_id) + '-' + str(it)  + ':' + str(toss) + ':' + str(candidate_idx)
                    
                    cand_sequence_token = cand_sequence_token_lst[candidate_idx]
                    cand_sequence       = decode_token_to_aa(cand_sequence_token)

                    # score candidate
                    cand_score, cand_aa_tgt, cand_aa_src, cand_pos_wise_dist_to_tgt, cand_tgt_pos_diff_aa, cand_pos_entropy, cand_ll\
                        = eval_candidate_pathway(cand_sequence, starting_seq, ending_seq, context_msa_file)
                
                    # get changed information
                    actual_pos_changed, num_pos_changed = tokens_changed(current_sequence_token, cand_sequence_token)

                    # Constraint leakage: nothing in the algorithm pins an
                    # UNMASKED column to its current residue -- `top_k_samples`
                    # and `sample_candidates` both beam-search every column, and
                    # only BERT-style copying keeps them still in practice. The
                    # set of columns that may be masked is model-independent;
                    # the set that actually changes is not, so it is counted
                    # rather than assumed away.
                    toss_sequences.append(cand_sequence)
                    toss_changed += num_pos_changed
                    toss_leaked += sum(
                        1 for position in actual_pos_changed
                        if position not in masked_positions
                    )

                    # create candidate state
                    candidate_state = {
                        'id': candidate_state_id,
                        'parent_id': current_state_id,
                        'name': f'seq_{candidate_state_id}',
                        'sequence': cand_sequence,
                        'sequence_token': cand_sequence_token,
                        'no_positions_masked': num_pos_mask,
                        'no_positions_changed': num_pos_changed,
                        'positions_masked': target_pos_change,
                        'positions_changed': actual_pos_changed,
                        'score': cand_score,
                        'aa_dist_target': cand_aa_tgt,
                        'aa_dist_source': cand_aa_src,
                        'pos_wise_dist_to_tgt': cand_pos_wise_dist_to_tgt,
                        'pos_entropy': cand_pos_entropy,
                        'llm_ll': cand_ll,
                        'tgt_pos_diff_aa': cand_tgt_pos_diff_aa,
                        'converged': False,
                        'status': 'temporary'
                         }
                    #print(f" 🌱 Candidate {candidate_idx} generated with score {cand_score}.")
                    # check if candidate can exit the beam else add candidate to this iteration
                    if cand_score <= stop_tol:
                        candidate_state['converged'] = True
                        candidate_state['status'] = 'ACCEPT'
                        path_history.append(candidate_state)
                        num_exit_candidates += 1
                        print(f" 🎉 Candidate {candidate_idx} converged with score {cand_score}.")
                    else:
                        this_beam_candidates.append(candidate_state)

                row = _degeneracy_row(
                    it, beam_idx, toss_sequences, num_pos_mask, toss_changed, toss_leaked
                )
                if row is not None:
                    degeneracy_rows.append(row)

            # use the simulated annealing criteria to accept or reject candidates
            accepted_beam_candidates = accept_or_reject_beam_candidates(this_beam_candidates, current_state, t, it)
            print(f" 🔦 Accepted {len(accepted_beam_candidates)} candidates")
            this_iteration_candidates.extend(accepted_beam_candidates)

        print(f" 🏹 Current Iteration Finished. Total candidates = {len(this_iteration_candidates)}")
        if len(this_iteration_candidates) == 0:
            print(f" 🚨 No more candidates left to process. Exiting.")
            break

        # use the log likelihood to accept the top N_BEAM candidates
        current_beam = accept_or_reject_iteration_candidates(this_iteration_candidates, N_BEAM)
        
        # Print which candidates were selected for the next beam
        selected_ids = [candidate['id'] for candidate in current_beam]
        print(f" 🔷 New Beam Size = {len(current_beam)} and selected candidates are {selected_ids}")

        # add the current beam to the path history and make their status as ACCEPT
        for candidate in current_beam:
            candidate['status'] = 'ACCEPT'
        path_history.extend(current_beam)

        # Temperature decisions
        t = max(ANNEAL_TEMP_MIN, ANNEAL_TEMP * (TEMP_DECAY ** it))

    final_it = it + 1
    print(f" 🏹 Final Iteration Finished. Total candidates = {len(path_history) - 1}")
    converged_status = num_exit_candidates > 0
    path_history.append({
        'id': final_it,
        'parent_id': -9999999, # can have multiple parents.
        'name': ending_seq_name,
        'sequence': ending_seq,
        'sequence_token': None,
        'no_positions_masked': 0,
        'no_positions_changed': 0,
        'positions_masked': [],
        'positions_changed': [],
        'score': 0,
        'aa_dist_target': 0,
        'aa_dist_source': total_aa_to_target,
        'pos_wise_dist_to_tgt': None,
        'pos_entropy': None,
        'llm_ll': None,
        'tgt_pos_diff_aa': None,
        'converged':  converged_status,
        'status': 'END' })

    # lets save the per-step degeneracy table
    with open(degeneracy_file, "w") as f:
        f.write("\t".join(DEGENERACY_COLUMNS) + "\n")
        for row in degeneracy_rows:
            f.write("\t".join(str(value) for value in row) + "\n")
    print(f"✅ Degeneracy table ({len(degeneracy_rows)} rows) saved to {degeneracy_file}")

    # lets save path history
    with open(f"{output_file_path}beam_evol_msat_history_{random_seed}.pkl", "wb") as f:
        pickle.dump(path_history, f)
    print(f"✅ Beam history and exit candidates saved to {output_file_path}beam_evol_msat_history_{random_seed}.pkl")

    # lets get all intermediate sequences and write to fasta file
    intermediate_sequences = [path for path in path_history if path['status'] == 'ACCEPT']
    with open(f"{output_file_path}beam_evol_msat_intermediate_seqs_{random_seed}.fasta", "w") as f:
        for path in intermediate_sequences:
            f.write(f">{path['name']}\n{path['sequence']}\n")
    print(f"✅ Intermediate sequences saved to {output_file_path}beam_evol_msat_intermediate_seqs_{random_seed}.fasta")

    # lets assemble paths if there is atleast one
    if converged_status:
        assemble_paths(path_history, output_file_path, random_seed)
        print(f"✅ Mutational paths are finished and written to the folder {output_file_path}")
    else:
        print(f"❌ No mutational paths found. No paths assembled.")
    
    # Perform consistency validation after all files are created
    validation_results = validate_path_consistency(path_history, output_file_path, random_seed)
    print(f"✅ Consistency validation completed: {validation_results}")
    
    return converged_status




                



            
        

        
       
        
        
        