"""
Configuration file for stochastic protein evolution using MSA-Transformer.

This module contains all hyperparameters and settings for running protein
evolution experiments using iterative refinement sampling (IRS) and 
attention-based positional coupling (APC) methods.
"""
import torch
import types
import os

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Evolution method selection
GENERATOR_METHOD = 'irs'  # Options: 'irs' (Independent Residue Sampling), 
                          #          'apc' (Attention-based Positional Coupling)
# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Root path - adjust based on your environment
ROOT_PATH = f"."

# Data paths
MAIN_DATA_PATH        = f"{ROOT_PATH}/data/"
INPUT_FILE_PATH       = f"{MAIN_DATA_PATH}input_data/"
OUTPUT_FILE_PATH      = f"{MAIN_DATA_PATH}output_data/"

# =============================================================================
# GENERATOR BACKEND  (ESM-2 650M baseline arm; Cell Systems revision, R2 #1)
# =============================================================================
# 'msa'  -> esm_msa1_t12_100M_UR50S, the published method. This is the default,
#           so a plain `python generate_pathway.py` still reproduces the
#           committed results bit for bit.
# 'esm2' -> ESM-2 650M, the single-sequence pLM baseline Reviewer 2 asked for.
#
# The endpoint constraint is NOT affected either way: distance_to_probabilities()
# (src/utils/helpers.py:280) is model-free and zeroes every position that already
# matches the target, so only differing positions are ever masked.
GENERATOR_BACKEND = os.environ.get('PROTMIXY_BACKEND', 'msa').strip().lower()

# Which model supplies mask SELECTION and scoring. Defaults to the generator, so
# PROTMIXY_BACKEND=esm2 on its own is the full swap that answers the reviewer.
# Set PROTMIXY_SELECTION_BACKEND=msa for the strict one-factor variant in which
# only the fill changes -- that is the one that isolates MSA conditioning.
SELECTION_BACKEND = os.environ.get(
    'PROTMIXY_SELECTION_BACKEND', GENERATOR_BACKEND).strip().lower()

if GENERATOR_BACKEND not in ('msa', 'esm2'):
    raise ValueError(f"PROTMIXY_BACKEND must be 'msa' or 'esm2', got {GENERATOR_BACKEND!r}")
if SELECTION_BACKEND not in ('msa', 'esm2'):
    raise ValueError(
        f"PROTMIXY_SELECTION_BACKEND must be 'msa' or 'esm2', got {SELECTION_BACKEND!r}")

ESM2_MODEL_NAME = os.environ.get('PROTMIXY_ESM2_MODEL', 'esm2_t33_650M_UR50D')

# ESM-2 rescales input embeddings by (1 - 0.12) / (1 - observed mask ratio)
# (esm/esm2.py:88-92). The ratio is counted per forward pass and this loop masks
# 0 positions at iteration 1 and ~P_MASK afterwards, so leaving it on would make
# the rescale step-varying. It is also exactly the sort of perturbation that
# weakens BERT-style copying of unmasked tokens, which the loop relies on.
ESM2_TOKEN_DROPOUT = False

# Run the FILL ungapped and restore gaps by column map. The shipped conditioning
# contexts are 54-68 % gap by character and ESM-2 saw ungapped UniRef50 only.
ESM2_UNGAPPED_FILL = os.environ.get(
    'PROTMIXY_ESM2_UNGAPPED_FILL', '1').strip().lower() not in ('0', 'false', 'no')

# Sequence identifiers - should be present in FULL_CONTEXT_FILE
START_SEQ_NAME        = os.environ.get(
    'PROTMIXY_START_SEQ', 'tr|A0A7Z9ZYA4|A0A7Z9ZYA4_9BACT')
END_SEQ_NAME          = os.environ.get(
    'PROTMIXY_END_SEQ', 'tr|A0A832BM14|A0A832BM14_9PROT')

# other combinations
# START_SEQ_NAME        = 'UPI001B01FD94'
# END_SEQ_NAME          = 'UPI0004AC0335'

# START_SEQ_NAME        = 'UPI002F2588A6'
# END_SEQ_NAME          = 'UPI0009A7E04F'

# START_SEQ_NAME        = 'UPI0005411E7E'
# END_SEQ_NAME          = 'UPI0004AC0335'

PAIR_OUTPUT_FILE_PATH = os.environ.get(
    'PROTMIXY_PAIR_DIR', f"{OUTPUT_FILE_PATH}{START_SEQ_NAME}_{END_SEQ_NAME}") .rstrip('/') + '/'

# The backend is carried in the output path so a baseline run can never overwrite
# the committed MSA-Transformer results: the method keeps writing <pair>/irs/,
# the baseline writes <pair>/irs_esm2/ and the strict variant <pair>/irs_esm2_selmsa/.
_BACKEND_SUFFIX = '' if GENERATOR_BACKEND == 'msa' else f'_{GENERATOR_BACKEND}'
if SELECTION_BACKEND != GENERATOR_BACKEND:
    _BACKEND_SUFFIX += f'_sel{SELECTION_BACKEND}'

GENERATOR_OUTPUT_PATH = f"{PAIR_OUTPUT_FILE_PATH}{GENERATOR_METHOD}{_BACKEND_SUFFIX}/"
FULL_CONTEXT_FILE     = os.environ.get(
    'PROTMIXY_FULL_CONTEXT', f'{PAIR_OUTPUT_FILE_PATH}full_context.aln')
MSA_CONTEXT_FILE      = os.environ.get(
    'PROTMIXY_MSA_CONTEXT', f'{PAIR_OUTPUT_FILE_PATH}cond_context.aln')
FULL_MSA_FILE         = f'{INPUT_FILE_PATH}full_context.aln'

# Create output directory if it doesn't exist
os.makedirs(GENERATOR_OUTPUT_PATH, exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# MSA-Transformer model for sequence generation
GENERATOR_MODEL_NAME = "esm_msa1_t12_100M_UR50S"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# PATHWAY GENERATION HYPERPARAMETERS
# =============================================================================
RANDOM_SEED = int(os.environ.get('PROTMIXY_SEED', 10))

# Simulated annealing parameters
ANNEAL_TEMP = 1.0
TEMP_DECAY  = 0.99
ANNEAL_TEMP_MIN = 0.0001 * ANNEAL_TEMP

# Masking parameters
MASK_ID = 32  # Token ID for masked positions
P_MASK = 0.05  # Proportion of sequence to mask
DISTANCE_TEMP = 0.1  # Temperature for probabilistic sampling (sharper distribution)
ENTROPY_THRESHOLD_FILTER = 30  # Percentile threshold for entropy filtering

# Beam search parameters
N_ITER = int(os.environ.get('PROTMIXY_N_ITER', 100))  # Maximum number of iterations
STOP_TOL_FACTOR = 0.25  # Stopping tolerance as fraction of initial distance
N_BEAM = 3  # Number of beams to maintain
N_TOSS = 3  # Number of sampling attempts per beam
N_CANDIDATES = 5  # Number of top-k candidates to generate per toss


# =============================================================================
# CONDITIONING CONTEXT GENERATION HYPERPARAMETERS
# =============================================================================

DENSE_SEED_THRESHOLD = 0.02

# =============================================================================
# FAIRNESS: the two arms must differ in the generator and in nothing else
# =============================================================================
# eLife 79854 Reviewer 1 made this objection to the method's own lineage --
# "I feel that the comparison of Figure 1 is a bit unfair", the MSA Transformer
# sampled at T=0 against bmDCA at T=1, "the proposed method also has free
# parameters". Both arms read this one module, so there is no second copy to
# drift; what follows records the reference values and says out loud when a run
# is not at them, so a non-matched run can never be reported as the benchmark.
MATCHED_HYPERPARAMETERS = {
    'N_BEAM': 3, 'N_TOSS': 3, 'N_CANDIDATES': 5, 'N_ITER': 100,
    'ANNEAL_TEMP': 1.0, 'TEMP_DECAY': 0.99, 'ANNEAL_TEMP_MIN': 0.0001,
    'P_MASK': 0.05, 'DISTANCE_TEMP': 0.1, 'ENTROPY_THRESHOLD_FILTER': 30,
    'STOP_TOL_FACTOR': 0.25, 'MASK_ID': 32, 'GENERATOR_METHOD': 'irs',
    'RANDOM_SEED': 10,
}

def hyperparameter_deviations():
    """Return every matched knob whose live value is off the reference."""
    live = globals()
    return {
        knob: (reference, live[knob])
        for knob, reference in MATCHED_HYPERPARAMETERS.items()
        if knob in live and live[knob] != reference
    }

# RANDOM_SEED is swept on purpose by the replicate design (R2 #4), so it is
# recorded but does not make a run non-matched.
SWEPT_KNOBS = ('RANDOM_SEED',)

def hyperparameters_matched():
    return not {k: v for k, v in hyperparameter_deviations().items() if k not in SWEPT_KNOBS}

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def print_config():
    """Print all configuration parameters."""
    config_vars = {
        key: value
        for key, value in globals().items()
        if not key.startswith("__") and not callable(value) and not isinstance(value, types.ModuleType)
    }
    
    exclude_keys = {"CONFIGS", "config_vars"}
    
    print("\n" + "="*80)
    print("PATHWAY GENERATION CONFIGURATION")
    print("="*80)
    for key, value in config_vars.items():
        if key not in exclude_keys:
            print(f"{key}: {value}")
    generator = ESM2_MODEL_NAME if GENERATOR_BACKEND == 'esm2' else GENERATOR_MODEL_NAME
    print("-"*80)
    print(f"ARM: generator={GENERATOR_BACKEND} ({generator}), selection={SELECTION_BACKEND}")
    if GENERATOR_BACKEND == 'esm2':
        print(f"ARM: token_dropout={ESM2_TOKEN_DROPOUT}, ungapped_fill={ESM2_UNGAPPED_FILL}")
    deviations = hyperparameter_deviations()
    if deviations:
        print("!"*80)
        for knob, (reference, live) in sorted(deviations.items()):
            tag = "swept" if knob in SWEPT_KNOBS else "NOT MATCHED"
            print(f"! {knob}: reference {reference}, this run {live}  [{tag}]")
        if not hyperparameters_matched():
            print("! THIS RUN IS NOT THE MATCHED BENCHMARK. Do not report it as one.")
        print("!"*80)
    print("="*80 + "\n")

# Auto-print configuration when module is imported
if __name__ != "__main__":
    print_config()
