"""Experiment configurations for progressive counterfactual experiments.

Each experiment has two config dicts:
- EXP*_CONFIG: model and training hyperparameters
- EXP*_SCM: SCM / data generation parameters
"""

import torch.nn as nn

# ---------- Experiment 0: Linear Sanity Check ----------
# 1-layer linear SCM, 2 features, 1 perturbed, deterministic.
# The model must learn a constant delta — any model should pass.

EXP0_CONFIG = dict(
    num_features=2,
    seq_len=32,
    batch_size=1,
    emsize=32,
    nlayers=2,
    nhead=2,
    nhid=64,
    dropout=0.0,
    epochs=50,
    steps_per_epoch=50,
    lr=0.001,
    warmup_epochs=3,
)

EXP0_SCM = dict(
    num_layers=2,
    num_causes=2,
    noise_std=0.0,
    prior_mlp_hidden_dim=4,
    prior_mlp_dropout_prob=0.0,
    prior_mlp_activations=nn.Identity,
    perturbation_strategy="fixed_magnitude",
    perturbation_prob=1.0,
    fixed_magnitude_k=1.0,
    perturbation_direction="positive",  # deterministic +1 direction
    use_fixed_scm=True,
)

EXP0_CRITERIA = dict(
    delta_mse=0.01,
    max_epochs_to_converge=30,
)

# ---------- Experiment 1: Single Nonlinear SCM ----------
# 2-layer Tanh SCM, 3 features, 1-2 perturbed, small noise.

EXP1_CONFIG = dict(
    num_features=3,
    seq_len=64,
    batch_size=1,
    emsize=128,
    nlayers=6,
    nhead=4,
    nhid=256,
    dropout=0.0,
    epochs=500,
    steps_per_epoch=100,
    lr=0.0005,
    warmup_epochs=10,
)

EXP1_SCM = dict(
    num_layers=3,
    num_causes=3,
    noise_std=0.05,
    prior_mlp_hidden_dim=8,
    prior_mlp_dropout_prob=0.0,
    prior_mlp_activations=nn.Tanh,
    perturbation_strategy="fixed_magnitude",  # deterministic perturbation (like exp0)
    perturbation_prob=1.0,       # all features perturbed → deterministic target
    perturbation_direction="positive",  # deterministic direction
    fixed_magnitude_k=1.0,
    use_fixed_scm=True,
    mask_supervision=False,      # disable mask supervision (mask is trivially all-1s)
)

EXP1_CRITERIA = dict(
    delta_mse=0.1,
    sign_accuracy=0.85,
    # SCM validity ceiling is ~85% for true deltas (due to stochastic re-propagation
    # with different internals). 0.50 = ~59% of theoretical max.
    scm_validity=0.50,
    loss_reduction=0.80,
)

# ---------- Experiment 2A: Single SCM, 5 features ----------

EXP2A_CONFIG = dict(
    num_features=5,
    seq_len=128,
    batch_size=1,
    emsize=128,
    nlayers=6,
    nhead=4,
    nhid=256,
    dropout=0.0,
    epochs=300,
    steps_per_epoch=100,
    lr=0.0005,
    warmup_epochs=10,
)

EXP2A_SCM = dict(
    num_layers=3,
    num_causes=5,
    noise_std=0.05,
    prior_mlp_hidden_dim=8,
    prior_mlp_dropout_prob=0.2,
    prior_mlp_activations=nn.Tanh,
    perturbation_strategy="uniform_random",
    perturbation_prob=0.5,
    use_fixed_scm=True,
)

EXP2A_CRITERIA = dict(
    delta_mse=0.15,
    sign_accuracy=0.80,
    scm_validity=0.60,
    zero_feature_accuracy=0.90,
)

# ---------- Experiment 2B: Single SCM, 10 features ----------

EXP2B_CONFIG = dict(
    num_features=10,
    seq_len=256,
    batch_size=1,
    emsize=128,
    nlayers=6,
    nhead=4,
    nhid=256,
    dropout=0.0,
    epochs=500,
    steps_per_epoch=100,
    lr=0.0003,
    warmup_epochs=20,
)

EXP2B_SCM = dict(
    num_layers=3,
    num_causes=10,
    noise_std=0.05,
    prior_mlp_hidden_dim=8,
    prior_mlp_dropout_prob=0.2,
    prior_mlp_activations=nn.Tanh,
    perturbation_strategy="uniform_random",
    perturbation_prob=0.5,
    use_fixed_scm=True,
)

EXP2B_CRITERIA = dict(
    delta_mse=0.25,
    sign_accuracy=0.70,
    scm_validity=0.50,
    zero_feature_accuracy=0.85,
)

# ---------- Experiment 3: Small Family of Similar SCMs (ICL) ----------

EXP3_CONFIG = dict(
    num_features=5,
    seq_len=128,
    batch_size=4,
    emsize=128,
    nlayers=6,
    nhead=4,
    nhid=256,
    dropout=0.0,
    epochs=500,
    steps_per_epoch=100,
    lr=0.0003,
    warmup_epochs=20,
)

EXP3_SCM = dict(
    num_layers=3,
    num_causes=5,
    noise_std=0.05,
    prior_mlp_hidden_dim=8,
    prior_mlp_dropout_prob=0.2,
    prior_mlp_activations=nn.Tanh,
    perturbation_strategy="uniform_random",
    perturbation_prob=0.5,
    num_scms=10,  # number of pre-generated SCMs in the family
)

EXP3_CRITERIA = dict(
    delta_mse=0.3,
    scm_validity=0.50,
)

# ---------- Experiment 4: Diverse SCMs, New at Test Time ----------

EXP4_CONFIG = dict(
    num_features=5,
    seq_len=256,
    batch_size=8,
    emsize=256,
    nlayers=8,
    nhead=4,
    nhid=512,
    dropout=0.1,
    epochs=200,
    steps_per_epoch=200,
    lr=0.0001,
    warmup_epochs=20,
    weight_decay=1e-4,
)

EXP4_SCM = dict(
    num_layers=3,
    num_causes=5,
    noise_std=0.05,
    prior_mlp_hidden_dim=16,
    prior_mlp_dropout_prob=0.2,
    prior_mlp_activations=nn.Tanh,
    perturbation_strategy="uniform_random",
    perturbation_prob=0.5,
)

EXP4_CRITERIA = dict(
    delta_mse=1.0,
    scm_validity=0.30,
    context_ablation_gap=0.15,
)

# Registry for CLI lookup
EXPERIMENT_REGISTRY = {
    "exp0": (EXP0_CONFIG, EXP0_SCM, EXP0_CRITERIA),
    "exp1": (EXP1_CONFIG, EXP1_SCM, EXP1_CRITERIA),
    "exp2a": (EXP2A_CONFIG, EXP2A_SCM, EXP2A_CRITERIA),
    "exp2b": (EXP2B_CONFIG, EXP2B_SCM, EXP2B_CRITERIA),
    "exp3": (EXP3_CONFIG, EXP3_SCM, EXP3_CRITERIA),
    "exp4": (EXP4_CONFIG, EXP4_SCM, EXP4_CRITERIA),
}
