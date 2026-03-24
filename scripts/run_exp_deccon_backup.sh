#!/usr/bin/env bash
# =============================================================================
# run_exp_deccon_backup.sh — Launch a DECOOD-vMF gossip training experiment
# =============================================================================
#
# Usage:
#   bash scripts/run_exp_deccon_backup.sh
#
# What this script does:
#   1. Sets all hyperparameters (see Config section below).
#   2. Creates required output directories (logs/, results/, params/).
#   3. Invokes run_trainer_dfeddg2.py with the configured flags.
#   4. Redirects all stdout to a timestamped log file under logs/experiments/.
#
# Key configuration variables
# ---------------------------
#   dataset          : Dataset name. 'digit' uses five domains (MNIST, USPS,
#                      SVHN, SynthDigits, MNIST-M); 'domainnet' uses DomainNet.
#   feature_iid      : 1 = IID feature distribution (same domain per client),
#                      0 = non-IID (different domains).
#   label_iid        : 1 = IID label distribution, 0 = non-IID (Dirichlet).
#   alpha            : Dirichlet concentration parameter for label split.
#                      Smaller α → more label non-IID. Ignored if label_iid=1.
#   lr               : SGD learning rate. Typical values: 0.05 / 0.1 / 0.3.
#   decood_loss_code : Which loss combination to use:
#                        'EC'  → Cross-Entropy + λ·CompLoss
#                        'CD'  → CompLoss + DisLoss only
#                        'ECD' → CE + λ·(CompLoss + DisLoss)
#   num_trials       : Number of independent runs (different random seeds).
#   num_clients      : Number of federated nodes. 5 for digit, 20 for domainnet.
#   n_rounds         : Total communication rounds.
#   feat_dim         : Embedding / prototype dimension (projection head output).
#   local_epochs     : Local training epochs per round per client.
#   local_bs         : Local batch size. Keep ≤ smallest client dataset size.
#   tau              : Temperature for supervised contrastive loss.
#   weighted_adj_mat : 1 = use data-heterogeneity-weighted adjacency matrix
#                      (update_adjacency_matrix), 0 = use uniform weights.
#   use_imb_loss     : 1 = enable class-imbalance-aware loss weighting, 0 = off.
#   backbone         : Backbone architecture. Options:
#                        'mobilenet_proj'  — MobileNetV2 + projection head
#                        'resnet18_proj'   — ResNet-18  + projection head
#                        'resnet34_proj'   — ResNet-34  + projection head
#   topo             : Graph topology. 'ring' | 'sparse' | 'fc'.
#   dist             : Data distribution. 'dir' (Dirichlet) | 'path' (pathological).
#   data_dir         : Path to the root data directory (relative to repo root).
#   save_folder_name : Directory for result CSV / pickle output.
#   params_dir       : Directory for model checkpoint files.
# =============================================================================

# ----------------- Config -----------------
dataset='digit'   # 'domainnet'
feature_iid=1
label_iid=0
alpha=1          # Dirichlet parameter
lr=0.3           # 0.05 / 0.1 / 0.3 for digit
decood_loss_code='EC'
num_trials=1
num_clients=5    # 20 / 5
n_rounds=2
feat_dim=512
local_epochs=1
local_bs=32      # 64 is often > sample size
tau=0.1
weighted_adj_mat=1
use_imb_loss=0
backbone='mobilenet_proj'   # resnet18_proj / resnet34_proj

topo='ring'
dist='dir'

data_dir='../../data/'
save_folder_name='results/'
params_dir='params/'

# seed=1234

# ----------------- Setup -----------------
# cd '/scratch3/pal194/deccon/'

# Ensure log and result folders exist
mkdir -p logs/experiments
mkdir -p "${save_folder_name}"
mkdir -p "${params_dir}"

echo "Run Deccon.................................................."

algorithm='decood'
comm='comm_decood_w'

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/experiments/${algorithm}_${comm}_${dataset}_fi${feature_iid}_li${label_iid}_topo${topo}_dist_${dist}_clients${num_clients}_${timestamp}.log"

cd ..
# ----------------- Run -----------------
python run_trainer_dfeddg2.py \
  --algorithm "${algorithm}" \
  --comm "${comm}" \
  --topo "${topo}" \
  --num_classes 10 \
  --normalize True \
  --decood_loss_code "${decood_loss_code}" \
  --lr "${lr}" \
  --feat_dim "${feat_dim}" \
  --num_rounds "${n_rounds}" \
  --dist "${dist}" \
  --local_epochs "${local_epochs}" \
  --num_clusters 1 \
  --clustering 'label' \
  --noise_level 0 \
  --data_type 'img' \
  --test_on_cosine True \
  --sel_on_kappa False \
  --dataset "${dataset}" \
  --num_trials "${num_trials}" \
  --num_clients "${num_clients}" \
  --feature_iid "${feature_iid}" \
  --label_iid "${label_iid}" \
  --local_bs "${local_bs}" \
  --tau "${tau}" \
  --weighted_adj_mat "${weighted_adj_mat}" \
  --backbone "${backbone}" \
  --use_imb_loss "${use_imb_loss}" \
  --alpha "${alpha}" \
  --data_dir "${data_dir}" \
  --save_folder_name "${save_folder_name}" \
  --params_dir "${params_dir}" \
  > "${log_file}"

echo "Log saved to: ${log_file}"