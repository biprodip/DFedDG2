#!/usr/bin/env bash

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

data_dir='../FedPCL/data/'
save_folder_name='results/'
params_dir='params/'

# Optional: seed (if you want to override default in Python)
# seed=1234

# ----------------- Setup -----------------
cd '/scratch3/pal194/deccon/'

# Ensure log and result folders exist
mkdir -p logs/experiments
mkdir -p "${save_folder_name}"
mkdir -p "${params_dir}"

echo "Run Deccon.................................................."

algorithm='Decood'
comm='comm_decood_w'

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/experiments/${algorithm}_${comm}_${dataset}_fi${feature_iid}_li${label_iid}_topo${topo}_dist_${dist}_clients${num_clients}_${timestamp}.log"

# ----------------- Run -----------------
python run_trainer_pcl_multi_iter.py \
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
