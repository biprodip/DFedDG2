# # run FedEM
# echo "Run FedEM"
# python run_experiment.py femnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.03 \
#  --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1


# Set variables based on the input arguments
algorithm='Decood'
comm='comm_decood'
dataset='digit'
feature_iid=1
label_iid=0

# Dynamically create the log file name
log_file="${algorithm}_${comm}_${dataset}_fi${feature_iid}_li${label_iid}.log"

# Run the Python script and redirect the output to the dynamically generated log file
python run_trainer_pcl_multi_iter.py \
--algorithm "$algorithm" \
--decood_loss_code 'EC' \
--comm "$comm" \
--num_classes 10 \
--normalize True \
--feat_dim 512 \
--no_shard_per_client 0 \
--num_rounds 5 \
--dist 'dir' \
--local_epochs 1 \
--num_clusters 1 \
--clustering 'label' \
--noise_level 0 \
--data_type 'img' \
--test_on_cosine True \
--sel_on_kappa False \
--dataset "$dataset" \
--num_trials 3 \
--num_bb 0 \
--num_clients 5 \
--feature_iid "$feature_iid" \
--label_iid "$label_iid" \
--alpha 1 >"$log_file"



# python run_trainer_pcl_multiple_iter.py \
# --algorithm 'Decood' \
# --decood_loss_code 'EC' \
# --comm 'comm_decood' \
# --num_classes 10 \
# --normalize True \
# --feat_dim 512 \
# --no_shard_per_client 0 \
# --num_rounds 5 \
# --dist 'dir' \
# --local_epochs 1 \
# --num_clusters 1 \
# --clustering 'label' \
# --noise_level 0 \
# --data_type 'img' \
# --test_on_cosine True \
# --sel_on_kappa False \
# --dataset digit \
# --num_trials 3 \
# --num_clients 5 \
# --num_bb 0 \
# --feature_iid 1 \
# --label_iid 0 \
# --alpha 1 >tset.log


# python run_em_multi.py \
# digit FedEM \
# --n_learners 3 \
# --num_trials 3 \
# --n_rounds 5 \
# --bz 128 \
# --lr 0.03 \
# --lr_scheduler multi_step \
# --log_freq 5 \
# --device cuda \
# --optimizer sgd \
# --seed 1234 \
# --verbose 1 \
# --dataset digit \
# --num_trials 3 \
# --num_clients 5 \
# --feature_iid 1 \
# --label_iid 0 \
# --alpha 1 >output_filn_em.log