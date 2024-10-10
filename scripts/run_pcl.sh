# python run_trainer.py \
# --algorithm 'Decood' \
# --decood_loss_code 'ECP' \
# --comm 'comm_decood_unc' \
# --dataset 'cifar10' \
# --normalize True \
# --feat_dim 512 \
# --no_shard_per_client 2 \
# --num_clients 5 \
# --num_rounds 5 \
# --dist 'path' \
# --num_clusters 2 \
# --clustering 'noise' \
# --noise_level 0 \
# --data_type 'img' \
# --test_on_cosine True \
# --sel_on_kappa True



# python run_trainer.py \
# --algorithm 'DecoodAvg' \
# --comm 'comm_decood_avg_clust' \
# --dataset 'cifar10' \
# --normalize True \
# --feat_dim 512 \
# --no_shard_per_client 2 \
# --num_clients 20 \
# --num_rounds 20 \



# python run_trainer.py \
# --algorithm 'Decood' \
# --decood_loss_code 'EC' \
# --comm 'comm_decood' \
# --dataset 'cifar100' \
# --num_classes 100 \
# --normalize True \
# --feat_dim 512 \
# --no_shard_per_client 15 \
# --num_clients 10 \
# --num_rounds 70 \
# --dist 'dir' \
# --alpha 5 \
# --num_clusters 1 \
# --clustering 'label' \
# --noise_level 0 \
# --data_type 'img' \
# --test_on_cosine True \
# --sel_on_kappa False



# python run_trainer.py \
# --algorithm 'FedGHProto' \
# --decood_loss_code 'EC' \
# --comm 'comm_fed_GH_proto' \
# --dataset 'cifar100' \
# --num_classes 100 \
# --normalize True \
# --feat_dim 512 \
# --no_shard_per_client 8 \
# --num_clients 10 \
# --num_rounds 70 \
# --dist 'dir' \
# --alpha .5 \
# --local_epochs 10 \
# --num_clusters 1 \
# --clustering 'label' \
# --noise_level 0 \
# --data_type 'img' \
# --test_on_cosine True \
# --sel_on_kappa False

# Set variables based on the input arguments
algorithm='Decood'
comm='comm_decood'
dataset='digit'
feature_iid=1
label_iid=0

# Dynamically create the log file name
log_file="${algorithm}_${comm}_${dataset}_fi${feature_iid}_ln${label_iid}.log"

# Run the Python script and redirect the output to the dynamically generated log file
python run_trainer_pcl_single_iter.py \
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

# python run_trainer_pcl_single_iter.py \
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
# --num_bb 0 \
# --num_clients 5 \
# --feature_iid 1 \
# --label_iid 0 \
# --alpha 1 >tset.log

# python run_trainer.py \
# --algorithm 'Decood' \
# --decood_loss_code 'ECD' \
# --comm 'comm_decood_avg_clust_w' \
# --dataset 'cifar10' \
# --num_classes 10 \
# --normalize True \
# --feat_dim 512 \
# --no_shard_per_client 8 \
# --num_clients 10 \
# --num_rounds 70 \
# --dist 'path' \
# --alpha .1 \
# --local_epochs 3 \
# --num_clusters 1 \
# --clustering 'label' \
# --noise_level 0 \
# --data_type 'img' \
# --test_on_cosine True \
# --sel_on_kappa False




