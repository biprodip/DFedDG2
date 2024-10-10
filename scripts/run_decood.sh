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



python run_trainer.py \
--algorithm 'FedGHProto' \
--decood_loss_code 'EC' \
--comm 'comm_fed_GH_proto' \
--dataset 'cifar100' \
--num_classes 100 \
--normalize True \
--feat_dim 512 \
--no_shard_per_client 8 \
--num_clients 10 \
--num_rounds 70 \
--dist 'dir' \
--alpha .5 \
--local_epochs 10 \
--num_clusters 1 \
--clustering 'label' \
--noise_level 0 \
--data_type 'img' \
--test_on_cosine True \
--sel_on_kappa False


# python run_trainer.py \
# --algorithm 'Decood' \
# --decood_loss_code 'EC' \
# --comm 'comm_decood' \
# --dataset 'cifar100' \
# --num_classes 100 \
# --normalize True \
# --feat_dim 512 \
# --no_shard_per_client 0 \
# --num_clients 6 \
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


# python run_trainer.py \
# --algorithm 'penz' \
# --comm 'comm_penz' \
# --dataset 'cifar10' \
# --num_classes 10 \
# --normalize True \
# --no_shard_per_client 2 \
# --num_clients 10 \
# --num_rounds 70 \
# --dist 'path' \
# --local_epochs 5 \
# --num_clusters 2 \
# --clustering 'label' \
# --noise_level 0 \
# --data_type 'img' \