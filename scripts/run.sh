# python run_trainer.py \
# --algorithm 'FedAvg' \
# --comm 'comm_gossip' \
# --dataset 'modelnet10' \
# --data_type 'pc' \
# --isCFL False \
# --clustering None \
# --num_clients 3 \
# --num_rounds 2 \


# python run_trainer.py \
# --algorithm 'FedAvg' \
# --comm 'comm_gossip' \
# --dataset 'cifar10' \
# --data_type 'img' \
# --isCFL False \
# --clustering None \
# --num_clients 3 \
# --num_rounds 2 \


python run_trainer.py \
--algorithm 'Decood' \
--decood_loss_code 'ECP' \
--comm 'comm_decood' \
--dataset 'cifar10' \
--normalize True \
--feat_dim 512 \
--no_shard_per_client 2 \
--num_clients 5 \
--num_rounds 5 \
--dist 'path' \
--num_clusters 2 \
--clustering 'label' \
--noise_level 0 \
--data_type 'img' \
--test_on_cosine True \
--sel_on_kappa True


# python run_trainer.py \
# --algorithm 'vmf' \
# --comm 'comm_em_vmf' \
# --dataset 'cifar10' \
# --data_type 'img' \
# --num_clients 10 \
# --num_rounds 10 \
# --check_running True \
# --no_shard_per_client 2 \
# --clustering 'label' \
# --dist 'path'
