#!/bin/bash

# # Run fedavg
# python run_ood_trainer_fedavg.py

# # Run ditto
# #python run_ood_trainer_ditto.py

# # Run fedproto
# python run_ood_trainer_fedproto.py

# # Run fedGH
# python run_ood_trainer_fedgh.py

# # # Run Decood
# # python run_ood_trainer_fedghproto.py

#Run decood
python run_ood_trainer_decood.py

# Run Decood
python run_ood_trainer_decood_avg_clust.py
