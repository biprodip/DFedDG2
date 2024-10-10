import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product

def get_matrix_cosine_similarity_from_grads(local_model_grads):
    """
    return the similarity matrix where the distance chosen to
    compare two clients is set with `distance_type`
    """
    n_clients = len(local_model_grads)
    similarity_matrix_c = torch.zeros((n_clients, n_clients))

    for i, j in tqdm(product(range(n_clients), range(n_clients)), desc='>> similarity', total=n_clients**2, ncols=80):
        grad_1, grad_2 = local_model_grads[i], local_model_grads[j]

        # Convert PyTorch gradients to NumPy arrays
        grad_1_np = [g.detach().cpu().numpy() for g in grad_1]
        grad_2_np = [g.detach().cpu().numpy() for g in grad_2]

        # Reshape the gradients to 1D vectors
        grad_1_flat = np.concatenate([g.flatten() for g in grad_1_np])
        grad_2_flat = np.concatenate([g.flatten() for g in grad_2_np])

        # Calculate cosine similarity
        similarity = cosine_similarity([grad_1_flat], [grad_2_flat])
        similarity_matrix_c[i, j] = torch.tensor(similarity, dtype=torch.float)


    return similarity_matrix_c




def get_cosine_similarity_from_grads(local_model_grads):
    """
    return the similarity matrix where the distance chosen to
    compare two clients is set with `distance_type`
    """
    n_clients = len(local_model_grads)
    
    grad_1, grad_2 = local_model_grads[0], local_model_grads[1]

    # Convert PyTorch gradients to NumPy arrays
    grad_1_np = [g.detach().cpu().numpy() for g in grad_1]
    grad_2_np = [g.detach().cpu().numpy() for g in grad_2]

    # Reshape the gradients to 1D vectors
    grad_1_flat = np.concatenate([g.flatten() for g in grad_1_np])
    grad_2_flat = np.concatenate([g.flatten() for g in grad_2_np])

    # Calculate cosine similarity
    similarity = cosine_similarity([grad_1_flat], [grad_2_flat])
    
    return similarity



# def comp_contribution(norm_diff, unc_util, num_neighbors, device, confidence_level = 0.95):
#     '''
#     num_total_clients : total neighbors from where to select
#     '''
    
#     V_set = [i for i in range(num_neighbors)]
#     marg_grad_util = norm_diff[:, list(V_set)].sum(0)
    
#     print(f"Grad utility {marg_grad_util}")
#     if (torch.max(marg_grad_util)-torch.min(marg_grad_util)) ==0:
#         norm_grad_util = torch.ones(len(marg_grad_util)).to(device)
#     else:
#         norm_grad_util = (marg_grad_util-torch.min(marg_grad_util)) / (torch.max(marg_grad_util)-torch.min(marg_grad_util))
#         #norm_grad_util = 1- norm_grad_util
#         norm_grad_util = norm_grad_util.to(device) 
#         print(f"Norm grad utility {norm_grad_util}")
    
    
#     # print(f"Norm grad rev {norm_grad_util}")
    
#     print(f"Unc utility {unc_util}")
#     if  (torch.max(unc_util)-torch.min(unc_util)) ==0:
#         norm_unc_util = torch.ones(len(unc_util)).to(device)
#     else:
#         norm_unc_util = (unc_util-torch.min(unc_util)) / (torch.max(unc_util)-torch.min(unc_util))
#         norm_unc_util = 1-norm_unc_util 
    
#     print(f"Norm unc utility {norm_unc_util}")

#     #tmp_sum = norm_unc_util + norm_grad_util
#     tmp_sum = norm_unc_util + norm_grad_util  
#     print(f"Total utility {tmp_sum}")
#     weights = tmp_sum/torch.sum(tmp_sum)
    
#     return weights



# def comp_contribution(norm_diff, unc_util, num_neighbors, device, confidence_level = 0.95):
#     '''
#     num_total_clients : total neighbors from where to select
#     '''
#     V_set = [i for i in range(num_neighbors)]
#     marg_grad_util = norm_diff[:, list(V_set)].sum(0)
    
#     print(f"Grad utility {marg_grad_util}")
#     print(f"Unc utility {unc_util}")

#     if (len(marg_grad_util)) == 1:
#         weights = torch.ones(1).to(device)
#     else:
#         epsilon = 0.0008
#         #norm_grad_util = 1 - torch.nn.functional.softmax(marg_grad_util, dim = 0) 
#         #reverse order
#         norm_unc_util = torch.max(unc_util) - unc_util
#         print(f"Rev unc utility {norm_unc_util}")
#         tot_util =  norm_unc_util/3 #norm_grad_util#norm_unc_util#unc_util#marg_grad_util #+ unc_util
        
#         weights = torch.nn.functional.softmax(tot_util, dim = 0)   #desc weight        
#         print(f"Weights: {weights}\n")

#     return weights



def comp_contribution(sample_count, grad_util, unc_util, tot_neighbor, client, neighbors, device):
    '''
    num_total_clients : total neighbors from where to select
    '''
    if (tot_neighbor) == 1:
        weights = torch.ones(1).to(device)
    else:
        epsilon = 0.00008
        #normalize all
        norm_Di = sample_count/torch.sum(sample_count)
        print(f'Norm sample count {norm_Di}')

        norm_grad_util = (grad_util+epsilon)/torch.sum(grad_util+epsilon) 
        print(f'Grad util: {norm_grad_util}')

        unc_util_asc = torch.max(unc_util) - unc_util #reverse
        #unc_util_asc = unc_util
        norm_unc_util_asc = (unc_util_asc+epsilon)/torch.sum(unc_util_asc+epsilon)
        
        #unc_util_asc = torch.clamp(unc_util_asc, epsilon)
        print(f"Rev unc utility {norm_unc_util_asc}")
        
        #unc asc
        #tot_util =  norm_Di * (unc_util_asc/2) #norm_Di * (unc_util_asc/2) #norm_grad_util#norm_unc_util#unc_util#marg_grad_util #+ unc_util
        tot_util =  norm_Di + unc_util_asc #norm_grad_util #norm_grad_util#norm_unc_util#unc_util#marg_grad_util #+ unc_util
        

        #div asc
        #tot_util =  norm_Di * (grad_util/2) #norm_grad_util#norm_unc_util#unc_util#marg_grad_util #+ unc_util
        
        #combo

        weights = torch.nn.functional.softmax(tot_util, dim = 0)   #desc weight        
        print(f"Weights: {weights}\n")

    return weights



# def cont_weights(args, client, neighbors):
     
#      print('Utility computation...')
#      neighbors = neighbors.append(client) #clint placed at end

#      tmp = [c.id for c in neighbors]
#      tmp_samples = [c.train_size for c in neighbors]
#      print(f'Neighbor id: {tmp}')
#      print(f'Neighbor total samples: {tmp_samples}')
#      #get selected client grads
     
#      all_grads = []
#      for cl in neighbors:
#        all_grads += [cl.grad]

#      #grad_sim_mat_c = get_matrix_cosine_similarity_from_grads(all_grads)
#      grad_sim_mat_c = get_cosine_similarity_from_grads(all_grads)
     
#      unc_util = torch.zeros(len(neighbors)).to(args.device)
#      i = 0
#      for c in neighbors:
#         _,_,_,unc_util[i] = c.performance_train(c.val_loader)#torch.div(c.performance_train(c.val_loader),3) #c.unc  #c.performance_train() #compute unc on local validation set
#         #print(f'Returned unc: {unc_util[i]}')
#         unc_util[i]=unc_util[i]
#         #print(f'Scaled unc: {unc_util[i]}')
#         i = i + 1

#      print('Computing weights...')
#      weights_self, weights_neighbors = comp_contribution(grad_sim_mat_c, unc_util, len(neighbors), args.device, args.confidence_level)
#      #print(f'Weights {weights}')


#      return weights_self, weights_neighbors

def cont_weights(args, client, neighbors):
     
     print('Utility computation...')
     
     N_id = [c.id for c in neighbors]
     print(f'Neighbor id: {N_id}')
     
     #sample count
     D_util = torch.zeros(len(neighbors)+1).to(args.device)
     
     D_i = [c.train_size for c in neighbors]
     D_i.append(client.train_size)
     print(D_i)
     D_util[:len(D_i)] = torch.tensor(D_i,dtype=torch.int16).to(args.device)
     
     print(f'Neighbor total samples: {D_util}')
               
     #gradient diversity
     grad_util = torch.zeros(len(neighbors)+1).to(args.device)
     j = 0
     for c in neighbors:
        all_grads = []
        all_grads += [client.grad]
        all_grads += [c.grad]
        sim = get_cosine_similarity_from_grads(all_grads) 
        #print(f'Sim: {sim}')
        grad_util[j] = torch.tensor(sim[0, 0], dtype=torch.float) 
        j = j + 1
     grad_util[j] = torch.tensor([1], dtype=torch.float)  #self similarity 
     print(f'Grad Util: {grad_util}')
                
        
     #uncertainty 
     unc_util = torch.zeros(len(neighbors)+1).to(args.device)
     i = 0
     for c in neighbors:
        _,_,_,unc_util[i] = c.performance_train(c.val_loader)#torch.div(c.performance_train(c.val_loader),3) #c.unc  #c.performance_train() #compute unc on local validation set
        #print(f'Returned unc: {unc_util[i]}')
        #print(f'Scaled unc: {unc_util[i]}')
        i = i + 1
     _,_,_,unc_util[i] = client.performance_train(client.val_loader)    
     print(f'Unc Util: {unc_util}')

     if args.submod:
        weights_neighbors = comp_contribution(D_util, grad_util, unc_util, len(neighbors), client, neighbors, args.device)
     else:
        weights_neighbors = D_util/torch.sum(D_util)
        print(f'Norm sample count {weights_neighbors}')
        
     
     #print(f'Weights {weights}')


     return weights_neighbors