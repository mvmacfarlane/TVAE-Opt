import torch
import numpy as np
import time
from de import minimize
import logging
import os
from tqdm import tqdm
from toy import cost_func as cost_fn



#Too much in the frame of TSP, need to generalise the concepts here
#I am not sure why this is nessesary
def decode(Z, model, config, instance, cost_fn,greedy):
    Z = torch.Tensor(Z).to(config.device)
    with torch.no_grad():

        #Sort out the potential to be greedy here
        #I am not sure that this is correct, although I think it might have the correct ammount of arguemtns
        tour_probs, tour_idx, tour_logp = model.decode(instance, Z, config,greedy = greedy)

    #This is the wrontg cost fun
    costs = cost_fn(tour_idx.long())
    return tour_idx, costs.tolist()


#Solving Single Problem
#This function needs to be completely fixed
def solve_instance(model, context, config, cost_fn):

    #Relevant variables
    batch_size = config.search_batch_size
    search_iterations = int(config.total_search/config.search_batch_size)

    #I dont like this switching
    best_cost, best_solution = search(
        
        model = model,
        config = config,
        context = context,
        cost_func = cost_fn,
        batch_size=  batch_size,

    )


    return best_cost,best_solution


#We need to get a greedy solution
def generate_solutions_instances(model,config,validation_context,latent_search_strat = 'random',cost_func = None):

    instance_num = len(validation_context.instances)
    contexts = validation_context.instances

    #Put model into evaluation mode
    model.eval()


    #Storing statistics
    cost_values = []
    runtime_values = []

    
    for context in tqdm(contexts):
        start_time = time.time()

        objective_value,_,_ = solve_instance(

            model = model,
            instance = context,
            config = config,
            cost_fn = cost_func,
            latent_search_strat = latent_search_strat,

        )

        runtime = time.time() - start_time

        cost_values.append(objective_value)
        runtime_values.append(runtime)

    """
    if not solutions and verbose:
        results = np.array(list(zip(cost_values, runtime_values)))
        np.savetxt(os.path.join(config.output_path, "search", 'results.txt'), results, delimiter=',', fmt=['%s', '%s'],
                   header="cost, runtime")
        logging.info("Final search results:")
        logging.info(f"Mean costs: {np.mean(cost_values)}")
        logging.info("Mean std: {}".format(np.mean(np.std(gap_values))))
        logging.info(f"Mean runtime: {np.mean(runtime_values)}")
    """

    return np.mean(runtime_values),np.mean(runtime_values)



