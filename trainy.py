from config_train import get_config
from torch.utils.data import DataLoader
import torch
import numpy as np
import datetime
import os
import logging
import sys
from search_control import generate_solutions_instances
from model import NN_Solver
from tqdm import tqdm
from copy import deepcopy
from utils import read_instance_data
import math

#Custom functions for our problem
import toy
from toy import cost_func
from general import Dataset_Random

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import time

from torch.distributions.uniform import Uniform

from IPython.display import clear_output


def plot_tensors(latents_train,latents_test,cost_train,cost_test,cost_test2,cost_func,config,epoch_idx):


    
    #Random solutions
    combined_original = torch.cat((latents_test.to("cpu"),cost_func(latents_test).unsqueeze(dim=1).to("cpu")),dim=1)

    #Encoded solutions int he latent space and the score that the deocder gives them
    combined_train = torch.cat((latents_train.to("cpu"),cost_train.to("cpu")),dim=1)
    combined_test = torch.cat((latents_train.to("cpu"),cost_test.to("cpu")),dim=1)

    #Random latents to see the whole space
    combined_test2 = torch.cat((latents_test.to("cpu"),cost_test2.to("cpu")),dim=1)
    

    overall_original = pd.DataFrame(combined_original.detach().numpy(),columns = ['x','y','ov'])

    #These two have encoded real solutions into the latent space
    overall_train = pd.DataFrame(combined_train.detach().numpy(),columns = ['x','y','ov'])
    overall_test = pd.DataFrame(combined_test.detach().numpy(),columns = ['x','y','ov'])

    overall_test2 = pd.DataFrame(combined_test2.detach().numpy(),columns = ['x','y','ov'])

    clear_output(wait=True)
    plt.clf()
    
    fig, axs = plt.subplots(ncols=5,figsize=(100,30))

    sns.scatterplot(data=overall_original, x="x", y="y", hue="ov",hue_norm = (-2,3),palette = "icefire",ax=axs[0],s=400)
    sns.scatterplot(data=overall_train, x="x", y="y", hue="ov",hue_norm = (-2,3),palette = "icefire",ax=axs[1],s=400)
    sns.scatterplot(data=overall_test, x="x", y="y", hue="ov",hue_norm = (-2,3),palette = "icefire",ax=axs[2],s=400)
    sns.scatterplot(data=overall_test2, x="x", y="y", hue="ov",hue_norm = (-2,3),palette = "icefire",ax=axs[3],s=400)
    sns.scatterplot(data=overall_test2, x="x", y="y", hue="ov",hue_norm = (2.94,3),palette = "icefire",ax=axs[4],s=400)

    axs[0].set_ylim(-1,1)
    axs[1].set_ylim(-1,1)
    axs[2].set_ylim(-1,1)
    axs[3].set_ylim(-1,1)

    axs[0].set_xlim(-1,1)
    axs[1].set_xlim(-1,1)
    axs[2].set_xlim(-1,1)
    axs[3].set_ylim(-1,1)

    save_path =  os.path.join(config.output_path, "latent_images","out:{}.png".format(epoch_idx))

    plt.show()
    fig.savefig(save_path) 





    

    





#Sample random solutions
def sample_solutions(model,batch_size,solution_num,config,random):

    solutions_1 = []
    solutions_2 = [] 
    epoch_num = math.ceil(solution_num/batch_size)


    #We have to generate solutions in batches which is very interesting since we can in theory try to improve within these as we try to generate data from which we will improve on
    for i in range(epoch_num):

        #Random Latents
        solution_1 = generate_uniform_vectors(batch_size,dim =2).to(config.device)
        solution_2 = generate_uniform_vectors(batch_size, dim =2).to(config.device)

        #Decode new solutions
        if not random:


            _, solution_1, _ = model.decode(
                
                solutions = None,
                context = None,
                latent_vector = solution_1,
                config = config,
                greedy = False,
                
            )

            _, solution_2, _ = model.decode(
                
                solutions = None,
                context = None,
                latent_vector = solution_2,
                config = config,
                greedy = False,
                
            )
            

        solutions_1 = solutions_1 + solution_1.tolist()
        solutions_2 = solutions_2 + solution_2.tolist()

    return solutions_1,solutions_2


def generate_data(model,batch_size,solution_num,config,random):

    solution_1 = generate_uniform_vectors(batch_size,dim =2).to(config.device)
    solution_2 = generate_uniform_vectors(batch_size, dim =2).to(config.device)

    #Decode new solutions
    if not random:

        _, solution_1, _ = model.decode(
            
            solutions = None,
            context = None,
            latent_vector = solution_1,
            config = config,
            greedy = False,
            
        )

        _, solution_2, _ = model.decode(
            
            solutions = None,
            context = None,
            latent_vector = solution_2,
            config = config,
            greedy = False,
            
        )

    return solution_1,solution_2
        



#Just getting rid of the advatnage multiplication
def calculate_weighted_RC_loss(solution_logp,advantage):

    advantage = torch.exp(0.5*advantage)
    #solution_probability = solution_logp.sum(dim=1)


    RL = -(solution_logp*advantage).sum()


    return RL

def calculate_KLD_loss(mean, log_var,advantage):

    advantage = torch.exp(0.5*advantage)

    x = (1 + log_var - mean.pow(2) - log_var.exp())

    #x = x*advantage.unsqueeze(dim=1)

    KLD = -0.5 * torch.sum(x)

    return KLD


#We want to minimise the distance to the centre sort of
def calculate_C_loss(mean,advantage):

    advantage = torch.exp(0.5*advantage)

    #Is this really calculatiung the correct distance
    distance_to_centre = torch.sqrt(torch.sum(mean*mean,dim=1))

    CL = (distance_to_centre*advantage).sum()

    return CL



def train_epoch(model,training_dataset,config, epoch_idx, optimizer,cost_func,writer):

    model.train()

    #We have switched off resampling instances so we cna really just monitor how the encoder trains
    #If we have different contexts for each problem then we resample them here
    #training_dataset.resample_instances()

    training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, num_workers=0, shuffle=False)

    #Generate solutions and add them to dataset
    #We need to sampling this using the model
    solutions_1,solutions_2 = sample_solutions(model,batch_size = config.batch_size, solution_num = config.epoch_size,config  = config,random = epoch_idx == 1)

    
    #Evaluating the solutions that we generated
    sample_costs = cost_func(torch.from_numpy(np.array(solutions_1+solutions_2)))


    #if epoch_idx == 1:
    #Remove the resetting of solutions which I think could be very important
    training_dataset.set_solutions(solutions_1,solutions_2)
    training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=False)
    

    #You should probably do multiple loops through single epoch
    for i in range(1):

        #Variables to track during training
        loss_RC_values = []
        loss_KLD_values = []
        loss_Centering_values = []
        mean_encoder_logvar = []
        loss_total = []



        #Update Model
        #for batch_id,batch in tqdm(enumerate(training_dataloader)):
        for i in range(46): #182

            batch_solutions_1, batch_solutions_2 = generate_data(
                
                model = model,
                batch_size = config.batch_size,
                solution_num = config.epoch_size,
                config  = config,
                random = epoch_idx == 1,

            )

        
            #This might be the wrong part
            #advantage_estimate = reward_1 - reward_2
            #indicator = (advantage_estimate > 0)
            #solution_best = (indicator.int().unsqueeze(dim=1)*batch_solutions_1 +(1-indicator.int()).unsqueeze(dim=1)*batch_solutions_2).long()
            #reward_2 = cost_func(batch_solutions_2)
            #_, batch_solutions_1, batch_solutions_2 = batch


            reward_1 = cost_func(batch_solutions_1)
            reward_2 = cost_func(batch_solutions_2)



            #advantage_estimate = reward_1
            #advantage_estimate = torch.ones(reward_1.shape).to(reward_1.device)
            #solution_best = batch_solutions_1

            #advantage_estimate = reward_1 - reward_2
            #indicator = (advantage_estimate > 0)
            #solution_best = (indicator.int().unsqueeze(dim=1)*batch_solutions_1 +(1-indicator.int()).unsqueeze(dim=1)*batch_solutions_2).long()

            solution_best = torch.cat((batch_solutions_1,batch_solutions_2),dim=0)

            #Now we need to normalise the advantages as they get very small later on
            advantage_estimate = torch.cat((reward_1 - reward_2,reward_2 - reward_1),dim=0)
            adv_mean = torch.mean(advantage_estimate)
            adv_std = torch.std(advantage_estimate)
            normalized_advantage = (advantage_estimate - adv_mean)/adv_std

            advantage_estimate = normalized_advantage




            #advantage_estimate = torch.ones(size = (solution_best.shape[0],1)).to(solution_best.device)






            _,mean,log_var,Z, tour_idx,tour_logp,_ = model(context = None,solution = solution_best,config = config)



            # Calculate Losses
            loss_RC = calculate_weighted_RC_loss(tour_logp,advantage_estimate)
            loss_KLD = calculate_KLD_loss(mean, log_var,advantage_estimate) 
            loss_Centering  = calculate_C_loss(mean,advantage_estimate)
            
            loss_score = (reward_1,cost_func(tour_idx))

            

            loss = (loss_RC + loss_KLD * config.KLD_weight)
            #loss = loss_RC

            
            
            #Update step
            optimizer.zero_grad()
            assert not torch.isnan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            #Storing loss terms
            loss_RC_values.append(loss_RC.item())
            loss_KLD_values.append(loss_KLD.item())
            loss_Centering_values.append(loss_Centering.item())
            loss_total.append(loss.item())

            #Storing useful data for interpretation
            mean_encoder_logvar.append(torch.exp(log_var).mean().item())


        performance = sum(loss_total)/len(loss_total)

        #print("Performance:{}".format(performance))
        #writer.add_scalar("Total Loss:{}".format(epoch_idx), performance, i)





    


    return loss_RC_values,loss_KLD_values,loss_Centering_values,mean_encoder_logvar,model,sample_costs,training_dataset



def generate_uniform_vectors(number,dim):

    coordinates = Uniform(-1,1).sample((number,2)) 

    return coordinates





def evaluate_epoch(model,training_dataset,config, epoch_idx, optimizer,cost_func,eval_num):

    model.eval()

    #We dont want a battle we just want one set of solutions for a range of latent vectors


    training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=False)

    for batch in training_dataloader:
        _, batch_solutions_1, batch_solutions_2 = batch


        #Cant we just decode the whole thing here
        _,mean,log_var,Z, solutions_test,tour_logp,solutions_train = model(context = None,solution = batch_solutions_1,config = config,teacher_forcing = False,greedy = True)

        break


    #Testing the coverage of the latent space
    uniform_latents = generate_uniform_vectors(number = eval_num,dim = 1)

    #Generate solutions for this
    _, solutions_test_2, _ = model.decode(
        
        solutions_train,
        None,
        uniform_latents.to(Z.device),
        config,
        greedy = True,
        
    )



    costs_train = cost_func(solutions_train).unsqueeze(dim=1)
    costs_test = cost_func(solutions_test).unsqueeze(dim=1)
    costs_test2 = cost_func(solutions_test_2).unsqueeze(dim=1)



    return Z,uniform_latents,costs_train,costs_test,costs_test2



def train(model,config,cost_func,run_id):

    #Loading model
    if config.model_path is not "None":
        model.load_state_dict(torch.load(config.model_path)['parameters'])

    #Tensorboard Layout
    layout = {
        "ABCDE": {
            "optimality_gap": ["Multiline", ["optimality_gap/random", "optimality_gap/de"]],
            "optimality_gap_budget:1": ["Multiline", ["optimality_gap_budget:1/random", "optimality_gap_budget:1/de"]],
        },
    }

    #Setting up tensorboard
    runs_path = os.path.join(config.output_path_fixed,"runs",str(config.exp_name))
    writer = SummaryWriter(runs_path)
    writer.add_custom_scalars(layout)

    #Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    #This in principle should not be nessesary
    #Basically all we are trying to get here is the validation data
    #Works for now but change soon
    #Should program it like there is a context
    epoch_size_freeze = config.epoch_size
    config.epoch_size = 93440
    #_, validation_data = read_instance_data(config)

    #Dataset object
    training_dataset = Dataset_Random(config.epoch_size, config.problem_size,config)
    #validation_dataset = Dataset_Random(config.network_validation_size, config.problem_size, config)


    config.epoch_size = epoch_size_freeze

    #Setting initial variables
    #best_avg_gap = np.inf

    for epoch_idx in range(1, config.nb_epochs + 1):

        #Training
        loss_RC_values,loss_KLD_values,loss_C_values,mean_encoder_logvar,model,sample_costs,training_dataset = train_epoch(model,training_dataset, config, epoch_idx, optimizer,cost_func,writer)

        #Validation
        latents_train,latents_test,cost_train,cost_test,cost_test2 = evaluate_epoch(model,training_dataset,config, epoch_idx, optimizer,cost_func,eval_num = 512)



        plot_tensors(latents_train,latents_test,cost_train,cost_test,cost_test2,cost_func,config,epoch_idx)


        #Logging Training Variables
        writer.add_scalar("Loss_RC", sum(loss_RC_values)/len(loss_RC_values), epoch_idx)
        writer.add_scalar("Loss_KLD", sum(loss_KLD_values)/len(loss_KLD_values), epoch_idx)
        writer.add_scalar("Loss_C", sum(loss_C_values)/len(loss_C_values), epoch_idx)
        writer.add_scalar("Encoder_Variance", sum(mean_encoder_logvar)/len(mean_encoder_logvar), epoch_idx)

        writer.add_scalar("Sample Mean", torch.mean(cost_test2), epoch_idx)
        writer.add_scalar("Sample Max", torch.max(cost_test2), epoch_idx)
        writer.add_scalar("Sample Min", torch.min(cost_test2), epoch_idx)
        writer.add_scalar("Sample Var", torch.var(cost_test2), epoch_idx)


    model_data = {
        'parameters': model.state_dict(),
        'code_version': VERSION,
        'problem': config.problem,
        'problem_size': config.problem_size,
        'Z_bound': 0,
        'avg_gap': 0,
        'training_epochs': epoch_idx,
        'model': "VAE_final"
    }



    # Save the last model after the end of the training
    torch.save(
        
        model_data,
        os.path.join(config.output_path,
        "models",
        "model_{0}_final.pt".format(run_id, epoch_idx)),
    )


VERSION = "0.4.0"
if __name__ == "__main__":
    run_id = np.random.randint(10000, 99999)
    now = datetime.datetime.now()

    config = get_config()

    if config.output_path == "":
        config.output_path = os.getcwd()
    config.output_path_fixed = config.output_path
    config.output_path = os.path.join(config.output_path,"experiment_info", config.exp_name + ":_" + str(now.day) + "." + str(now.month) +
                                      "." + str(now.year) + "_" + str(run_id))
    os.makedirs(os.path.join(config.output_path, "models"))

    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')
    logging.info("Started Training Run")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("Version: {0}".format(VERSION))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")



    
    Model = NN_Solver(config).to(config.device)


    train(Model, config,toy.cost_func,run_id = run_id)


"""
if (epoch_idx-1) % config.validation_period == 0:

    #Hardcoded for now, later need to figure out what I want to do with this
    config.search_space_bound = 1loss


    #Need to investigate this function as It is a bit of a mess

    
    avg_gap, _,_,budget_gaps = generate_solutions_instances(
        
        model = model,
        config = config,
        validation_context = validation_data,

        greedy = True, #I am not sure if this makes sense surely it is a given that it is greedy
        latent_search_strat = 'random',
        cost_func = cost_func,
        
    )

    # Save most recent version of the model
    model_data = {
        'parameters': model.state_dict(),
        'code_version': VERSION,
        'problem': config.problem,
        'problem_size': config.problem_size,
        'Z_bound': 0,
        'avg_gap': avg_gap,
        'training_epochs': epoch_idx,
        'model': "VAE_final"
    }

    torch.save(model_data, os.path.join(config.output_path, "models",
                                        "model_{0}.pt".format(run_id, epoch_idx)))


    writer.add_scalar("optimality_gap_budget:1/random", budget_gaps[1], epoch_idx)
"""