import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
from torch.distributions import Normal
from torch.distributions import MultivariateNormal



#There is no greedy option here!
class Encoder(nn.Module):
    def __init__(self,latent_size):
        super(Encoder, self).__init__()

        self.latent_size = latent_size

        # Define the encoder & decoder models
        layers = []
        layers.append(nn.Linear(2,128))
        layers.append(nn.ReLU())
        layer_num = 3
        for i in range(layer_num):
            layers.append(nn.Linear(128,128))
            layers.append(nn.ReLU())

        self.latent_embed = nn.Sequential(*layers)

        self.output_head_mu = nn.Linear(128,self.latent_size)
        self.output_head_sigma = nn.Linear(128,self.latent_size)




    def reparameterise(self,mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    #We need to ensure that the encoder tried to put things out of it if you know what I mean
    def forward(self,solution):


        
        solution = solution.float()

        hx = self.latent_embed(solution)

        mu = self.output_head_mu(hx)
        log_sigma = self.output_head_sigma(hx)

        Z = self.reparameterise(mu, log_sigma)
        Z = scale_within_unit(Z)




        


        


        #unit = torch.ones(size = (Z.shape[0],1)).to(Z.device)
        #zeros = torch.zeros(size = (Z.shape[0],1)).to(Z.device)
        #indicator = torch.cat((unit,zeros),dim=1)
        #Z = Z*indicator
        
        #Making it the identity encoder
        #Z = solution

        return Z, mu, log_sigma


#Currently I dont think that this function is correct anymore!
#Change this so we are just adding activations
def scale_within_unit(vectors):

    """
    lengths = vectors.norm(dim=1,p=2).unsqueeze(dim=1)
    outside_unit = lengths > 1
    lengths = outside_unit.int()*lengths + (1-outside_unit.int())
    vectors = vectors/lengths
    """

    #Just scale within the square
    m = nn.Tanh()
    vectors = m(vectors)

    return vectors




class Decoder(nn.Module):
    def __init__(self,latent_size):
        super(Decoder, self).__init__()


        self.latent_size = latent_size



        #Think this model needs to be a bit bigger
        layers = []
        layer_num = 3


        layers.append(nn.Linear(self.latent_size,128))
        layers.append(nn.Tanh())
        for i in range(layer_num):
            layers.append(nn.Linear(128,128))
            layers.append(nn.Tanh())

        self.latent_embed = nn.Sequential(*layers)


        #No these output heads also needs to be scaled I beleive somehow since they should be outputing within the distribution of course
        layers = []
        layers.append(nn.Linear(128,128))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(128,2))
        layers.append(nn.Tanh())
        self.output_head_mu = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(128,128))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(128,2))
        layers.append(nn.Tanh())
        self.output_head_sigma = nn.Sequential(*layers)


    def reparameterise(self,mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    #How do we ensure that is is between certain values, well technically the 
    def forward(self, context, solution, Z, teacher_forcing,greedy):




        tour_logp = []

        #Single straight prediction from the output
        hx = self.latent_embed(Z)
        mu = self.output_head_mu(hx)
        log_sigma = self.output_head_sigma(hx)

        #print(torch.exp(log_sigma)[0])

        #import time
        #time.sleep(2)


        #Choose action
        if not teacher_forcing:
            #action = self.reparameterise(mu,log_sigma)
            #Clipping only outside of training
            if greedy:
                action = mu
                action = torch.clip(mu,-1,1)

                #So this is basically decoding a greedy solution which is not what we want I think
            else:
                action = self.reparameterise(mu,log_sigma)
                action = torch.clip(action,-1,1)

        else:
            action = solution

        cov_mat = torch.diag_embed(torch.exp(log_sigma)).to(mu.device)
        m = MultivariateNormal(mu.double(), cov_mat.double())


        tour_logp = m.log_prob(action.double())






        return None,action, tour_logp,mu,log_sigma




#General function solver
class Improver(nn.Module):
    def __init__(self):
        super(Improver, self).__init__()

        #Improvement Layers
        layers = []
        layers.append(nn.Linear(2,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,2))

        self.improve_solution = nn.Sequential(*layers)


    #forward pass on specific solution to be encoded
    #We dont even use the config yet,why is that
    def forward(self,Z):

        Z_improved = self.improve_solution(Z)
        Z_improved = scale_within_unit(Z_improved)
    
        return Z_improved

#General function solver
class NN_Solver(nn.Module):
    def __init__(self, config):
        super(NN_Solver, self).__init__()

        self.latent_size = 2

        self.encoder = Encoder(latent_size = self.latent_size)
        self.decoder = Decoder(latent_size = self.latent_size)
        self.improve_solution = Improver()

        


    #forward pass on specific solution to be encoded
    #We dont even use the config yet,why is that
    def forward(self,context,solution,config,teacher_forcing = True,greedy = False):


        #Do we want this variance at test time, interesting question
        #Encoder is not running correctly
        Z, mu, log_var = self.encoder(solution)


        #Simply Improving the solution
        Z_improved = self.improve_solution(Z)
        




        #Just output the vector the probability and the log probability, not sure why we have two of the others? What is the point

        #If it is greedy though we should be encoding correctly as well right?

        _, tour_idx, tour_logp,mu2,log_sigma2 = self.decoder(
            
            context = context,
            solution = solution,
            Z = Z,
            teacher_forcing = teacher_forcing,   
            greedy = greedy,        

        )

        _, tour_idx_improved,tour_logp_improved,_,_ = self.decoder(
            
            context = context,
            solution = solution,
            Z = Z_improved,
            teacher_forcing = False,   
            greedy = True,        

        )

                
        return None, mu, log_var, Z, tour_idx, tour_logp,solution,tour_idx_improved,tour_logp_improved

    #This just greedily decodes what you think the solution is for a given latent variable
    #Only reason we have this decoding step is that we output a latent one by one
    #This still seems reasonable so I think we should keep it
    def decode(self,solutions,context,latent_vector,config,greedy = False):

        #why cant we pass greedy to the decoder
        output_prob, tour_idx, tour_logp,_,_ = self.decoder(

            context = context,
            solution = solutions,
            Z = latent_vector,
            teacher_forcing = False,
            greedy = greedy,
            
        )

        return output_prob, tour_idx, tour_logp

    """
    #Not actually sure when we need to use this
    def reset_decoder(self, batch_size, config):
        self.instance_hidden = None
        self.dummy_solution = torch.zeros(batch_size, 1).long().to(config.device)
    """

