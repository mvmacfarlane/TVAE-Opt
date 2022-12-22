import argparse
import torch



def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="CVAE-Opt Training")

    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--problem', type=str, default='TSP')
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument('--batch_size', default=128, type=int)
    
    parser.add_argument('--epoch_size', type=int, default=93440, help='Number of instances used for training')
    parser.add_argument('--nb_epochs', default=300, type=int)
    parser.add_argument('--search_validation_size', default=100, type=int)
    parser.add_argument('--network_validation_size', default=6400, type=int)
    parser.add_argument('--search_space_size', default=100, type=int)
    parser.add_argument('--KLD_weight', default=None, type=float)  # Beta in the paper
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--instances_path', type=str, default="data/tsp/training/tsp_100_instances.zip")
    parser.add_argument('--solutions_path', type=str, default="data/tsp/training/tsp_100_solutions.zip")
    parser.add_argument('--q_percentile', default=99, type=float)


    #Model Type
    parser.add_argument('--model_type', default='None', type=str)


    #Search parameters
    parser.add_argument('--search_timelimit', default=600, type=int)
    #parser.add_argument('--search_iterations', default=300, type=int)  Much better to define the total ammount of search
    parser.add_argument('--total_search', default=1, type=int)
    parser.add_argument('--search_batch_size', default=600, type=int)

    # Differential Evolution
    parser.add_argument('--de_mutate', default=0.3, type=float)
    parser.add_argument('--de_recombine', default=0.95, type=float)

    #New added
    parser.add_argument('--exp_name', default ='None', type=str)
    parser.add_argument('--validation_period', default = 1, type=int)



    #Whether we use latent variable or not
    parser.add_argument('--latent', default = 0, type=int)
    parser.add_argument('--symmetry', default = 0, type=int)
    parser.add_argument('--model_reset', default = 0, type=int)
    parser.add_argument('--latent_search_strat', default = 'random', type=str)
    parser.add_argument('--loss', default = "rl", type=str)
    parser.add_argument('--epoch_iterations', default = 1, type=int)


    


    #Load Model
    parser.add_argument('--model_path', default = "None", type=str)
    



    config = parser.parse_args()

    config.device = torch.device(config.device)


    return config
