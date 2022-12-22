import numpy as np
from torch.utils.data import Dataset
import torch


#Used for Reinforcement Learning without solved instances
class Dataset_Random(Dataset):

    def __init__(self, size, problem_size,config):
        self.size = size
        self.problem_size = problem_size
        self.config = config

        self.instances = [np.random.uniform(low = 0, high = 1, size = (problem_size,2)) for x in range(size)]

        self.solutions_1 = [problem_size*[0] for x in range(size)]
        self.solutions_2 = [problem_size*[0] for x in range(size)]

        assert len(self.instances) == len(self.solutions_1)
        assert len(self.instances) >= size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        context = self.instances[idx]


        solution_1 = torch.from_numpy(np.array(self.solutions_1[idx])).to(self.config.device)
        solution_2 = torch.from_numpy(np.array(self.solutions_2[idx])).to(self.config.device)


        return context, solution_1,solution_2

    def set_solutions(self,solutions_1,solutions_2):
        self.solutions_1 = solutions_1
        self.solutions_2 = solutions_2

    #This can be implemented once we introduce a context
    def resample_instances(self):
        self.instances = [np.random.uniform(low = 0, high = 1, size = (self.problem_size,2)) for x in range(self.size)]








def tours_length(locations, tours):
    locations_tour_input = torch.gather(locations, 1, tours.unsqueeze(2).expand_as(locations))
    y = torch.cat((locations_tour_input, locations_tour_input[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask

def update_dynamic(instance, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    instance = instance.clone()
    instance[:, :, 2].scatter_(1, chosen_idx.unsqueeze(1), 0)
    return instance
