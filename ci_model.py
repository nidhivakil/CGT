import torch
from torch import nn

class CI_Predictor_w_MLP(nn.Module):
  def __init__(self, max_num_nodes, nb_of_indices):
    super(CI_Predictor_w_MLP, self).__init__()

    self.ci_predictor = nn.Sequential(
      nn.Linear(max_num_nodes * max_num_nodes, max_num_nodes),
      nn.ReLU(),
      nn.Linear(max_num_nodes, max_num_nodes),
      nn.ReLU(),
      nn.Linear(max_num_nodes, nb_of_indices),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.ci_predictor(x)