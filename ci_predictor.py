
import os
os.system("pip install tabulate")
import sys
sys.path.append("/data_augmentation/auto_encoder_study")
import torch
from torch import nn
import numpy as np
import argparse
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import copy
import random
from ci_model import CI_Predictor_w_MLP

def get_std_name(args):
  ds = args.dataset
  hop = args.hop
  max_num_nodes = args.max_num_nodes
  k_percent_str = args.k_percent_str
  lr = str(args.lr).replace(".", "_")
  e = args.epochs

  name = f'ci_pred_{ds}_h_{hop}_m_{max_num_nodes}_k_{k_percent_str}_lr_{lr}_e_{e}'
  return name

parser = argparse.ArgumentParser('Interface for CI Predictor')
parser.add_argument('--dataset', type=str, default='citeseer', help='dataset name - arxiv,wordnet, citeseer')
# dataset arguments
parser.add_argument('--hop', type=int, default=3, help="3,2")
parser.add_argument('--max_num_nodes', type=int, default=50, help="50,20, 60, 70")
parser.add_argument('--k_percent_str', type=str, default="0_5", help="0_5,0_6")
parser.add_argument('--device', type=str, default="cuda:0", help="gpu-device")
parser.add_argument('--lr', type=float, default=0.001, help="0.0001")
parser.add_argument('--epochs', type=int, default=200, help="100,200")

args = parser.parse_args()
print(args)

std_name = get_std_name(args)
sys.stdout = sys.stderr = open(f"/data/nidhi/data_augmentation/ci_logs/{std_name}.log", "w")
print(args)
sys.stdout.flush()


dataset = args.dataset
hop = args.hop
k_percent_str = args.k_percent_str
max_num_nodes = args.max_num_nodes

complexity_difference_file = f"/data/nidhi/data_augmentation/auto_encoder_study_dataset/cgg_{dataset}_graphs_hop_{hop}_max_node_{max_num_nodes}_complexity_difference_max_edge_{k_percent_str}.npy"
source_file = f"/data/nidhi/data_augmentation/auto_encoder_study_dataset/cgg_{dataset}_graphs_hop_{hop}_max_node_{max_num_nodes}_source_max_edge_{k_percent_str}.npy"
target_file = f"/data/nidhi/data_augmentation/auto_encoder_study_dataset/cgg_{dataset}_graphs_hop_{hop}_max_node_{max_num_nodes}_target_max_edge_{k_percent_str}.npy"
node_att_file = f"/data/nidhi/data_augmentation/auto_encoder_study_dataset/cgg_{dataset}_graphs_hop_{hop}_max_node_{max_num_nodes}_att_max_edge_{k_percent_str}.npy"
cs_file = f"/data/nidhi/data_augmentation/auto_encoder_study_dataset/cgg_{dataset}_graphs_hop_{hop}_max_node_{max_num_nodes}_cs_{k_percent_str}.npy"
ct_file = f"/data/nidhi/data_augmentation/auto_encoder_study_dataset/cgg_{dataset}_graphs_hop_{hop}_max_node_{max_num_nodes}_ct_{k_percent_str}.npy"

complexity_difference = np.load(complexity_difference_file)
target = np.load(source_file)
source = np.load(target_file)
node_att = np.load(node_att_file)
ct = np.load(cs_file)
cs = np.load(ct_file)
print(source.shape, target.shape, cs.shape, ct.shape)

print(complexity_difference.shape, source.shape, target.shape, node_att.shape, cs.shape, ct.shape)
number_of_indices = cs.shape[1]
max_num_nodes = node_att.shape[1]
print(max_num_nodes)


class custom_dataset(Dataset):
  def __init__(self, source_graphs, target_graphs, ci_diffs, node_att, cs, ct):
    self.source_graphs = source_graphs
    self.target_graphs = target_graphs
    self.ci_diffs = ci_diffs
    self.node_att = node_att
    self.cs = cs
    self.ct = ct

  def __len__(self):
    return len(self.source_graphs)

  def __getitem__(self, index):
    return (
      self.source_graphs[index],
      self.target_graphs[index],
      self.ci_diffs[index],
      self.node_att[index],
      self.cs[index],
      self.ct[index],
    )
idx = list(range(len(source)))
import random

random.seed(123)
random.shuffle(idx)
train = idx[: int(node_att.shape[0] * 0.9)]
test = idx[int(node_att.shape[0] * 0.9): int(node_att.shape[0] * 0.95)]
val = idx[int(node_att.shape[0] * 0.95):]

print("length of train, test, val: ", len(train), len(test), len(val))

train_examples = custom_dataset(
    source[train],
    target[train],
    complexity_difference[train],
    node_att[train],
    cs[train],
    ct[train],
)
test_examples = custom_dataset(
    source[test],
    target[test],
    complexity_difference[test],
    node_att[test],
    cs[test],
    ct[test],
)

train_data_loader = DataLoader(train_examples, batch_size=4096, num_workers=1)
test_data_loader = DataLoader(test_examples, batch_size=4096, num_workers=1)
device = torch.device(args.device)

loss_fn_ci= nn.MSELoss()


model = CI_Predictor_w_MLP(max_num_nodes,number_of_indices)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train():
    epochs = args.epochs
    for e in range(epochs):
        losses = []

        model.train()
        for batch in tqdm(train_data_loader):
            source_graphs, target_graphs, ci_diffs, _node_att, cs, ct = batch

            source_graphs, target_graphs, ci_diffs, _node_att, cs, ct = (
                source_graphs.float(),
                target_graphs.float(),
                ci_diffs.float(),
                _node_att.float(),
                cs.float(),
                ct.float(),
            )
            graphs = torch.cat([source_graphs, target_graphs])
            C = torch.cat([cs,ct])

            graphs = graphs.to(device)
            C = C.to(device)
            C_predicted = model(graphs.reshape(graphs.shape[0],-1))
            loss = loss_fn_ci(C_predicted.flatten(), C.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())


        losses = np.array(losses)

        print(
            f"e ={e} \t loss ={losses.mean():.6f}"
        )
        sys.stdout.flush()

    save_model_loc = f"/data/nidhi/data_augmentation/ci_saved_model/{std_name}.pth"
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, save_model_loc)

    print(f"model saved to {save_model_loc}")
    sys.stdout.flush()
    return model

def test(model):
    model.eval()
    predictions = []
    ground_truth = []
    for batch in tqdm(test_data_loader):
        source_graphs, target_graphs, ci_diffs, _node_att, cs, ct = batch

        source_graphs, target_graphs, ci_diffs, _node_att, cs, ct = (
            source_graphs.float(),
            target_graphs.float(),
            ci_diffs.float(),
            _node_att.float(),
            cs.float(),
            ct.float(),
        )
        graphs = torch.cat([source_graphs, target_graphs])
        C = torch.cat([cs,ct])

        graphs = graphs.to(device)
        C = C.to(device)
        C_predicted = model(graphs.reshape(graphs.shape[0],-1)).detach().cpu()

        predictions.append(C_predicted)
        ground_truth.append(C.detach().cpu())


    P = torch.concat(predictions,dim = 0)
    GT = torch.concat(ground_truth,dim = 0)
    rows=[]
    rows = [["", "density", "number of edges", "number of nodes", "node connetivity", "avg clustering",
                       "closeness centrality", "number of local bridges", "transitivity", "edge connectivity", "number of cliques", "estrada index", "treewidth", "diameter"]]
    from tabulate import tabulate

    for p,gt in zip(P,GT):

        rows.append(["predicted"] + [np.round(i, 3) for i in p.tolist()])
        rows.append(["ground truth"] + [np.round(i, 3) for i in gt.tolist()])

    print(tabulate(rows, tablefmt="fancy_grid"))

    mse = nn.MSELoss()
    print("error:", mse(P.flatten(), GT.flatten()).item())
    sys.stdout.flush()

if __name__ == "__main__":
    model = train()
    test(model)