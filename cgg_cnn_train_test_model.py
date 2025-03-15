import sys
import os
os.system("pip install tabulate")
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
import networkx as nx
import pickle
from sklearn.metrics.pairwise import rbf_kernel
from utils import calculate_indices

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd

# wordnet,hops 2, max_num_nodes 80, max_edge_remove 10
# citeseer, hops 3 , max_num_nodes 50 , max_edge_remove 14
# arxiv, hops 2, max_num_nodes 80, max_edge_remove 25

parser = argparse.ArgumentParser('Interface for CGG')
parser.add_argument('--dataset', type=str, default='citeseer', help='dataset name - arxiv,wordnet, citeseer')
# dataset arguments
parser.add_argument('--hop', type=int, default=3, help="3,2")
parser.add_argument('--max_num_nodes', type=int, default=50, help="50,20, 60, 70")
parser.add_argument('--k_percent_str', type=str, default="0_5", help="0_5,0_6")
parser.add_argument('--device', type=str, default="cuda:0", help="gpu-device")
parser.add_argument('--lr', type=float, default=0.0001, help="0.0001")
parser.add_argument('--epochs', type=int, default=200, help="100,200")
parser.add_argument('--lam', type=float, default=0.0008, help="0.00001: to weight triplet loss")
parser.add_argument('--lam_ci', type=float, default=1, help="to weight ci loss")
parser.add_argument('--lam_f', type=float, default=1, help="to weight feature loss")
parser.add_argument('--select_ci', type=int, default=13, help="select indices")
parser.add_argument('--flip', type=str, default="no", help="randomly swap source with target examples for variation")
parser.add_argument('--source_zero', type=str, default="no", help="make source graph with 0 edges")
parser.add_argument('--mask', type=str, default="no", help="mask padded values while computing loss")
parser.add_argument('--log', type=int, default=1, help="1 to create log, 0 to skip")
parser.add_argument('--checkpoint',type = str, default="", help="checkpoint of pretrained model")
parser.add_argument('--model_type', type=str, default="mlp", help = "select model cnn/mlp")

args = parser.parse_args()
print(args)
print("With MLP in encoding complexity indices")

def get_ci_predictor(args, max_node_value, number_of_indices):
    print("With pre-trained ci predictor")
    if args.dataset == "citeseer":
        model_loc = f"/data/nidhi/data_augmentation/ci_saved_model/ci_pred_citeseer_h_3_m_{max_num_nodes}_k_0_5_lr_0_0001_e_1000.pth"
    elif args.dataset == "arxiv":
        model_loc = f"/data/nidhi/data_augmentation/ci_saved_model/ci_pred_arxiv_h_2_m_{max_num_nodes}_k_0_5_lr_0_001_e_1000.pth"
    # elif args.dataset == "wordnet":
    #     model_loc = f"/data/nidhi/data_augmentation/ci_saved_model/ci_pred_wordnet_h_2_m_{max_num_nodes}_k_0_5_lr_0_001_e_1000.pth"
    elif "wordnet" in args.dataset :
        model_loc = f"/data/nidhi/data_augmentation/ci_saved_model/ci_pred_wordnet_h_2_m_{max_num_nodes}_k_0_5_lr_0_001_e_1000.pth"
    elif "mutag" in args.dataset :
        model_loc = f"/data/nidhi/data_augmentation/ci_saved_model/ci_pred_mutag_h_0_m_{max_num_nodes}_k_0_5_lr_0_001_e_3000.pth"
    elif "molbace" in args.dataset :
        model_loc = f"/data/nidhi/data_augmentation/ci_saved_model/ci_pred_molbace_h_0_m_{max_num_nodes}_k_0_5_lr_0_001_e_3000.pth"
    else:
        model_loc = None

    ci_predictor = CI_Predictor_w_MLP(max_node_value, number_of_indices)
    checkpoint = torch.load(model_loc)
    ci_predictor.load_state_dict(checkpoint["state_dict"])

    ci_predictor.to(device)
    return ci_predictor


def get_std_name(args, time_stamp):
    ds = args.dataset
    hop = args.hop
    max_num_nodes = args.max_num_nodes
    k_percent_str = args.k_percent_str
    lr = str(args.lr).replace(".", "_")
    e = args.epochs
    lam = str(args.lam).replace(".", "_")
    lam_ci = str(args.lam_ci).replace(".", "_")
    lam_f = str(args.lam_f).replace(".", "_")
    select_ci = int(args.select_ci)
    flip = args.flip
    source_zero = args.source_zero
    mask = args.mask

    name = f'cgg_model_type_{args.model_type}_{ds}_h_{hop}_m_{max_num_nodes}_k_{k_percent_str}_lr_{lr}_e_{e}_lam_{lam}_ci_{lam_ci}_f_{lam_f}_ci_{select_ci}_{flip}_sz_{source_zero}_mask_{mask}'
    name = name + f'_{time_stamp}'
    name = name.replace("-", "_")
    return name


time_stamp = str(datetime.datetime.now()).replace("-", "_").replace(" ", "_")
time_stamp = time_stamp.replace(":", "_").replace(".", "_")
std_name = get_std_name(args, time_stamp)

import sys
if args.log == 0:
    pass
else:
    sys.stdout = sys.stderr = open(f"/data/nidhi/data_augmentation/cgg_logs/{std_name}.log", "w")
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
nb_target = np.array([len(max(nx.connected_components(nx.from_numpy_array(t)), key=len)) for t in target])
nb_source = np.array([len(max(nx.connected_components(nx.from_numpy_array(s)), key=len)) for s in source])

if args.flip == "yes":
    flip_idx = list(range(len(source)))

    random.seed(123)
    random.shuffle(flip_idx)

    nb_flips = 0
    for i in flip_idx:
        if i % 2 == 0:
            nb_flips = nb_flips + 1
            # print("before:", source[i].sum(), target[i].sum())
            tmp = copy.deepcopy(source[i])
            source[i] = copy.deepcopy(target[i])
            target[i] = tmp
            # print("after:", source[i].sum(), target[i].sum())
            tmp = copy.deepcopy(cs[i])
            cs[i] = copy.deepcopy(ct[i])
            ct[i] = tmp

    print(f"number of examples flipped = {nb_flips}")

if args.source_zero == "yes":
    for i in range(len(source)):
        source[i] = 0
        cs[i] = 0

print(complexity_difference.shape, source.shape, target.shape, node_att.shape, cs.shape, ct.shape)
number_of_indices = cs.shape[1]
max_num_nodes = node_att.shape[1]
print(max_num_nodes)


class Ci_diff_weights(nn.Module):
    def __init__(self):
        super(Ci_diff_weights, self).__init__()

        # ci_difference encoding through MLP for weighted loss value (for source - prediction pair)
        self.weighted_loss_for_s_linear_layer = nn.Sequential(
            nn.Linear(26, 13),
            nn.ReLU(),
            nn.Linear(13, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

        self.weighted_loss_for_s_t_linear_layer = nn.Sequential(
            nn.Linear(26, 13),
            nn.ReLU(),
            nn.Linear(13, 2),
            nn.BatchNorm1d(2),
            nn.Sigmoid(),
        )

        self.cs_linear = nn.Sequential(nn.Linear(13, 13), nn.ReLU())

        self.ct_linear = nn.Sequential(nn.Linear(13, 13), nn.ReLU())

    def forward(self, cs, ct):
        cs = self.cs_linear(cs)
        ct = self.ct_linear(ct)

        cs_ct = torch.concat([cs, ct], dim=1)
        # weights_for_triplet = self.weighted_loss_for_s_t_linear_layer(cs_ct)
        weight_for_source = self.weighted_loss_for_s_linear_layer(cs_ct)

        return weight_for_source  # weights_for_triplet


class CGG_CNN_w_Features(nn.Module):
    def __init__(self, ci_predictor=None):
        super(CGG_CNN_w_Features, self).__init__()
        ci_kernel_size = 2
        kernel_default = 5
        kernel_wsg = 1
        kernel_size = kernel_default
        feature_dim = 50

        # feature encoding through Conv2d
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
        )

        self.feature_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
        )

        # ci_difference, c_source, c_target encoding through Conv1d

        self.cs_linear = nn.Sequential(nn.Linear(13, 13), nn.ReLU())

        self.ct_linear = nn.Sequential(nn.Linear(13, 13), nn.ReLU())

        self.ci_diff_linear = nn.Sequential(nn.Linear(13, 13), nn.ReLU())

        self.cs_ct_ci_diff = nn.Sequential(
            nn.Linear(39, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # self.cs_ct_ci_diff = nn.Sequential(
        #     nn.Conv1d(1, 4, kernel_size=8, stride=1),
        #     nn.ReLU(),
        #     nn.Conv1d(4, 8, kernel_size=4, stride=4),
        #     nn.ReLU(),
        # )

        # self.ci_difference_cnn = nn.Sequential(
        #     nn.Conv1d(1, 32, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        # )

        # Adj matrix encoding through Conv2d
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
        )
        self.edge_source_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=1),
            nn.Sigmoid(),
        )

        self.edge_target_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=1),
            nn.Sigmoid(),
        )

        if ci_predictor is None:

            self.ci_predictor = nn.Sequential(
                nn.Linear(max_num_nodes * max_num_nodes, max_num_nodes),
                nn.ReLU(),
                nn.Linear(max_num_nodes, ct.shape[-1]),
            )
        else:
            self.ci_predictor = ci_predictor
            for param in self.ci_predictor.parameters():
                param.requires_grad = False

    def forward(self, adj_matrix, feature_matrix, ci_diff, cs, ct):
        feat_encoder_output = self.feature_encoder(feature_matrix)
        z = self.edge_encoder(adj_matrix)

        z = z + feat_encoder_output

        cs = self.cs_linear(cs)
        ct = self.ct_linear(ct)
        ci_diff = self.ci_diff_linear(ci_diff)
        cs_ct_ci_diff = torch.concat([cs, ct, ci_diff], dim=1)

        cs_ct_ci_diff_out = self.cs_ct_ci_diff(cs_ct_ci_diff)  # .unsqueeze(1) CNN

        cs_ct_ci_diff_out = (
            cs_ct_ci_diff_out.reshape(cs.shape[0], -1).unsqueeze(2).unsqueeze(3)
        )

        # ci_diff_cnn_out = self.ci_difference_cnn(ci_diff.unsqueeze(1)).unsqueeze(2)

        z_and_cs_ct_ci_diff = torch.einsum("abik, abnj -> abik", z, cs_ct_ci_diff_out)

        adj_matrix_source_pred = self.edge_source_decoder(z)
        adj_matrix_target_pred = self.edge_target_decoder(z_and_cs_ct_ci_diff)
        feature_pred = self.feature_decoder(z)

        pred_ct = self.ci_predictor(adj_matrix_target_pred.reshape(ct.shape[0], -1))
        # pred_ct = CI_Predictor_w_MLP(adj_matrix_target_pred.reshape(ct.shape[0], -1))

        return (adj_matrix_source_pred, adj_matrix_target_pred, feature_pred, pred_ct)

class CGG_MLP_w_Features(nn.Module):
    def __init__(self, ci_predictor=None):
        super(CGG_MLP_w_Features, self).__init__()
        ci_kernel_size = 2
        kernel_default = 5
        kernel_wsg = 1
        kernel_size = kernel_default
        feature_dim = 50
        mn = max_num_nodes
        # feature encoding through Conv2d
        self.feature_encoder = nn.Sequential(
            nn.Linear(mn*mn, mn*mn),
            nn.ReLU(),
            nn.Linear(mn*mn, mn*mn),
            nn.ReLU(),
        )

        self.feature_decoder = nn.Sequential(
            nn.Linear(mn*mn, mn*mn),
            nn.ReLU(),
            nn.Linear(mn*mn, mn*mn),
            nn.ReLU(),
        )

        # ci_difference, c_source, c_target encoding through Conv1d

        self.cs_linear = nn.Sequential(nn.Linear(13, 13), nn.ReLU())

        self.ct_linear = nn.Sequential(nn.Linear(13, 13), nn.ReLU())

        self.ci_diff_linear = nn.Sequential(nn.Linear(13, 13), nn.ReLU())

        self.cs_ct_ci_diff = nn.Sequential(
            nn.Linear(39, 4096),
            nn.ReLU(),
            nn.Linear(4096, mn*mn),
            nn.ReLU()
        )

        # Adj matrix encoding through Conv2d
        self.edge_encoder = nn.Sequential(
            nn.Linear(mn*mn, mn*mn),
            nn.ReLU(),
            nn.Linear(mn*mn, mn*mn),
            nn.ReLU(),
        )
        self.edge_source_decoder = nn.Sequential(
            nn.Linear(mn*mn, mn*mn),
            nn.ReLU(),
            nn.Linear(mn*mn, mn*mn),
            nn.Sigmoid(),
        )

        self.edge_target_decoder = nn.Sequential(
            nn.Linear(mn*mn, mn*mn),
            nn.ReLU(),
            nn.Linear(mn*mn, mn*mn),
            nn.Sigmoid(),
        )

        if ci_predictor is None:

            self.ci_predictor = nn.Sequential(
                nn.Linear(max_num_nodes * max_num_nodes, max_num_nodes),
                nn.ReLU(),
                nn.Linear(max_num_nodes, ct.shape[-1]),
            )
        else:
            self.ci_predictor = ci_predictor
            for param in self.ci_predictor.parameters():
                param.requires_grad = False

    def forward(self, adj_matrix, feature_matrix, ci_diff, cs, ct):
        adj_matrix = adj_matrix.reshape(adj_matrix.shape[0],-1)
        feature_matrix = feature_matrix.reshape(feature_matrix.shape[0], -1)
        feat_encoder_output = self.feature_encoder(feature_matrix)

        z = self.edge_encoder(adj_matrix)

        z = z + feat_encoder_output

        cs = self.cs_linear(cs)
        ct = self.ct_linear(ct)
        ci_diff = self.ci_diff_linear(ci_diff)
        cs_ct_ci_diff = torch.concat([cs, ct, ci_diff], dim=1)

        cs_ct_ci_diff_out = self.cs_ct_ci_diff(cs_ct_ci_diff)  # .unsqueeze(1) CNN

        # cs_ct_ci_diff_out = (
        #     cs_ct_ci_diff_out.reshape(cs.shape[0], -1).unsqueeze(2).unsqueeze(3)
        # )

        # ci_diff_cnn_out = self.ci_difference_cnn(ci_diff.unsqueeze(1)).unsqueeze(2)

        z_and_cs_ct_ci_diff = torch.mul( z, cs_ct_ci_diff_out)

        adj_matrix_source_pred = self.edge_source_decoder(z)
        adj_matrix_target_pred = self.edge_target_decoder(z_and_cs_ct_ci_diff)
        feature_pred = self.feature_decoder(z)

        pred_ct = self.ci_predictor(adj_matrix_target_pred.reshape(ct.shape[0], -1))
        # pred_ct = CI_Predictor_w_MLP(adj_matrix_target_pred.reshape(ct.shape[0], -1))

        return (adj_matrix_source_pred, adj_matrix_target_pred, feature_pred, pred_ct)

class custom_dataset(Dataset):
    def __init__(self, source_graphs, target_graphs, ci_diffs, node_att, cs, ct, nb_target, nb_source):
        self.source_graphs = source_graphs
        self.target_graphs = target_graphs
        self.ci_diffs = ci_diffs
        self.node_att = node_att
        self.cs = cs
        self.ct = ct
        self.nb_target = nb_target
        self.nb_source = nb_source

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
            self.nb_target[index],
            self.nb_source[index]
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
    nb_target[train],
    nb_source[train]

)
test_examples = custom_dataset(
    source[test],
    target[test],
    complexity_difference[test],
    node_att[test],
    cs[test],
    ct[test],
    nb_target[test],
    nb_source[test]
)

train_data_loader = DataLoader(train_examples, batch_size=1024, num_workers=1)
test_data_loader = DataLoader(test_examples, batch_size=1024, num_workers=1)
device = torch.device(args.device)


def custom_loss_for_s_fn(
        adj_matrix_target_pred, target_graphs, source_graphs, weight_for_source
):
    pos_distance = torch.norm(adj_matrix_target_pred - target_graphs, dim=1).unsqueeze(
        -1
    )
    neg_distance = torch.norm(adj_matrix_target_pred - source_graphs, dim=1).unsqueeze(
        -1
    )

    zeroes = (
        torch.zeros(pos_distance.shape[0], pos_distance.shape[1]).float().to(device)
    )
    margins = (
        torch.ones(pos_distance.shape[0], pos_distance.shape[1]).float().to(device)
    )
    # print(f"weight_for_source:  {weight_for_source}, weight_for_source:  {weight_for_source}")

    loss_tensor = torch.max(
        torch.concat(
            [pos_distance - weight_for_source * neg_distance + margins, zeroes], dim=0
        ),
        dim=0,
    ).values
    return loss_tensor.mean()


def custom_loss_for_s_t_fn(
        adj_matrix_target_pred,
        target_graphs,
        source_graphs,
        weight_for_source,
        weight_for_target,
):
    pos_distance = torch.norm(adj_matrix_target_pred - target_graphs, dim=1).unsqueeze(
        -1
    )
    neg_distance = torch.norm(adj_matrix_target_pred - source_graphs, dim=1).unsqueeze(
        -1
    )

    zeroes = (
        torch.zeros(pos_distance.shape[0], pos_distance.shape[1]).float().to(device)
    )
    margins = (
        torch.ones(pos_distance.shape[0], pos_distance.shape[1]).float().to(device)
    )

    loss_tensor = torch.max(
        torch.concat(
            [
                weight_for_target * pos_distance
                - weight_for_source * neg_distance
                + margins,
                zeroes,
            ],
            dim=0,
        ),
        dim=0,
    ).values
    return loss_tensor.mean()


ci_predictor = get_ci_predictor(args, max_num_nodes, number_of_indices)
if args.model_type == "cnn":
    print("Using CNN")
    model = CGG_CNN_w_Features(ci_predictor)
elif args.model_type == "mlp":
    print("Using MLP")
    model = CGG_MLP_w_Features(ci_predictor)
else:
    model = None

if args.checkpoint != "":
    print("loading checkpoint")
    checkpoint = torch.load(f"/data/nidhi/data_augmentation/cgg_saved_model/{args.checkpoint}.pth")
    model.load_state_dict(checkpoint["state_dict"])

print(f"using model class= {type(model)}")
model2 = Ci_diff_weights()
source_loss_fn = nn.BCELoss()
target_loss_fn = nn.BCELoss()
loss_fn_feature = nn.MSELoss()
diff_source_predicted_loss_fn = nn.BCELoss()
pred_ct_loss_fn = nn.MSELoss()
margin_loss_fn = nn.TripletMarginLoss(p=2, reduction="none")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
model = model.to(device)
model2 = model2.to(device)

relation_value = []

epochs = args.epochs
for e in range(epochs):
    source_losses = []
    target_losses = []
    source_pred_diff_losses = []
    triple_losses = []
    feature_losses = []
    pred_ct_losses = []
    losses = []
    model2.train()
    model.train()
    for batch in tqdm(train_data_loader):
        source_graphs, target_graphs, ci_diffs, _node_att, cs, ct, nb_t, nb_s = batch
        source_graphs, target_graphs, ci_diffs, _node_att, cs, ct = (
            source_graphs.float(),
            target_graphs.float(),
            ci_diffs.float(),
            _node_att.float(),
            cs.float(),
            ct.float(),
        )

        source_graphs = source_graphs.unsqueeze(1)
        _node_att = _node_att.unsqueeze(1)

        source_graphs = source_graphs.to(device)
        _node_att = _node_att.to(device)
        ci_diffs = ci_diffs.to(device)
        target_graphs = target_graphs.to(device)
        cs = cs.to(device)
        ct = ct.to(device)

        (adj_matrix_source_pred, adj_matrix_target_pred, feature_pred, pred_ct) = model(
            source_graphs, _node_att, ci_diffs, cs, ct
        )
        weight_for_source = model2(cs, ct)

        """
        #weights_for_triplet = model2(cs, ct)

        weight_for_source, weight_for_target = (
            weights_for_triplet[:, 0],
            weights_for_triplet[:, 1],
        )
        weight_for_source = weight_for_source.unsqueeze(-1)
        weight_for_target = weight_for_target.unsqueeze(-1)
        """
        source_loss = source_loss_fn(
            adj_matrix_source_pred.flatten(), source_graphs.flatten()
        )
            
        if args.mask == "yes":
            mask = torch.ones(adj_matrix_target_pred.shape).to(device)
            for idx, n in enumerate(nb_t.numpy().tolist()):
                mask[idx, :, n:, :] = 0
                mask[idx, :, :, n:] = 0

            adj_matrix_target_pred = adj_matrix_target_pred * mask

        target_loss = target_loss_fn(
            adj_matrix_target_pred.flatten(), target_graphs.flatten()
        )

        # distance_from_source = torch.norm(adj_matrix_target_pred.flatten() -  source_graphs.flatten(),2)

        triple_loss = custom_loss_for_s_fn(
            adj_matrix_target_pred.reshape(-1, max_num_nodes * max_num_nodes),
            target_graphs.reshape(-1, max_num_nodes * max_num_nodes),
            source_graphs.reshape(-1, max_num_nodes * max_num_nodes),
            weight_for_source,
        )

        """
        triple_loss = custom_loss_for_s_t_fn(
            adj_matrix_target_pred.reshape(-1, max_num_nodes * max_num_nodes),
            target_graphs.reshape(-1, max_num_nodes * max_num_nodes),
            source_graphs.reshape(-1, max_num_nodes * max_num_nodes),
            weight_for_source,
            weight_for_target,
        )
        """
        feature_loss = loss_fn_feature(feature_pred.reshape(-1), _node_att.reshape(-1))

        pred_ct_loss = pred_ct_loss_fn(pred_ct.reshape(-1), ct.reshape(-1))

        loss = (
                source_loss
                + target_loss
                + args.lam * triple_loss
                + args.lam_ci * pred_ct_loss
                + args.lam_f * feature_loss
        )  # 0.0008 *
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        source_losses.append(source_loss.item())
        target_losses.append(target_loss.item())
        triple_losses.append(triple_loss.mean().item())
        feature_losses.append(feature_loss.item())
        pred_ct_losses.append(pred_ct_loss.item())

    source_losses = np.array(source_losses)
    target_losses = np.array(target_losses)
    losses = np.array(losses)
    triple_losses = np.array(triple_losses)
    feature_losses = np.array(feature_losses)
    pred_ct_losses = np.array(pred_ct_losses)
    print(
        f"e ={e} \t loss ={losses.mean():.6f} \t source loss ={source_losses.mean():.6f} \t target loss ={target_losses.mean():.6f} \t triple loss = {triple_losses.mean():.6f} \t feature_losses = {feature_losses.mean():.6f} \t pred_ct_loss = {pred_ct_losses.mean():.6f}"
    )
    sys.stdout.flush()

if args.checkpoint == "":

    save_model_loc = f"/data/nidhi/data_augmentation/cgg_saved_model/{std_name}.pth"
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, save_model_loc)

    print(f"model saved to {save_model_loc}")
    sys.stdout.flush()


predicted_graphs = []
target_graphs_list = []
source_graphs_list = []

for batch in tqdm(test_data_loader):
    source_graphs, target_graphs, ci_diffs, _node_att, cs, ct,_,_ = batch
    source_graphs, target_graphs, ci_diffs, _node_att, cs, ct = (
        source_graphs.float(),
        target_graphs.float(),
        ci_diffs.float(),
        _node_att.float(),
        cs.float(),
        ct.float(),
    )

    source_graphs = source_graphs.unsqueeze(1)
    _node_att = _node_att.unsqueeze(1)

    source_graphs = source_graphs.to(device)
    _node_att = _node_att.to(device)
    ci_diffs = ci_diffs.to(device)
    target_graphs = target_graphs.to(device)
    cs = cs.to(device)


    

    ct = ct.to(device)

    (adj_matrix_source_pred, adj_matrix_target_pred, feature_pred, pred_ct) = model(
        source_graphs, _node_att, ci_diffs, cs, ct
    )

    predicted_graphs.append(
        adj_matrix_target_pred.reshape(-1, max_num_nodes, max_num_nodes).detach().cpu()
    )
    target_graphs_list.append(
        target_graphs.reshape(-1, max_num_nodes, max_num_nodes).detach().cpu()
    )
    source_graphs_list.append(
        source_graphs.reshape(-1, max_num_nodes, max_num_nodes).detach().cpu()
    )


predicted_graphs = torch.concat(predicted_graphs)
target_graphs_list = torch.concat(target_graphs_list)
source_graphs_list = torch.concat(source_graphs_list)

predicted_graph = [
    a.detach().cpu().numpy().reshape(max_num_nodes, max_num_nodes).round()
    for a in predicted_graphs
]
predicted_graph = [nx.from_numpy_array(a) for a in predicted_graph]


target_graphs = [
    a.detach().cpu().numpy().reshape(max_num_nodes, max_num_nodes)
    for a in target_graphs_list
]
target_graphs = [nx.from_numpy_array(a) for a in target_graphs]

source_graphs = [
    a.detach().cpu().numpy().reshape(max_num_nodes, max_num_nodes)
    for a in source_graphs_list
]
source_graphs = [nx.from_numpy_array(a) for a in source_graphs]

def get_largest_connected_component(graph):
    largest_cc = max(nx.connected_components(graph), key=len)
    G_cc = graph.subgraph(largest_cc)
    return G_cc

predicted_graph = [get_largest_connected_component(p) for p in predicted_graph]
target_graph = [get_largest_connected_component(gt) for gt in target_graphs]
source_graph = [get_largest_connected_component(p) for p in source_graphs]

pickle.dump({
    "predicted_graph":predicted_graph,
    "target_graph":target_graph,
    "source_graph":source_graph
},open(f"/data/nidhi/data_augmentation/cgg_graphs/{std_name}.pkl","wb"))

list_of_p_indices = []
list_of_t_indices = []

for p, g in tqdm(zip(predicted_graph, target_graph)):
    P = calculate_indices(p)
    G = calculate_indices(g)
    list_of_p_indices.append(P)
    list_of_t_indices.append(G)

P_numpy = np.array(list_of_p_indices)
G_numpy = np.array(list_of_t_indices)

diff_P_G = np.abs(P_numpy - G_numpy)
performance = diff_P_G.mean(axis=0)

performance = performance.reshape(1, -1)

df_performance = pd.DataFrame(
    performance,
    columns=[
        "Density",
        "Edges",
        "Nodes",
        "Node_Connectivity",
        "Avg_Clustering",
        "Closeness_Centrality",
        "Number_of_local_bridge",
        "transitivity",
        "edge_connectivity",
        "len_find_cliques",
        "estrada_index",
        "treewidth_min_degree",
        "diameter",
    ],
).T
df_performance = df_performance.rename(columns={0: "Difference_Avg"})
df_performance["Target_Avg"] = G_numpy.mean(axis=0)
df_performance["Pred_Avg"] = P_numpy.mean(axis=0)
print(df_performance.index)
df_performance = df_performance.reset_index()

# df_performance.loc[14] = ["average"] + df_performance[
#    ["Difference_Avg", "Target_Avg", "Pred_Avg"]
# ].mean(axis=0).tolist()
df_performance = df_performance[
    df_performance["index"].isin(
        [
            "Density",
            "Edges",
            "Nodes",
            "Node_Connectivity",
            "Avg_Clustering",
            "Closeness_Centrality",
            "Number_of_local_bridge",
            "transitivity",
            "edge_connectivity",
            "len_find_cliques",
            "treewidth_min_degree",
            "diameter",
            "average",
        ]
    )
].round(2)

df_performance.loc[len(df_performance) + 1] = ["average"] + df_performance[
    ["Difference_Avg", "Target_Avg", "Pred_Avg"]
].mean(axis=0).tolist()

from tabulate import tabulate

print(tabulate(df_performance, headers = 'keys', tablefmt = 'psql'))
sys.stdout.flush()

P = [[calculate_indices(g)] for g in predicted_graph]
T = [[calculate_indices(g)] for g in target_graph]
P = np.array(P).reshape(-1, 13)
T = np.array(T).reshape(-1, 13)
print(P.shape, T.shape)
sys.stdout.flush()
cols = [
    "Density",
    "Edges",
    "Nodes",
    "Node_Connectivity",
    "Avg_Clustering",
    "Closeness_Centrality",
    "Number_of_local_bridge",
    "transitivity",
    "edge_connectivity",
    "len_find_cliques",
    "estrada_index",
    "treewidth_min_degree",
    "diameter",
]
distances = []
for i, c in zip(range(13), cols):
    p = P[:, i].reshape(-1, 1)
    t = T[:, i].reshape(-1, 1)
    rbf_P = rbf_kernel(p)
    rbf_T = rbf_kernel(t)
    rbf_PT = rbf_kernel(p, t)
    distance = rbf_P.mean() + rbf_T.mean() - 2 * rbf_PT.mean()
    distances.append([c, distance])

pd.DataFrame(distances, columns=["properties", "mmd-rbf"]).round(4)

rbf_P = rbf_kernel(P)
rbf_T = rbf_kernel(T)
rbf_PT = rbf_kernel(P, T)
distance = rbf_P.mean() + rbf_T.mean() - 2 * rbf_PT.mean()
print(f"overall MMD = {distance}")
sys.stdout.flush()


