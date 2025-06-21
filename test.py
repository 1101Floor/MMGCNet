import time

import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.dataloader import TrajectoryDataset
from model.CVAE import CVAE
from model.sampler import Sampler
from utils.metrics import compute_ADE, compute_FDE, count_miss_samples
from utils.sddloader import SDD_Dataset

sys.path.append(os.getcwd())
from utils.torchutils import *
from utils.utils import prepare_seed, AverageMeter
import copy

parser = argparse.ArgumentParser()

# task setting
parser.add_argument('--obs_len', type=int, default=20)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--sdd_scale', type=float, default=50.0)
parser.add_argument('--dataset', type=str, default='Lng',help='Lng')  
parser.add_argument('--dataset_Mo', type=str, default='Mo',help='Mo') 
# model architecture
parser.add_argument('--pos_concat', type=bool, default=True)
parser.add_argument('--cross_motion_only', type=bool, default=False)

parser.add_argument('--tf_model_dim', type=int, default=64)
parser.add_argument('--tf_ff_dim', type=int, default=64)
parser.add_argument('--tf_nhead', type=int, default=8)
parser.add_argument('--tf_dropout', type=float, default=0.1)

parser.add_argument('--he_tf_layer', type=int, default=2)  # he = history encoder
parser.add_argument('--fe_tf_layer', type=int, default=2)  # fe = future encoder
parser.add_argument('--fd_tf_layer', type=int, default=2)  # fd = future decoder

parser.add_argument('--he_out_mlp_dim', default=None)
parser.add_argument('--fe_out_mlp_dim', default=None)
parser.add_argument('--fd_out_mlp_dim', default=None)

parser.add_argument('--num_tcn_layers', type=int, default=1)
parser.add_argument('--asconv_layer_num', type=int, default=3)

parser.add_argument('--pred_dim', type=int, default=2)

parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--nz', type=int, default=32)
parser.add_argument('--sample_k', type=int, default=20)

parser.add_argument('--max_train_agent', type=int, default=100)
parser.add_argument('--rand_rot_scene', type=bool, default=True)
parser.add_argument('--discrete_rot', type=bool, default=False)

# sampler architecture
parser.add_argument('--qnet_mlp', type=list, default=[512, 256])
parser.add_argument('--share_eps', type=bool, default=True)
parser.add_argument('--train_w_mean', type=bool, default=True)

# testing options
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--sample_num', type=int, default=50)

parser.add_argument('--sampler_epoch', type=int, default=100)
parser.add_argument('--vae_epoch', type=int, default=100)
args = parser.parse_args([])
def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def test(cvae, sampler, loader_test, traj_scale):
    ade_meter = AverageMeter()
    fde_meter = AverageMeter()
    raw_data_dict = {}
    step =0 
    # total_cnt = 0
    # miss_cnt = 0

    for cnt, batch in enumerate(loader_test):
        step+=1
        seq_name = 'data'
        frame_idx = int(batch.pop()[0])
        batch = [tensor[0].cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        obs_loss_mask, pred_loss_mask, V_obs, A_obs, V_tr, A_tr, A_TPCA_obs,\
        A_TPCA_tr, A_vs_obs, A_vs_tr = batch
        
        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame_idx))
        sys.stdout.flush()
        V_obs_tmp = V_obs.unsqueeze(0).permute(0, 3, 1, 2)
        with torch.no_grad():
            cvae.set_data(obs_traj_rel, pred_traj_gt_rel, obs_loss_mask, pred_loss_mask,\
                       V_obs_tmp, A_obs.squeeze(),A_TPCA_obs.squeeze(), A_vs_obs.squeeze())
            dec_motion, _, _, attn_weights = sampler.forward(cvae)  # [N sn T 2]  # testing function
        dec_motion = dec_motion * traj_scale
        
        traj_gt = pred_traj_gt.transpose(1, 2) * traj_scale  # [N 2 T] -> [N T 2]
        obs_traj_gt = obs_traj.transpose(1, 2) * traj_scale

        agent_traj = []
        agent_pre = []
        sample_motion = dec_motion.detach().cpu().numpy()  # [7 20 12 2]

        V_x = seq_to_nodes(obs_traj.unsqueeze(0).data.cpu().numpy().copy())

        V_obs = seq_to_nodes(obs_traj_rel.unsqueeze(0).data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.copy(),
                                         V_x[0,:,:].copy())#[8,3,2]
                                         
        V_y = seq_to_nodes(pred_traj_gt.unsqueeze(0).data.cpu().numpy().copy())
        V_tr = seq_to_nodes(pred_traj_gt_rel.unsqueeze(0).data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.copy(),
                                         V_x[-1,:,:].copy())#[12,3,2]
        
        num_v, simp, pred_len, fea_len = sample_motion.shape[0],sample_motion.shape[1],sample_motion.shape[2],sample_motion.shape[3]
        sample_motion_transfer = torch.from_numpy(sample_motion).permute(1,2,0,3)

        for i in range(sample_motion_transfer.shape[0]):  # traverse each person  list -> ped dimension
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(sample_motion_transfer[i, :, :, :].data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            agent_pre.append(V_pred_rel_to_abs)
        
        
        for i in range(np.array(agent_pre).shape[2]):
            agent_traj.append(np.array(agent_pre)[:,:,i,:])
        agent_traj = np.array(agent_traj)
        traj_gt = traj_gt.detach().cpu().numpy()

        # calculate ade and fde and get the min value for 20 samples
        ade = compute_ADE(agent_traj, traj_gt)
        ade_meter.update(ade, n=cvae.agent_num)

        fde = compute_FDE(agent_traj, traj_gt)
        fde_meter.update(fde, n=cvae.agent_num)
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(obs_traj_gt.cpu())
        raw_data_dict[step]['trgt'] = copy.deepcopy(traj_gt)
        raw_data_dict[step]['pred'] = copy.deepcopy(agent_traj)
        
    return ade_meter, fde_meter, raw_data_dict  # , miss_rate

obs_seq_len = args.obs_len

pred_seq_len = args.pred_len

data_set_Mo = './dataset/' + args.dataset_Mo + '/'
data_set = './dataset/' + args.dataset + '/'
prepare_seed(args.seed)

torch.set_default_dtype(torch.float32)
device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

traj_scale = 1.0
'''dset_test = TrajectoryDataset(
    data_dir =  data_set+'test/',
    data_dir_Mo = data_set_Mo+'test/',
    obs_len = obs_seq_len,
    pred_len = pred_seq_len,
    skip=1,norm_lap_matr=True)
torch.save(dset_test, "./dataset/dset_test.pt")'''
dset_test = torch.load("./dataset/dset_test.pt")

dset_test = DataLoader(
    dset_test,
    batch_size=1,
    shuffle=False,
    num_workers=0)
cvae = CVAE(args)
sampler = Sampler(args)

vae_dir = './checkpoints/' + args.dataset + '/vae/'
all_vae_models = os.listdir(vae_dir)
if len(all_vae_models) == 0:
    print('VAE model not found!')


default_vae_model = 'model_%04d.p' % args.vae_epoch
if default_vae_model not in all_vae_models:
    default_vae_model = all_vae_models[-1]
cp_path = os.path.join(vae_dir, default_vae_model)
print('loading model from checkpoint: %s' % cp_path)
model_cp = torch.load(cp_path, map_location='cpu')
cvae.load_state_dict(model_cp)

cvae.set_device(device)
cvae.eval()

# load sampler model
sampler_dir = './checkpoints/' + args.dataset + '/sampler/'
all_sampler_models = os.listdir(sampler_dir)
if len(all_sampler_models) == 0:
    print('sampler model not found!')

default_sampler_model = 'model_%04d.p' % args.sampler_epoch
if default_sampler_model not in all_sampler_models:
    default_sampler_model = all_sampler_models[-1]
# load sampler model
cp_path = os.path.join(sampler_dir, default_sampler_model)
model_cp = torch.load(cp_path, map_location='cpu')
sampler.load_state_dict(model_cp)
# torch.save(model_cp['model_dict'], cp_path)
print('loading model from checkpoint: %s' % cp_path)

sampler.set_device(device)
sampler.eval()

# run testing
ade_meter, fde_meter, raw_data_dict = test(cvae, sampler, dset_test, traj_scale)

print('-' * 20 + ' STATS ' + '-' * 20)
print('ADE: %.4f' % ade_meter.avg)
print('FDE: %.4f' % fde_meter.avg)