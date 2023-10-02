import mdtraj as md
from ase import Atoms
import nglview as nv
import networkx as nx
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGPooling, InnerProductDecoder
import torch_geometric.data as data
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
print("1/8 -Imports done")
# load the traj
traj = md.load('../../../xtc_files/5lk_eql2.xtc', top='../../../xtc_files/conf.gro')
print("2/8 - traj load done")
# creating the frame feature matrix
nframes = traj.xyz.shape[0]
print("3/8 - nframes done")
natoms=traj.topology.residue(0).n_atoms
print("4/8 - natoms  done")
nmols=traj.topology.n_residues
print("5/8 - nmols done")
frame_feature=[]
for i in tqdm(range(nframes)):
    com_pos=[] # position of Center of Mass of each molecule
    for res in range(nmols):
        com_pos.append(traj.xyz[i][res*natoms:(res+1)*natoms].mean(axis=0))
    frame_feature.append(com_pos)
print("6/8 - for loop done")
frame_feature=np.array(frame_feature)
print("7/8 - np array for frame feature done ")
print("8/8 - Last - printing shape of frame feature")
print(frame_feature.shape)