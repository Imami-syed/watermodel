#initial imports
import mdtraj as md
from ase import Atoms
from nglview import show_ase
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from typing import Optional, List, Tuple, Dict, Any
import nglview as nv
import sys
import os
import time

# torch imports
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import Module
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.transforms import pad

# torch geometric imports
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, BatchNorm, GATConv, Linear
from torch_geometric.nn import GCNConv, SAGPooling, InnerProductDecoder
import torch_geometric.data as data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import TransformerConv, GATConv, TopKPooling, BatchNorm


EPS = 1e-15
MAX_LOGSTD = 10

# function to calculate the adjacency matrix from edge list
def convert_to_adj(edge_index,num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1
    return adj

def convert_to_edge_index(adj):
    return adj.nonzero().t()



# Creating the model 

class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def forward(self, z: Tensor, edge_index: Tensor,
                sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True)[0] + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index.long(), sigmoid=True)[0] +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
            and :math:`\log\sigma^2`.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        """"""
        self.__mu__, self.__logstd__, self.edge_index = self.encoder(
            *args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z, self.edge_index

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

# Variational Graph Autoencoder

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels,batch_size,n_atoms):
        
        self.embedding_size1 = 15
        self.embedding_size2 = 9
        self.linear_size1 = 100
        self.linear_size2 = 4
        
        self.batch_size = batch_size
        self.n_atoms = n_atoms
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        super().__init__()
        self.conv1 = GATConv(self.in_channels,self.embedding_size1,heads=3)
        self.head_transform1 = Linear(self.embedding_size1*3, self.embedding_size1)
        self.bn1 = BatchNorm(self.embedding_size1)
        
        self.conv2 = GCNConv(self.embedding_size1,self.embedding_size2)
        self.bn2 = BatchNorm(self.embedding_size2)
        
        self.linear1 = Linear(self.embedding_size2, self.linear_size1)
        self.linear2 = Linear(self.linear_size1,self.linear_size2)
        
        self.transform = Linear(self.linear_size2*self.n_atoms,self.out_channels)
        
        self.mu = Linear(self.out_channels, self.out_channels)
        self.logstd = Linear(self.out_channels, self.out_channels)

    def forward(self, x, edge_index):
        self.batch_size = x.shape[0]//self.n_atoms
        x = self.conv1(x, edge_index)
        x = self.head_transform1(x)
        x = self.bn1(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        
        x = self.linear1(x)
        x = F.leaky_relu(x)

        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = x.reshape(self.batch_size,self.n_atoms,-1)
        x = x.reshape(self.batch_size,-1)
        
        x = self.transform(x)
        x = F.leaky_relu(x)
        
        
        x,y,z = self.mu(x), self.logstd(x), edge_index
        return x,y,z

# Variational Graph Decoder
class VariationalGCNDecoder(torch.nn.Module):
    def __init__(self,in_channels,out_channels,batch_size,n_atoms):
        self.embedding_size1 = 9
        self.embedding_size2 = 3
        self.embedding_size3 = 3
        self.linear_size1 = 512
        self.linear_size2 = 128
        self.batch_size = batch_size
        self.n_atoms = n_atoms
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()
        self.inv_transform = Linear(self.in_channels,self.n_atoms) 
        
        self.conv1 = GCNConv(1, self.embedding_size1)
        self.bn1 = BatchNorm(self.embedding_size1)

        self.conv2 = GCNConv(self.embedding_size1,self.embedding_size2)
        self.bn2 = BatchNorm(self.embedding_size2)

        self.conv3 = GCNConv(self.embedding_size2,self.embedding_size3)

        self.linear1 = Linear(self.embedding_size3, self.linear_size1)
        self.linear2 = Linear(self.linear_size1, self.linear_size2)
        self.linear3 = Linear(self.linear_size2, self.out_channels)

    def forward(self, x, edge_index, sigmoid=True):
        self.batch_size = x.shape[0]//self.n_atoms

        x = self.inv_transform(x)
        x = F.leaky_relu(x)

        x = x.reshape(x.shape[0]*x.shape[1],1)
        
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)        
        x = self.conv3(x,edge_index)
        
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.linear3(x)
        x = F.leaky_relu(x)        
        return x, edge_index
    

# early stopping class for the model
# this class is used to stop the training of the model 
# if the validation loss does not decrease for a certain number of epochs

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False