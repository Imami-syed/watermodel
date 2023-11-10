from stage1 import *
from mpl_toolkits import mplot3d
import os
from mpl_toolkits.mplot3d import Axes3D
# from torch.utils.tensorboard import SummaryWriter
model1="model1.pt"
stage1=torch.load("./models/"+model1)
n_neigh=20
#  n_neighbours represent 15  
def condenseframe(frame):
    mol_pos=[]
    edge_list=np.array([[ 0,  1,  0,  2],
                        [ 1,  0,  2,  0]])
    for mol_no,res in enumerate(frame.top.residues):
        pos=[]
        for atom in res.atoms:
            pos.append(frame.xyz[0][atom.index])
        pos=np.array(pos)
        # print(pos)
        avg_pos=np.mean(pos,axis=0)
        pos=pos-pos[0]
        # print(avg_pos)
        atomic_nums = np.array([[atom.element.atomic_number for atom in res.atoms]]).T
        # print(atomic_nums)
        vdwr = np.array([[atom.element.radius for atom in res.atoms]]).T
        # print(vdwr)
        node_features = np.concatenate((pos,vdwr,atomic_nums),axis=1)
        graph = data.Data(x=torch.from_numpy(node_features),edge_index=torch.from_numpy(edge_list))#.to("cuda")
        encoded = stage1.encode(graph.x,graph.edge_index)
        model_out = np.mean(encoded[0].detach().cpu().numpy(),axis=0)
        feature = np.concatenate((model_out,avg_pos))
        mol_pos.append(feature)
    mol_pos = np.array(mol_pos)
    return mol_pos
def condenseAllFrames(frames):
    """
    Condenses all frames in a trajectory
    """
    condensed_frames = []
    for frame in tqdm(frames):
        condensed_frames.append(condenseframe(frame))
    return np.array(condensed_frames)

def getNClosest(frame,mol_id,n):
    """ Returns the n closest molecules to the given molecule. """
    frame = frame[:,-3:]
    coord = frame[mol_id]
    dists = np.linalg.norm(frame-coord,axis=1)
    return np.argsort(dists)[1:n+1]
def get_graph(frame,mol_id,n_neigh,str_type):
    neigs = getNClosest(frame,mol_id,n_neigh)

    to_list = []
    from_list = []
    for mols_id in range(1,1+len(neigs)):
        to_list.append(mols_id)
        from_list.append(0)
        
        to_list.append(0)
        from_list.append(mols_id)
    
    
    edge_list = np.array([to_list,from_list])
    features = np.concatenate((np.array([frame[mol_id]]),np.array(frame[neigs])),axis=0)
    
    if(str_type == "cry"):
        graph = data.Data(x=torch.from_numpy(features),edge_index=torch.from_numpy(edge_list),y=torch.tensor([1]))
    elif (str_type == "melt"):
        graph = data.Data(x=torch.from_numpy(features),edge_index=torch.from_numpy(edge_list),y=torch.tensor([0]))
        
    return graph

def get_graphs(frames,str_type):
    graphs = []
    for frame in tqdm(frames):
        for mol_id in range(len(frame)):
#             frame_recon = changeFrame(frame,mol_id)
            graphs.append(get_graph(frame,mol_id,n_neigh,str_type))
    return graphs
def pad(graphs):
    max_nodes = 0
    for graph in graphs:
        max_nodes = max(max_nodes,graph.x.shape[0])
    
    padded =[]
    for graph in graphs:
        num_features = graph.num_features
        x = graph.x
        pad = torch.tensor([[0]*num_features] * (max_nodes-x.shape[0]))
        graph.x = torch.concatenate((x,pad))
    
    return graphs

def condenseFrame_same_res_id(f):
    """takes mdtraj frame object as input works even if all molecules have same residue ids
    (divides based on number of atoms in each molecule) """
    n_atoms = 4
    xyz = f.xyz[0]
    r = (np.random.rand(xyz.shape[0],xyz.shape[1]) - 0.5)*0.5
    xyz = xyz + r
    n_parts = xyz.shape[0]//n_atoms
    mols_pos = np.array(np.array_split(xyz,n_parts))

    edge_list = np.array([[ 0,  1,  0,  2],
                          [ 1,  0,  2,  0]])

    atomic_nums =   [[8],[1],[1],[0]]
    vdwr = [[0.152 ],[0.12 ],[0.12 ],[0 ]]

    condensed = []
    for mol_pos in mols_pos:
        avg_pos = np.mean(mol_pos,axis=0)
        recentered = mol_pos - mol_pos[0]
        node_features = np.concatenate((recentered,vdwr,atomic_nums),axis=1)
        graph = data.Data(x=torch.from_numpy(node_features),edge_index=torch.from_numpy(edge_list)).to("cuda")

        encoded = stage1.encode(graph.x,graph.edge_index)

        model_out = np.mean(encoded[0].detach().cpu().numpy(),axis=0)
        feature = np.concatenate((model_out,avg_pos))

        condensed.append(feature)
    condensed = np.array(condensed)
    return condensed
def condenseAllFrames_same_res_id(frame,n_frames):
    """
    Condenses all frames in a trajectory
    """
    condensed_frames = []
    for frame_id in tqdm(range(n_frames)):
        cf = condenseFrame_same_res_id(frame)
        
        condensed_frames.append(cf)
        
    return np.array(condensed_frames)

def convert_to_adj(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    return adj

def convert_to_edge_index(adj):
    edge_index = adj.nonzero().t()
    return edge_index



# Stage 2 Model 
from typing import Optional, Tuple 

import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling

EPS = 1e-15
MAX_LOGSTD = 10

from typing import Optional, Tuple 

import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling

EPS = 1e-15
MAX_LOGSTD = 10


class InnerProductDecoder_S2(torch.nn.Module):
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


class GAE_S2(torch.nn.Module):
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
        self.decoder = InnerProductDecoder_S2() if decoder is None else decoder
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


class VGAE_S2(GAE_S2):
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

class VariationalGCNEncoder_S2(torch.nn.Module):
    def __init__(self, in_channels, out_channels,batch_size,n_mols):
        
        self.embedding_size1 = 15
        self.embedding_size2 = 9
        self.linear_size1 = 100
        self.linear_size2 = 4
        
        self.batch_size = batch_size
        self.n_mols = n_mols
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
        
        self.transform = Linear(self.linear_size2*self.n_mols,self.out_channels)
        
        self.mu = Linear(self.out_channels, self.out_channels)
        self.logstd = Linear(self.out_channels, self.out_channels)

    def forward(self, x, edge_index):
        self.batch_size = x.shape[0]//self.n_mols
        x = self.conv1(x, edge_index)
        x = self.head_transform1(x)
        x = self.bn1(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        
        x = self.linear1(x)
        x = F.leaky_relu(x)

        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = x.reshape(self.batch_size,self.n_mols,-1)
        x = x.reshape(self.batch_size,-1)
        
        x = self.transform(x)
        x = F.leaky_relu(x)
        
        
        x,y,z = self.mu(x), self.logstd(x), edge_index
        return x,y,z

class VariationalGCNDecoder_S2(torch.nn.Module):
    def __init__(self,in_channels,out_channels,batch_size,n_mols):
        self.embedding_size1 = 9
        self.embedding_size2 = 3
        self.embedding_size3 = 3
        self.linear_size1 = 512
        self.linear_size2 = 128
        self.batch_size = batch_size
        self.n_mols = n_mols
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()
        self.inv_transform = Linear(self.in_channels,self.n_mols) 
        
        self.conv1 = GCNConv(1, self.embedding_size1)
        self.bn1 = BatchNorm(self.embedding_size1)

        self.conv2 = GCNConv(self.embedding_size1,self.embedding_size2)
        self.bn2 = BatchNorm(self.embedding_size2)

        self.conv3 = GCNConv(self.embedding_size2,self.embedding_size3)

        self.linear1 = Linear(self.embedding_size3, self.linear_size1)
        self.linear2 = Linear(self.linear_size1, self.linear_size2)
        self.linear3 = Linear(self.linear_size2, self.out_channels)

    def forward(self, x, edge_index, sigmoid=True):
        self.batch_size = x.shape[0]//self.n_mols
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
    
