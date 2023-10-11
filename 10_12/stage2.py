from stage1 import *
model1="Intra.pt"
stage1=torch.load("./models/"+model1)

def condenseFrame(frame):
    mol_pos=[]
    edge_list=np.array([[ 0,  1,  0,  2],
                        [ 1,  0,  2,  0]])
    for mol_no,res in enumerate(frame.top.residues):
        pos=[]
        for atom in res.atoms:
            mol_pos.append(frame.xyz[0][atom.index])
        pos=np.array(pos)
        avg_pos=np.mean(pos,axis=0)
        atomic_nums = np.array([[atom.element.atomic_number for atom in res.atoms]]).T
        vdwr = np.array([[atom.element.radius for atom in res.atoms]]).T
        node_features = np.concatenate((pos,vdwr,atomic_nums),axis=1)
        graph = data.Data(x=torch.from_numpy(node_features),edge_index=torch.from_numpy(edge_list)).to("cuda")
        encoded = stage1.encode(graph.x,graph.edge_index)
        model_out = np.mean(encoded[0].detach().cpu().numpy(),axis=0)
        feature = np.concatenate((model_out,avg_pos))
        mol_pos.append(feature)
    mol_pos = np.array(mol_pos)
    return mol_pos
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


def condenseAllFrames(frames):
    """
    Condenses all frames in a trajectory
    """
    condensed_frames = []
    for frame in tqdm(frames):
        condensed_frames.append(condenseFrame(frame))
    return np.array(condensed_frames)
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
        recentered = mol_pos 
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