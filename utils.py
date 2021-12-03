import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd
import dgl



def load_data_2(directory):
    # chemical_similarity as feature
    TFSM_1 = np.loadtxt('/directory/data/chemical_similarity/TF_TF_chemical_similarity.txt')
    NTFSM_1 = np.loadtxt('/directory/data/chemical_similarity/NTF_NTF_chemical_similarity.txt')
    TF_NTF_SM_1 = np.loadtxt('/directory/data/chemical_similarity/TF_NTF_chemical_similarity.txt')
    NTF_TF_SM_1 = np.loadtxt('/directory/data/chemical_similarityNTF_TF_chemical_similarity.txt')

    # seq_similarity as feature
    TFSM_2 = np.loadtxt('/directory/data//seq_similarity/TF_TF_seq_similarity.txt')
    NTFSM_2 = np.loadtxt('/directory/data/seq_similarity/NTF_NTF_seq_similarity.txt')
    TF_NTF_SM_2 = np.loadtxt('/directory/data/seq_similarity/TF_NTF_seq_similarity.txt')
    NTF_TF_SM_2 = np.loadtxt('/directory/data/seq_similarity/NTF_TF_seq_similarity.txt')
    
    return TFSM_1, NTFSM_1, TF_NTF_SM_1, NTF_TF_SM_1,TFSM_2, NTFSM_2, TF_NTF_SM_2, NTF_TF_SM_2



def sample(directory, random_seed):
    all_associations = pd.read_csv('/directory/data/all_TFgene_NTFgene_pairs.csv', names=['TFgene', 'NTFgene', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]

    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)  

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    #all_associations.reset_index(drop=True, inplace=True)

    return sample_df.values



def build_graph(directory, random_seed, ctx):
    # dgl.load_backend('mxnet')
    TFSM, NTFSM, TF_NTF_SM, NTF_TF_SM,TFSM_2, NTFSM_2, TF_NTF_SM_2, NTF_TF_SM_2 = load_data(directory)
    samples = sample(directory, random_seed)

    print('Building graph ...')
    g1 = dgl.DGLGraph(multigraph=True)
    g1.add_nodes(TFSM.shape[1] + NTFSM.shape[1])
    node_type = nd.zeros(g1.number_of_nodes(), dtype='float32', ctx=ctx)
    node_type[:TFSM.shape[1]] = 1
    g = g1.to(ctx)   
    g.ndata['type'] = node_type

    # concate features
    print('Adding TFgene features ...')
    TFgene_data = nd.zeros(shape=(g.number_of_nodes(), TFSM.shape[1]+TFSM_2.shape[1]), dtype='float32', ctx=ctx)
    TFgene_data[:TFSM.shape[0], :TFSM.shape[1]] = nd.from_numpy(TFSM)
    TFgene_data[:TFSM.shape[0],TFSM.shape[1]:TFSM.shape[1]+TFSM_2.shape[1]] = nd.from_numpy(TFSM_2)
    TFgene_data[TFSM.shape[0]: TFSM.shape[0]+NTFSM.shape[0], :NTF_TF_SM.shape[1]] = nd.from_numpy(NTF_TF_SM)
    TFgene_data[TFSM.shape[0]: TFSM.shape[0]+NTFSM.shape[0], NTF_TF_SM.shape[1]:NTF_TF_SM.shape[1]+NTF_TF_SM_2.shape[1]] = nd.from_numpy(NTF_TF_SM_2)
    g.ndata['TFgene_features'] = TFgene_data

    print('Adding NTFgene features ...')
    NTFgene_data = nd.zeros(shape=(g.number_of_nodes(), NTFSM.shape[1]+NTFSM_2.shape[1]), dtype='float32', ctx=ctx)
    NTFgene_data[:TFSM.shape[0], :TF_NTF_SM.shape[1]] = nd.from_numpy(TF_NTF_SM)
    NTFgene_data[:TFSM.shape[0],TF_NTF_SM.shape[1]:TF_NTF_SM.shape[1]+TF_NTF_SM_2.shape[1]] = nd.from_numpy(TF_NTF_SM_2)
    NTFgene_data[TFSM.shape[0]: TFSM.shape[0]+NTFSM.shape[0], :NTFSM.shape[1]] = nd.from_numpy(NTFSM)
    NTFgene_data[TFSM.shape[0]: TFSM.shape[0]+NTFSM.shape[0], NTFSM.shape[1]:NTFSM.shape[1]+NTFSM_2.shape[1]] = nd.from_numpy(NTFSM_2)
    g.ndata['NTFgene_features'] = NTFgene_data

    print('Adding edges ...')
    TFgene_ids = list(range(1, TFSM.shape[1] + 1))
    NTFgene_ids = list(range(1, NTFSM.shape[1]+1))

    TFgene_ids_invmap = {id_: i for i, id_ in enumerate(TFgene_ids)}
    NTFgene_ids_invmap = {id_: i for i, id_ in enumerate(NTFgene_ids)}

    sample_TFgene_vertices = [TFgene_ids_invmap[id_] for id_ in samples[:, 0]]
    sample_NTFgene_vertices = [NTFgene_ids_invmap[id_] + TFSM.shape[0] for id_ in samples[:, 1]]

    g.add_edges(sample_TFgene_vertices, sample_NTFgene_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    
    g.add_edges(sample_NTFgene_vertices, sample_TFgene_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
                      
    g.readonly()
    print('Successfully build graph !!')

    return g, TFgene_ids_invmap, NTFgene_ids_invmap, TFSM
