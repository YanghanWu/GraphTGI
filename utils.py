import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd
import dgl



def load_data(directory):
    # chemical_similarity as feature
    TFSM_1 = np.loadtxt('/directory/data/chemical_similarity/TF_TF_chemical_similarity.txt')
    tgSM_1 = np.loadtxt('/directory/data/chemical_similarity/tg_tg_chemical_similarity.txt')
    TF_tg_SM_1 = np.loadtxt('/directory/data/chemical_similarity/TF_tg_chemical_similarity.txt')
    tg_TF_SM_1 = np.loadtxt('/directory/data/chemical_similarity/tg_TF_chemical_similarity.txt')

    # seq_similarity as feature
    TFSM_2 = np.loadtxt('/directory/data//seq_similarity/TF_TF_seq_similarity.txt')
    tgSM_2 = np.loadtxt('/directory/data/seq_similarity/tg_tg_seq_similarity.txt')
    TF_tg_SM_2 = np.loadtxt('/directory/data/seq_similarity/TF_tg_seq_similarity.txt')
    tg_TF_SM_2 = np.loadtxt('/directory/data/seq_similarity/tg_TF_seq_similarity.txt')
    
    return TFSM_1, tgSM_1, TF_tg_SM_1, tg_TF_SM_1,TFSM_2, tgSM_2, TF_tg_SM_2, tg_TF_SM_2



def sample(directory, random_seed):
    all_associations = pd.read_csv('/directory/data/all_TF_tg_pairs.csv', names=['TF', 'tg', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]

    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)  

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    #all_associations.reset_index(drop=True, inplace=True)

    return sample_df.values



def build_graph(directory, random_seed, ctx):
    # dgl.load_backend('mxnet')
    TFSM, tgSM, TF_tg_SM, tg_TF_SM,TFSM_2, tgSM_2, TF_tg_SM_2, tg_TF_SM_2 = load_data(directory)
    samples = sample(directory, random_seed)

    print('Building graph ...')
    g1 = dgl.DGLGraph(multigraph=True)
    g1.add_nodes(TFSM.shape[1] + tgSM.shape[1])
    node_type = nd.zeros(g1.number_of_nodes(), dtype='float32', ctx=ctx)
    node_type[:TFSM.shape[1]] = 1
    g = g1.to(ctx)   
    g.ndata['type'] = node_type

    # concate features
    print('Adding TF features ...')
    TF_data = nd.zeros(shape=(g.number_of_nodes(), TFSM.shape[1]+TFSM_2.shape[1]), dtype='float32', ctx=ctx)
    TF_data[:TFSM.shape[0], :TFSM.shape[1]] = nd.from_numpy(TFSM)
    TF_data[:TFSM.shape[0],TFSM.shape[1]:TFSM.shape[1]+TFSM_2.shape[1]] = nd.from_numpy(TFSM_2)
    TF_data[TFSM.shape[0]: TFSM.shape[0]+tgSM.shape[0], :tg_TF_SM.shape[1]] = nd.from_numpy(tg_TF_SM)
    TF_data[TFSM.shape[0]: TFSM.shape[0]+tgSM.shape[0], tg_TF_SM.shape[1]:tg_TF_SM.shape[1]+tg_TF_SM_2.shape[1]] = nd.from_numpy(tg_TF_SM_2)
    g.ndata['TF_features'] = TF_data

    print('Adding target gene features ...')
    tg_data = nd.zeros(shape=(g.number_of_nodes(), tgSM.shape[1]+tgSM_2.shape[1]), dtype='float32', ctx=ctx)
    tg_data[:TFSM.shape[0], :TF_tg_SM.shape[1]] = nd.from_numpy(TF_tg_SM)
    tg_data[:TFSM.shape[0],TF_tg_SM.shape[1]:TF_tg_SM.shape[1]+TF_tg_SM_2.shape[1]] = nd.from_numpy(TF_tg_SM_2)
    tg_data[TFSM.shape[0]: TFSM.shape[0]+tgSM.shape[0], :tgSM.shape[1]] = nd.from_numpy(tgSM)
    tg_data[TFSM.shape[0]: TFSM.shape[0]+tgSM.shape[0], tgSM.shape[1]:tgSM.shape[1]+tgSM_2.shape[1]] = nd.from_numpy(tgSM_2)
    g.ndata['tg_features'] = tg_data

    print('Adding edges ...')
    TF_ids = list(range(1, TFSM.shape[1] + 1))
    tg_ids = list(range(1, tgSM.shape[1]+1))

    TF_ids_invmap = {id_: i for i, id_ in enumerate(TF_ids)}
    tg_ids_invmap = {id_: i for i, id_ in enumerate(tg_ids)}

    sample_TF_vertices = [TF_ids_invmap[id_] for id_ in samples[:, 0]]
    sample_tg_vertices = [tg_ids_invmap[id_] + TFSM.shape[0] for id_ in samples[:, 1]]

    g.add_edges(sample_TF_vertices, sample_tg_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    
    g.add_edges(sample_tg_vertices, sample_TF_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
                      
    g.readonly()
    print('Successfully build graph !!')

    return g, TF_ids_invmap, tg_ids_invmap, TFSM
