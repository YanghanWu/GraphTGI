import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
from mxnet.gluon.nn import activations
from mxnet.ndarray.gen_op import Activation
import numpy as np
from dgl.nn import  GATConv

class GraphTGI(nn.Block):
    def __init__(self, encoder, decoder):
        super(GraphTGI, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, TF, tg):
        h = self.encoder(G)
        
        h_TF = h[TF]
        h_tg = h[tg]

        
        return self.decoder(h_TF, h_tg),G.ndata['h']


class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, n_layers, G, aggregator, dropout, slope, ctx):
        super(GraphEncoder, self).__init__()

        self.G = G
        self.TF_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.tg_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)

        self.layers = nn.Sequential()
        
        in_feats = embedding_size
        
        self.layers.add(GATConv(embedding_size, embedding_size, 2, feat_drop = dropout,attn_drop = 0.5, negative_slope = 0.5, residual = True,allow_zero_in_degree=True))
        self.layers.add(GATConv(embedding_size, embedding_size, 2, feat_drop = dropout,attn_drop = 0.5, negative_slope = 0.5, residual = True,allow_zero_in_degree=True))
         

        self.TF_emb = TFEmbedding(embedding_size, dropout)
        self.tg_emb = tgEmbedding(embedding_size, dropout)

    def forward(self, G):
        # Generate embedding on disease nodes and mirna nodesd
        assert G.number_of_nodes() == self.G.number_of_nodes()


        G.apply_nodes(lambda nodes: {'h': self.TF_emb(nodes.data)}, self.TF_nodes)
        G.apply_nodes(lambda nodes: {'h': self.tg_emb(nodes.data)}, self.tg_nodes)

        for layer in self.layers:
            layer(G,G.ndata['h'])

        return G.ndata['h']


class TFEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(TFEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=True))
            seq.add(nn.Dropout(dropout))
        self.proj_TF = seq

    def forward(self, ndata):
        extra_repr = self.proj_TF(ndata['TF_features'])

        return extra_repr


class tgEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(tgEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=True))
            seq.add(nn.Dropout(dropout))
        self.proj_tg = seq

    def forward(self, ndata):
        extra_repr = self.proj_tg(ndata['tg_features'])
        return extra_repr


class BilinearDecoder(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        self.activation = nn.Activation('sigmoid')
        with self.name_scope():
            self.W = self.params.get('dot_weights', shape=(feature_size, feature_size))

    def forward(self, h_TF, h_tg):
        
        results_mask = self.activation((nd.dot(h_TF, self.W.data()) * h_tg).sum(1))

        return results_mask
