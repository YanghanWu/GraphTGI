import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
from mxnet.gluon.nn import activations
from mxnet.ndarray.gen_op import Activation
import numpy as np
from dgl.nn import  GATConv

class GNNMDA(nn.Block):
    def __init__(self, encoder, decoder):
        super(GNNMDA, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, TFgenes, NTFgenes):
        h = self.encoder(G)
        
        h_TFgenes = h[TFgenes]
        h_NTFgenes = h[NTFgenes]

        
        return self.decoder(h_TFgenes, h_NTFgenes),G.ndata['h']


class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, n_layers, G, aggregator, dropout, slope, ctx):
        super(GraphEncoder, self).__init__()

        self.G = G
        self.TFgene_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.NTFgene_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)

        self.layers = nn.Sequential()
        
        in_feats = embedding_size
        
        self.layers.add(GATConv(embedding_size, embedding_size, 2, feat_drop = dropout,attn_drop = 0.5, negative_slope = 0.5, residual = True,allow_zero_in_degree=True))
        self.layers.add(GATConv(embedding_size, embedding_size, 2, feat_drop = dropout,attn_drop = 0.5, negative_slope = 0.5, residual = True,allow_zero_in_degree=True))
         

        self.TFgene_emb = TFgeneEmbedding(embedding_size, dropout)
        self.NTFgene_emb = NTFgeneEmbedding(embedding_size, dropout)

    def forward(self, G):
        # Generate embedding on disease nodes and mirna nodesd
        assert G.number_of_nodes() == self.G.number_of_nodes()


        G.apply_nodes(lambda nodes: {'h': self.TFgene_emb(nodes.data)}, self.TFgene_nodes)
        G.apply_nodes(lambda nodes: {'h': self.NTFgene_emb(nodes.data)}, self.NTFgene_nodes)

        for layer in self.layers:
            layer(G,G.ndata['h'])

        return G.ndata['h']


class TFgeneEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(TFgeneEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=True))
            seq.add(nn.Dropout(dropout))
        self.proj_TFgene = seq

    def forward(self, ndata):
        extra_repr = self.proj_TFgene(ndata['TFgene_features'])

        return extra_repr


class NTFgeneEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(NTFgeneEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=True))
            seq.add(nn.Dropout(dropout))
        self.proj_NTFgene = seq

    def forward(self, ndata):
        extra_repr = self.proj_NTFgene(ndata['NTFgene_features'])
        return extra_repr


class BilinearDecoder(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        self.activation = nn.Activation('sigmoid')
        with self.name_scope():
            self.W = self.params.get('dot_weights', shape=(feature_size, feature_size))

    def forward(self, h_TFgene, h_NTFgene):
        
        results_mask = self.activation((nd.dot(h_TFgene, self.W.data()) * h_NTFgene).sum(1))

        return results_mask
