import pandas as pd
import torch
# from torch.nn import BasicGNN
from torch_geometric.nn import GCNConv,GCN2Conv
from Positional_model import FTPositionalDecoder
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops,
)
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch_geometric.nn.inits import reset
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np
EPS = 1e-15
MAX_LOGSTD = 10
class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    # def forward(self, z, edge_index, sigmoid=True):
    #     r"""Decodes the latent variables :obj:`z` into edge probabilities for
    #     the given node-pairs :obj:`edge_index`.
    #
    #     Args:
    #         z (Tensor): The latent space :math:`\mathbf{Z}`.
    #         sigmoid (bool, optional): If set to :obj:`False`, does not apply
    #             the logistic sigmoid function to the output.
    #             (default: :obj:`True`)
    #     """
    #     value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    #     return torch.sigmoid(value) if sigmoid else value

    def forward(self, z, sigmoid=False):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
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
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder

        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    # def parameters(self, recurse=True):
    #     for name, param in self.named_parameters(recurse=recurse):
    #         yield param
    def forward(self, x, edge_index):
        z1, z2, z3 = self.encoder(x, edge_index)
        return z1

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)


    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
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
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder, feat_dim, zdim):
        super().__init__(encoder, decoder)
        self.input_feat_dim = feat_dim
        self.zdim = zdim
        self.fde1 = nn.Linear(self.zdim,512)
        self.fde2 = nn.Linear(512,self.input_feat_dim)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
    # def parameters(self, recurse=True):
    #     for name, param in self.named_parameters(recurse=recurse):
    #         yield param
    def forward(self, x, edge_index, coords):
        z1, z2, z3, po_emb = self.encode(x, edge_index, coords)
        return z1
    def feature_decoder(self, po_emb):
        h1 = F.relu(self.fde1(po_emb))
        return torch.relu(self.fde2(h1))
    def encode(self, *args, **kwargs):
        """"""
        self.__mu__, self.__logstd__, po_emb = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z, self.__mu__, self.__logstd__, po_emb

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


class VGAEencoder(torch.nn.Module):
    def __init__(self, input_feat_dim, zdim, pe_alpha, enc_type):
        super().__init__()
        self.pe_alpha = pe_alpha
        self.enc_type = enc_type
        self.fen1 = nn.Linear(input_feat_dim, 512)
        self.fen2 = nn.Linear(512, zdim)
        self.gc1 = GCNConv(zdim, 32, bias=False)
        self.gc2 = GCNConv(32, 3, bias=False)
        self.gc3 = GCNConv(32, 3, bias=False)
        self.dropout = 0.

    def forward(self, x, edge_index,coords):
        x, adj = x, edge_index
        fe_emb = self.feature_encoder(x)
        #---------Positional Encoder------#
        B = fe_emb.shape[0]
        D = fe_emb.shape[1]
        in_dim = fe_emb.shape[1]
        zdim = fe_emb.shape[1]
        # pe_alpha = 0.1
        petype = 'add'
        # enc_type = 'dummy'      #'geom_ft'
        enc_dim = None
        FTPE_model = FTPositionalDecoder(in_dim, D, petype, self.pe_alpha, enc_type=self.enc_type, enc_dim=enc_dim)
        assert coords.size(0) == fe_emb.size(0)
        z = fe_emb.view(fe_emb.size(0), *([1] * (coords.ndimension() - 2)), zdim)
        z = torch.cat((coords, z.expand(*coords.shape[:-1], zdim)), dim=-1)
        po_emb = FTPE_model(z)
        #----------------------------------#
        hidden1 = self.gc1(fe_emb, adj).tanh()
        hidden1 = F.dropout(hidden1, self.dropout)
        mu = self.gc2(hidden1, adj)
        mu = F.dropout(mu, self.dropout)
        logstd = self.gc3(hidden1, adj)
        logstd = F.dropout(logstd, self.dropout)
        return mu, logstd, po_emb

    def feature_encoder(self, x):
        emb0 = F.relu(self.fen1(x))
        emb1 = F.relu(self.fen2(emb0))
        return emb1

class GAEencoder(torch.nn.Module):
    def __init__(self, input_feat_dim):
        super().__init__()
        self.gc1 = GCNConv(input_feat_dim, 32, bias=False)
        self.gc2 = GCNConv(32, 3, bias=False)
        self.dropout = 0.

    # def forward(self, data):
    def forward(self, x, edge_index):
        x, adj = x, edge_index
        # hidden1 = self.gc1(x, adj).relu()
        hidden1 = self.gc1(x, adj).tanh()
        hidden1 = F.dropout(hidden1,self.dropout)
        z = self.gc2(hidden1, adj)
        z = F.dropout(z, self.dropout)
        # print('GAEmodel')
        return z, z, None
















