import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_sparse
from torch import FloatTensor
from geomloss import SamplesLoss

class H2GCN(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            k: int = 2,
            dropout: float = 0.5,
            use_relu: bool = True
    ):
        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.k = k
        self.act = F.relu if use_relu else lambda x: x
        self.use_relu = use_relu
        self.w_embed = nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )

        # self.w_classify = nn.Parameter(
        #     torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
        #     requires_grad=True
        # )
        # self.params = [self.w_embed, self.w_classify]
        self.w1 = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, 1)),
            requires_grad=True
        )
        self.w0 = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, 1)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w1, self.w0]

        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w1)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, adj, x, t, dt_idx):
        '''
        :param adj: torch.sparse.Tensor
        :param x: FloatTensor
        :return: FloatTensor:
        '''
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        # feature embedding stage
        rs = [self.act(torch.mm(x, self.w_embed))]  # (num_nodes, feature_dim) * (feature_dim, hidden_dim)
        # neighborhood aggregation stage
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)  # formula (7)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        # train_test_split
        r_final, t = r_final[dt_idx], t[dt_idx]

        samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.005, backend="tensorized")
        imbalance_loss = samples_loss(r1[torch.where(t == 0)[0]], r1[torch.where(t == 1)[0]]) +\
                         samples_loss(r2[torch.where(t == 0)[0]], r2[torch.where(t == 1)[0]])

        # TODO: check the dimension
        pred_0 = torch.mm(r_final[torch.where(t == 0)[0]], self.w0)
        pred_1 = torch.mm(r_final[torch.where(t == 1)[0]], self.w1)
        pred_c0 = torch.mm(r_final[torch.where(t == 1)[0]], self.w0)
        pred_c1 = torch.mm(r_final[torch.where(t == 0)[0]], self.w1)
        return pred_0.double(), pred_1.double(), pred_c0.double(), pred_c1.double(), imbalance_loss
