from typing import Any

import torch
from torch import nn
from torch import tanh
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, n_entity, n_relation, embedding_size: int):
        super(BaseModel, self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.embedding_size = embedding_size

    def forward(self, x):
        return None


class DistMult(BaseModel):
    def __init__(self, n_entity, n_relation, embedding_size: int,
                 Zeta: float, P: int = 2, I=5):
        """
        :param embedding_size: entity space and relation space share the same embedding size
        :param L: loss = softplus(-L * score(h, r, t)), L is a hyper-parameter
        :param Lambda: weight of whole regularization term
        :param Zeta: weight of |E|^2 * |R| in the regularization term
        :param P: 正则化项中, 指定 p 阶范数
        """
        super().__init__(n_entity, n_relation, embedding_size)

        self.entity_emb = nn.Embedding(self.n_entity, self.embedding_size)
        # DistMult use dialog matrix, so each relation is represented as a vector
        self.relation_emb = nn.Embedding(self.n_relation, self.embedding_size)

        self.init_weights()

        # self.criterion = nn.Softplus().cuda()

        # self.L = L
        # self.Lambda = Lambda
        self.Zeta = Zeta
        self.P = P
        self.I = I

    def init_weights(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, x):
        # head entity, relation, tail entity
        h = tanh(self.entity_emb(x[:, 0]))
        r = tanh(self.relation_emb(x[:, 1]))
        t = tanh(self.entity_emb(x[:, 2]))

        # score of each fact, size: [batch_size]
        scores = self.I * torch.sum(h * r * t, 1)

        # loss on positive data
        # positive_loss = self.criterion(-self.L * scores)

        # positive_loss = torch.sum(positive_loss)

        # >>> calculate the regularization term
        cnt_entity = len(set(x[:, 0].tolist() + x[:, 2].tolist()))
        cnt_relation = len(set(x[:, 1].tolist()))

        sub_term = self.Zeta * (cnt_entity ** 2) * cnt_relation
        regul_term = torch.sum(scores - sub_term) ** self.P
        # regul_term = torch.sum(scores) ** self.P

        # regul_term = self.Lambda * torch.sum(scores) ** self.P

        # total loss = loss on positive data + regularization term
        # tot_loss = positive_loss + regul_term

        # print(f"{tot_loss} = {positive_loss} + {regul_term}")
        # return tot_loss
        return scores, regul_term
