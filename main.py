import os
from models.KGEModels import DistMult
from models.utils.Trainer import Trainer
from models.utils.Data import KGEDataset

import torch
import numpy as np
from torch.utils.data import DataLoader


DATA_DIR = "./data/wn18rr/"

EMBEDDING_SIZE = 128
ZETA = 0.001
P = 1
I = 5

BATCH_SIZE = 32
EPOCH = 10

train_params = {
    "Lambda": 1e-3,
    "L": -1
}


def read_dict(file_name):
    """ read dict from disk
    """
    data = dict()
    with open(os.path.join(DATA_DIR, file_name), 'r') as f:
        for line in f:
            id_, name = line.strip().split("\t")
            data[name] = int(id_)
    return data


def read_triples(file_name: str, entity2id: dict, relation2id: dict):
    """ read triples from dist and convert name to id
    """
    triples = []
    with open(os.path.join(DATA_DIR, file_name), "r") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            triples.append([entity2id[h], relation2id[r], entity2id[t]])
    return np.array(triples)


if __name__ == '__main__':

    torch.cuda.empty_cache()

    # for entities, name -> int
    entity2id = read_dict("entities.dict")
    # that for relations
    relation2id = read_dict("relations.dict")

    # number of entities and relations in KG
    n_entity = len(entity2id)
    n_relation = len(relation2id)

    print(f"number of entities: {n_entity}")
    print(f"number of relations: {n_relation}")

    # train set & validation set, both are GPU Tensor
    train_triples = torch.from_numpy(read_triples("train.txt", entity2id, relation2id)).long()
    val_triples = torch.from_numpy(read_triples("valid_clean.txt", entity2id, relation2id)).long()

    # train DataSet and corresponding DataLoader
    train_set = KGEDataset(
        n_entity, n_relation,
        facts=train_triples
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # validation DateSet and corresponding DataLoader
    val_set = KGEDataset(
        n_entity, n_relation,
        facts=val_triples
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    print(f"train set size: {len(train_set)}")
    print(f"val set size:   {len(val_set)}")

    model = DistMult(
        n_entity, n_relation, embedding_size=EMBEDDING_SIZE,
        Zeta=ZETA, P=P, I=I
    )

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=None, params=train_params
    )

    trainer.train(tot_epoch=EPOCH)

    # load test set and evaluate model
    # test_triples = torch.from_numpy(read_triples("test_clean.txt", entity2id, relation2id)).long()
    #
    # test_set = KGEDataset(
    #     n_entity, n_relation, test_triples
    # )
    #
    # print(f"test set size: {len(test_set)}")