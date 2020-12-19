import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def read_dict(data_dir, file_name):
    """ read dict from disk
    """
    data = dict()
    with open(os.path.join(data_dir, file_name), 'r') as f:
        for line in f:
            id_, name = line.strip().split("\t")
            data[name] = int(id_)
    return data


def read_triples(data_dir, file_name: str, entity2id: dict, relation2id: dict):
    """ read triples from dist and convert name to id
    """
    triples = []
    with open(os.path.join(data_dir, file_name), "r") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            triples.append([entity2id[h], relation2id[r], entity2id[t]])
    return np.array(triples)


class PartDataSet(Dataset):
    def __init__(self, triples: np.array):
        self.triples = torch.from_numpy(triples).long()
        self.length = self.triples.size()[0]

    def __getitem__(self, index: int):
        return self.triples[index]

    def __len__(self) -> int:
        return self.length


class KGEDataset(Dataset):
    def __init__(self, args: dict):
        data_dir = args["data_dir"]
        self.entity2id = read_dict(data_dir, "entities.dict")
        self.relation2id = read_dict(data_dir, "relations.dict")

        # number of entities and relations
        self.n_entity = len(self.entity2id)
        self.n_relation = len(self.relation2id)

        # only including facts (positive data), without negative sampling
        # i.e. only use facts in KG and do not generate negative data
        self.data = self.get_data(data_dir)
        self.loaders = self.get_loaders(args)

    def get_data(self, data_dir):
        data = {}
        for t in ["train", "valid", "test"]:
            triples = read_triples(data_dir, t+".txt", self.entity2id, self.relation2id)
            data[t] = PartDataSet(triples)
        return data

    def get_loaders(self, args: dict):
        loaders = {}
        for t in ["train", "valid", "test"]:
            loaders[t] = DataLoader(
                dataset=self.data[t], batch_size=1 if t is "test" else args["batch_size"],
                shuffle=True, num_workers=4,
                pin_memory=True
            )
        return loaders

    def get_entity_num(self):
        return self.n_entity

    def get_relation_num(self):
        return self.n_relation
