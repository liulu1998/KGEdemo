from torch.utils.data import Dataset, DataLoader


class KGEDataset(Dataset):
    def __init__(self, n_entity, n_relation, facts):
        # only including facts (positive data), without negative sampling
        # i.e. only use facts in KG and do not generate negative data
        self.facts = facts
        self.facts_set = set(self.facts)
        self.length = len(self.facts)
        # number of entities and relations
        self.n_entity = n_entity
        self.n_relation = n_relation

    def __getitem__(self, index: int):
        positive_data = self.facts[index]
        # only use positive data
        return positive_data

    def __len__(self) -> int:
        return self.length
