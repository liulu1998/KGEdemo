import torch
import numpy as np
from torch.utils.data import DataLoader
from models.utils.Measure import Measure
from models.utils.Data import KGEDataset


class Tester:
    def __init__(self, model, dateset, test_loader):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.model.eval()

        self.dataset = dateset

        self.test_loader = test_loader
        self.measure = Measure()
        self.all_facts_as_set_of_tuples = set(self.all_facts_as_tuples(dateset))

        print("num of all facts:", end=' ')
        print(len(self.all_facts_as_set_of_tuples))

    def get_rank(self, sim_scores):
        # assuming the test fact is the first one
        return (sim_scores >= sim_scores[0]).sum()

    def create_queries(self, fact, head_or_tail):
        head, rel, tail = fact
        n_entity = self.dataset.get_entity_num()

        if head_or_tail == "head":
            return [(i, rel, tail) for i in range(n_entity)]
        elif head_or_tail == "tail":
            return [(head, rel, i) for i in range(n_entity)]

    def add_fact_and_shred(self, fact, queries, raw_or_fil):
        if raw_or_fil == "raw":
            result = [tuple(fact)] + queries
        elif raw_or_fil == "fil":
            result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)
        # return self.shred_facts(result)
        # 不用 shred
        return torch.LongTensor(result).to(self.device)

    def shred_facts(self, triples):
        heads = [triples[i][0] for i in range(len(triples))]
        rels = [triples[i][1] for i in range(len(triples))]
        tails = [triples[i][2] for i in range(len(triples))]
        return torch.LongTensor(heads).to(self.device), \
               torch.LongTensor(rels).to(self.device), \
               torch.LongTensor(tails).to(self.device)

    def all_facts_as_tuples(self, dataset: KGEDataset):
        tuples = []
        # spl: "train" "val" "test"
        for fact in dataset.data["train"]:
            tuples.append(tuple(fact))
        for fact in dataset.data["valid"]:
            tuples.append(tuple(fact))
        for fact in dataset.data["test"]:
            tuples.append(tuple(fact))
        return tuples

    def test(self):
        settings = ["raw", "fil"]

        for i, fact in enumerate(self.test_loader):
            # batch size = 1
            fact = fact[0]
            for head_or_tail in ["head", "tail"]:
                queries = self.create_queries(fact, head_or_tail)
                for raw_or_fil in settings:
                    facts = self.add_fact_and_shred(fact, queries, raw_or_fil)
                    sim_scores, regul_term = self.model(facts)
                    sim_scores = sim_scores.cpu().data.numpy()
                    rank = self.get_rank(sim_scores)
                    self.measure.update(rank, raw_or_fil)

        self.measure.normalize(len(self.test_loader.dataset))
        self.measure.print_()
        return self.measure.mrr["fil"]
