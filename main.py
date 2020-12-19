from models.KGEModels import DistMult
from models.utils.Tester import Tester
from models.utils.Trainer import Trainer
from models.utils.Data import KGEDataset
import torch

args = {
    "data_dir": "./data/wn18am/",
    "batch_size": 128,
    "optimizer": "Adagrad",
    "epochs": 10,
    "lr": 1e-3,
    "Lambda": 1e-3,
    "embedding_size": 128,
    "Zeta": 1e-5,
    "P": 1,
    "I": 20,
}

# 是否加载模型
LOAD_MODEL = True
# 是否 test
TEST = True


if __name__ == '__main__':
    torch.cuda.empty_cache()

    dataset = KGEDataset(args)

    n_entity, n_relation = dataset.get_entity_num(), dataset.get_relation_num()

    if LOAD_MODEL:
        model = torch.load("./result/model_epoch10.pkl")
        print(model)
    else:
        model = DistMult(n_entity, n_relation, args)
        trainer = Trainer(model=model, dateset=dataset, params=args)
        trainer.train(args)

    if TEST:
        # load test set and evaluate model
        test_loader = dataset.loaders["test"]

        tester = Tester(model, dataset, test_loader)
        mrr = tester.test()

        print(f"MRR: {mrr}")
