from models.KGEModels import DistMult
from models.utils.Tester import Tester
from models.utils.Trainer import Trainer
from models.utils.Data import KGEDataset
import torch

# 是否加载模型
LOAD_MODEL = False
# 是否 test
TEST = False

args = {
    "data_dir": "./data/wn18am/",
    "embedding_size": 128,
    "batch_size": 128,
    "epochs": 10,
    "optimizer": "Adagrad",
    # early-stopping patience
    "patience": 2,
    # learning rate
    "lr": 1,
    # weight of regularization term
    "Lambda": 1e-3,
    "Zeta": 1e-5,
    "P": 1,
    "I": 20,
}


if __name__ == '__main__':
    torch.cuda.empty_cache()

    dataset = KGEDataset(args)
    n_entity, n_relation = dataset.get_entity_num(), dataset.get_relation_num()
    print("dataset loaded !")

    if LOAD_MODEL:
        checkpoint = torch.load("./result/checkpoint.pth")
        model = checkpoint["model"]
        # model = torch.load("./result/model_epoch10.pkl")
        print(f"model loaded !")
    else:
        model = DistMult(n_entity, n_relation, args)
        print("model initialized !")
        trainer = Trainer(model=model, dateset=dataset, params=args)
        print("start training !")
        trainer.train(args)

    if TEST:
        # load test set and evaluate model
        test_loader = dataset.loaders["test"]

        tester = Tester(model, dataset, test_loader)
        mrr = tester.test()

        print(f"MRR: {mrr}")
