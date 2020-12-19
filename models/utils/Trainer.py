import torch
from torch import nn, optim

from models.utils.Visual import draw_graph
from models.utils.Data import KGEDataset


class Trainer:
    def __init__(self, model, dateset: KGEDataset, params: dict={}):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.model.to(self.device)

        self.dateset = dateset
        self.train_loader = self.dateset.loaders["train"]
        self.val_loader = self.dateset.loaders["valid"]

        self.criterion = nn.Softplus().to(self.device)

        if params["optimizer"] is "Adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=params["lr"],
                # weight_decay=1e-2
            )
        elif params["optimizer"] is "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters()
            )
        else:
            raise ValueError("unsupported optimizer !")

        # weight of regularization term
        self.Lambda = params["Lambda"]

    def cal_loss(self, scores, regul_term):
        positive_loss = torch.sum(self.criterion(-1 * scores))
        tot_loss = positive_loss + self.Lambda * regul_term
        # print(f"{tot_loss} = {positive_loss} + {self.Lambda * regul_term}")
        return tot_loss, positive_loss

    def one_epoch(self, epoch: int):
        # >>> train
        batch_train_losses = []
        batch_pos_losses = []
        batch_regul_losses = []

        self.model.train()
        for x_batch in self.train_loader:
            x_batch = x_batch.to(self.device)

            scores, regul_term = self.model(x_batch)
            tot_loss, pos_loss = self.cal_loss(scores, regul_term)

            self.optimizer.zero_grad()
            tot_loss.backward()
            self.optimizer.step()
            # save loss on current batch

            batch_train_losses.append(tot_loss.data)
            # batch_pos_losses.append(pos_loss.data)
            # batch_regul_losses.append(regul_term)

        # <<< for batch in trainLoader
        avg_train_loss = sum(batch_train_losses) / len(batch_train_losses)
        # avg_train_pos_loss = sum(batch_pos_losses) / len(batch_pos_losses)
        # avg_train_regul_loss = sum(batch_regul_losses) / len(batch_regul_losses)
        print(f"epoch {epoch + 1} train loss: {avg_train_loss: .6f}")
        # print(f"epoch {epoch + 1} train pos loss: {avg_train_pos_loss: .6f}")
        # print(f"epoch {epoch + 1} train regul: {avg_train_regul_loss: .6f}")
        # <<< train

        # >>> validation
        batch_val_losses = []
        self.model.eval()
        with torch.no_grad():
            for x_batch in self.val_loader:
                x_batch = x_batch.to(self.device)
                scores, regul_term = self.model(x_batch)
                tot_loss, pos_loss = self.cal_loss(scores, regul_term)
                batch_val_losses.append(tot_loss.data)
            # <<< for batch in val_loader
        avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
        print(f"epoch {epoch + 1} val loss: {avg_val_loss: .6f}",
              end='\n\n')
        # <<< val
        return avg_train_loss, avg_val_loss

    def save_model(self):
        pass

    def train(self, args: dict):
        epochs = args["epochs"]

        result = {
            "epochs": epochs,
            "train_loss": [],
            "val_loss": []
        }

        for epoch in range(epochs):
            train_loss, val_loss = self.one_epoch(epoch)
            result["train_loss"].append(train_loss)
            result["val_loss"].append(val_loss)
        # <<< for epoch
        torch.save(self.model, f"./result/model_epoch{epochs}.pkl")
        draw_graph(result, smooth=False)
