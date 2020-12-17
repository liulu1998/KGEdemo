import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models.utils.Visual import draw_graph


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer=None, params: dict={}):
        self.model: nn.Module = model
        # transfer to GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader

        self.criterion = nn.Softplus().cuda()
        self.optimizer = optim.Adagrad(
            self.model.parameters(),
            lr=1e-2,
            weight_decay=1e-2
        )

        # weight of regularization term
        self.Lambda = params["Lambda"]
        # weight in SoftPlus
        self.L = params["L"]

    def cal_loss(self, scores, regul_term):
        positive_loss = torch.sum(self.criterion(-self.L * scores))
        tot_loss = positive_loss + self.Lambda * regul_term
        print(f"{tot_loss} = {positive_loss} + {self.Lambda * regul_term}")
        return tot_loss

    def one_epoch(self, epoch: int):
        # >>> train
        batch_train_losses = []
        self.model.train()
        for x_batch in self.train_loader:
            x_batch = x_batch.cuda()

            scores, regul_term = self.model(x_batch)
            tot_loss = self.cal_loss(scores, regul_term)


            self.optimizer.zero_grad()
            tot_loss.backward()
            self.optimizer.step()
            # save loss on current batch
            batch_train_losses.append(tot_loss.data)
        # <<< for batch in trainLoader
        avg_train_loss = sum(batch_train_losses) / len(batch_train_losses)
        print(f"epoch {epoch + 1} train loss: {avg_train_loss: .6f}")
        # <<< train

        # >>> validation
        batch_val_losses = []
        self.model.eval()
        with torch.no_grad():
            for x_batch in self.val_loader:
                x_batch = x_batch.cuda()
                scores, regul_term = self.model(x_batch)
                tot_loss = self.cal_loss(scores, regul_term)
                batch_val_losses.append(tot_loss.data)
            # <<< for batch in val_loader
        avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
        print(f"epoch {epoch + 1} val loss: {avg_val_loss: .6f}",
              end='\n')
        # <<< val
        return avg_train_loss, avg_val_loss

    def save_model(self):
        pass

    def train(self, tot_epoch):
        result = {
            "epochs": tot_epoch,
            "train_loss": [],
            "val_loss": []
        }

        for epoch in range(tot_epoch):
            train_loss, val_loss = self.one_epoch(epoch)
            result["train_loss"].append(train_loss)
            result["val_loss"].append(val_loss)
        # <<< for epoch
        torch.save(self.model, f"./result/model_epoch{tot_epoch}.pkl")
        draw_graph(result, smooth=False)
