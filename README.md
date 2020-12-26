# KG Embedding Demo

尝试使用 “Stay Positive: Knowledge Graph Embedding Without Negative Sampling” 中的方法优化 KGE 任务

## 方法

在该论文中，提出了 ```Stay Positive``` 训练方法，其中：

- 完全不使用 负采样，即不生成负样本，只使用 KG 中的事实 ( fact )
- KGE 损失定义为 正例损失 与 正则化项 的和

$$
\mathcal{L}(\theta)=\sum_{(\mathrm{h}, r, \mathrm{t}) \in \mathcal{G}_{\mathcal{E}, \mathcal{R}}} \mathcal{L}^{+}\left(\phi_{\mathcal{M}(\theta)}(\mathrm{h}, \mathrm{r}, \mathrm{t})\right)+\lambda \mathcal{L}^{s p}\left(\phi_{\mathcal{M}(\theta)}\right)
$$



## 结果

```json
args = {
    "data_dir": "./data/wn18am/",
    "embedding_size": 128,
    "batch_size": 128,
    "epochs": 1000,
    "optimizer": "Adagrad",
    # early-stopping patience
    "patience": 20,
    # learning rate
    "lr": 1e-3,
    # weight of regularization term
    "Lambda": 1e-3,
    "Zeta": 1e-5,
    "P": 1,
    "I": 20,
}
```

```json
Filtered setting:
	Hit@1 = 0.0
	Hit@3 = 0.0001709986320109439
	Hit@10 = 0.0005129958960328317
	MR = 15998.95417236662
	MRR = 0.00034867075199669146
```

epoch 549 stop with Early-Stopping, patience: 20

![image-20201220100049375](http://47.93.245.14:9000/images/2020/12/20/image-20201220100049375.png)

