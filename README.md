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

### Result 1

参数设置：

```json
BATCH_SIZE = 32
EPOCH = 20

train_params = {
    "Lambda": 0.001,
    "L": -1
}

model = DistMult(
    n_entity, n_relation, 			
    embedding_size=50,
    Zeta=0.001, P=1
)

optimizer = Adagrad()
```



<img src="http://47.93.245.14:9000/images/2020/12/16/image-20201216191637550.png" alt="image-20201216191637550" style="zoom: 67%;" />

### Result 2

参数设置：
```json
BATCH_SIZE = 32
EPOCH = 20

train_params = {
    "Lambda": 0.001,
    "L": -1
}

model = DistMult(
    n_entity, n_relation, 			
    embedding_size=50,
    Zeta=0.001, P=1
)

optimizer = Adagrad()

在 model.forward 中, h r t 的向量表示首先经过了 tanh
```

<img src="http://47.93.245.14:9000/images/2020/12/16/image-20201216194033832.png" alt="image-20201216194033832" style="zoom:67%;" />