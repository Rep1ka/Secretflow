#!/usr/bin/env python
# coding: utf-8

# # 基于隐语Secretflow框架的水平联邦学习落地实践——学生在校表现预测的多方联合训练
# ## 本地模拟测试代码

#     该文档包含一个本地模拟流程，从一个.csv文件开始，模拟了双方联合训练的全流程。

# ### 数据拆分及预处理

# In[1]:


import tempfile
import pandas as pd
import secretflow as sf

alldata_df = pd.read_csv("./student-mat.csv")
h_alice_df = alldata_df.loc[:100]
h_bob_df = alldata_df.loc[100:200]
h_test_df=alldata_df.loc[200:]

_, h_alice_path = tempfile.mkstemp()
_, h_bob_path = tempfile.mkstemp()
_,h_test_path=tempfile.mkstemp()
h_alice_df.to_csv("./A.csv", index=False)
h_bob_df.to_csv("./B.csv", index=False)
h_test_df.to_csv("./test.csv", index=False)


# ### 基于pytorch后端的FLModel联邦学习模拟

#     首先经过隐语框架将两份数据分别加载到Frame中（这个过程中双方的数据没有出域）。
#     我们要预测的目标特征为G3，即学生的最终测试成绩。我们将数据分为G3与其他特征，用于在后续模型中进行训练。

# In[9]:


# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address="local", log_to_driver=True)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

from secretflow.data.horizontal import read_csv
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from secretflow.data.split import train_test_split

path_dict = {alice: "./A.csv", bob: "./B.csv"}

aggregator = PlainAggregator(charlie)
comparator = PlainComparator(charlie)

hdf = read_csv(filepath=path_dict, aggregator=aggregator, comparator=comparator)
train_label = hdf["G3"]
train_data = hdf.drop(columns="G3")

testframe = pd.read_csv("./test.csv")
test_label = testframe["G3"]
test_data = testframe.drop(columns="G3")


#     接下来开始训练

# In[13]:


from secretflow.ml.nn.core.torch import (
    metric_wrapper,
    optim_wrapper,
    BaseModule,
    TorchModel,
)
from secretflow.ml.nn import FLModel
from torchmetrics import Accuracy, Precision
from secretflow.security.aggregation import SecureAggregator
#from secretflow.utils.simulation.datasets import load_mnist
from torch import nn, optim
from torch.nn import functional as F

class ConvNet(BaseModule):
    """Small ConvNet for MNIST."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        self.fc = nn.Linear(self.fc_in_dim, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.fc_in_dim)
        x = self.fc(x)
        return F.softmax(x, dim=1)

class SimpleNN(BaseModule):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(31, 64)  # 输入特征数31，隐藏层神经元数64（可以调整）
        self.fc2 = nn.Linear(64, 32)  # 隐藏层神经元数64，下一层神经元数32（可以调整）
        self.fc3 = nn.Linear(32, 10)  # 最后一层，假设有10个类别

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一个全连接层及激活函数
        x = F.relu(self.fc2(x))  # 第二个全连接层及激活函数
        x = self.fc3(x)          # 最后一层输出 logits
        return F.softmax(x, dim=1)  # 输出概率分布

loss_fn = nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
model_def = TorchModel(
    model_fn=SimpleNN,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(Accuracy, task="multiclass", num_classes=10, average='micro'),
        metric_wrapper(Precision, task="multiclass", num_classes=10, average='micro'),
    ],
)


# In[14]:


device_list = [alice, bob]
server = charlie
aggregator = SecureAggregator(server, [alice, bob])

# spcify params
fl_model = FLModel(
    server=server,
    device_list=device_list,
    model=model_def,
    aggregator=aggregator,
    strategy='fed_avg_w',  # fl strategy
    backend="torch",  # backend support ['tensorflow', 'torch']
)


# In[15]:


history = fl_model.fit(
    train_data,
    train_label,
    validation_data=(test_data, test_label),
    epochs=20,
    batch_size=16,
    aggregate_freq=1,
)


# In[ ]:


from matplotlib import pyplot as plt

# Draw accuracy values for training & validation
plt.plot(history["global_history"]['multiclassaccuracy'])
plt.plot(history["global_history"]['val_multiclassaccuracy'])
plt.title('FLModel accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

