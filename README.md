# 使用联邦学习统计影响学生学习的相关因素
jahdadwhk
## 董嘉琦
dkjlahsdkjahwdkj
### 高铭伟
jhakdhkajhdlwha





sSAdAsadasd






asd
sadsad



[高铭伟](#高铭伟)
$ python mlfromscratch/examples/convolutional_neural_network.py

+---------+
| ConvNet |
+---------+
Input Shape: (1, 8, 8)
+----------------------+------------+--------------+
| Layer Type           | Parameters | Output Shape |
+----------------------+------------+--------------+
| Conv2D               | 160        | (16, 8, 8)   |
| Activation (ReLU)    | 0          | (16, 8, 8)   |
| Dropout              | 0          | (16, 8, 8)   |
| BatchNormalization   | 2048       | (16, 8, 8)   |
| Conv2D               | 4640       | (32, 8, 8)   |
| Activation (ReLU)    | 0          | (32, 8, 8)   |
| Dropout              | 0          | (32, 8, 8)   |
| BatchNormalization   | 4096       | (32, 8, 8)   |
| Flatten              | 0          | (2048,)      |
| Dense                | 524544     | (256,)       |
| Activation (ReLU)    | 0          | (256,)       |
| Dropout              | 0          | (256,)       |
| BatchNormalization   | 512        | (256,)       |
| Dense                | 2570       | (10,)        |
| Activation (Softmax) | 0          | (10,)        |
+----------------------+------------+--------------+
Total Parameters: 538570

Training: 100% [------------------------------------------------------------------------] Time: 0:01:55
Accuracy: 0.987465181058


Figure: Classification of the digit dataset using CNN.

Density-Based Clustering
$ python mlfromscratch/examples/dbscan.py


Figure: Clustering of the moons dataset using DBSCAN.

Generating Handwritten Digits
$ python mlfromscratch/unsupervised_learning/generative_adversarial_network.py

+-----------+
| Generator |
+-----------+
Input Shape: (100,)
+------------------------+------------+--------------+
| Layer Type             | Parameters | Output Shape |
+------------------------+------------+--------------+
| Dense                  | 25856      | (256,)       |
| Activation (LeakyReLU) | 0          | (256,)       |
| BatchNormalization     | 512        | (256,)       |
| Dense                  | 131584     | (512,)       |
| Activation (LeakyReLU) | 0          | (512,)       |
| BatchNormalization     | 1024       | (512,)       |
| Dense                  | 525312     | (1024,)      |
| Activation (LeakyReLU) | 0          | (1024,)      |
| BatchNormalization     | 2048       | (1024,)      |
| Dense                  | 803600     | (784,)       |
| Activation (TanH)      | 0          | (784,)       |
+------------------------+------------+--------------+
Total Parameters: 1489936

+---------------+
| Discriminator |
+---------------+
Input Shape: (784,)
+------------------------+------------+--------------+
| Layer Type             | Parameters | Output Shape |
+------------------------+------------+--------------+
| Dense                  | 401920     | (512,)       |
| Activation (LeakyReLU) | 0          | (512,)       |
| Dropout                | 0          | (512,)       |
| Dense                  | 131328     | (256,)       |
| Activation (LeakyReLU) | 0          | (256,)       |
| Dropout                | 0          | (256,)       |
| Dense                  | 514        | (2,)         |
| Activation (Softmax)   | 0          | (2,)         |
+------------------------+------------+--------------+
Total Parameters: 533762


Figure: Training progress of a Generative Adversarial Network generating
handwritten 
1

1
