
# 常见指令

1.  dir()：可以返回内部的变量和函数
2.  help()：可以返回使用帮助
3.  torch.flatten()：用来把多维数据转化成一维的

# jupyter使用技巧

1.  和help类似，Dataset??就可以返回组织形式更好的help

# 数据pipline

1.  tensor:基本数据单元，可以理解为向量，

    可以直接通过torch.tensor(\[])来创建，或者读取一张图片就是一个三通道的tensor

    通过tensor.shape可以得到一个数组来表示tensor的维度，这个数组的长度是不一定的，即使原数据是一个二维矩阵，也可以通过reshape(data,(-1,5,5))来得到\[1,5,5]，填-1表示会根据其他数据自动计算
    最常用是四个数，分别表示batch size，channel，width，height 
2.  Dataset（数据集）：获取数据，形成数据，id，label的组合形式

    torchvision里面有很多标准数据集，具体内容可以在pytorch官网查看，在代码里可以直接下载

    ```python
    import torchvision
    from torch.utils.tensorboard import SummaryWriter

    dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root="./dataset", train = True, transform=dataset_transform, download = True)
    test_set = torchvision.datasets.CIFAR10(root="./dataset", train = False, transform=dataset_transform, download = True)

    writer = SummaryWriter("p10")
    for i in range(10):
        img, target = test_set[i]
        writer.add_image("test_set", img, i)

    writer.close()原始数据的形式
    ```

    *   文件夹名称是label，文件夹内都是对应的图片
    *   图片和label分别在两个文件夹内，对应的图片和label的名字相同，适用于label比较复杂的情况
    *   直接把label当作图片文件名
3.  Dataloader：组合数据，传入后续的网络
    ```python
    import torchvision
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root="./dataset", train = False, transform=dataset_transform)

    # loader的目的是把多张图片的img和target分别打包，比如batch_size=16表示，拿到16张图片，把他们打包，同时把他们的target也打包
    test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    writer = SummaryWriter("dataloader")
    for epoch in range(2): # 验证shuffle的功能实现
        i = 0
        for data in test_loader:
            imgs, targets = data
            writer.add_images("epoch = {}".format(epoch), imgs, i)
            i = i + 1
    writer.close()

    ```

# Tensorboard

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

img_path = "C:\\Users\djh19\\Desktop\\test_project\\hymenoptera_data\\train\\ants\\0013035.jpg"
img = Image.open(img_path)
# 传入image可以有多种格式，np.array只是其中一种
# 注意需要满足（3，H，W）的格式需求，如果是（H,W,3）的话需要加 dataformats = "HWC"
img_array = np.array(img)
writer.add_image("test", img_array, 1, dataformats="HWC") # 可以把1改成不同的数字，可以在一个框里出现多个图片

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()
```

运行完成之后，输入tensorboard --logdir=logs --port=6006，可以得到一个网址（包含port，可自己指定，以防和别人冲突）来显示结果，这个网页不需要关闭，放程序修改过后，在页面里点击刷新即可

# Transform

提供了很多工具来对图片进行处理

1.  常见工具

    *   ToTensor

        ```python
        from torchvision import transforms
        from torch.utils.tensorboard import SummaryWriter
        from PIL import Image
        writer = SummaryWriter("logs")
        img_path = "C:\\Users\djh19\\Desktop\\test_project\\hymenoptera_data\\train\\ants\\0013035.jpg"
        img = Image.open(img_path)
        tensor_trans = transforms.ToTensor()
        tensor_img = tensor_trans(img) # 返回一个tensor类型的图片
        writer.add_image("Tensor_img", tensor_img)
        writer.close()
        ```
2.  Normalize
3.  Resize
4.  Compose
5.  RandomCrop


# [神经网络搭建](https://pytorch.org/docs/stable/index.html)

## 神经网络结构

*   [Containers](https://pytorch.org/docs/stable/nn.html#containers)
*   [Convolution Layers](https://pytorch.org/docs/stable/nn.html#convolution-layers)
    *   &#x20;  [卷积](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

        一张图片可以理解为二维的方阵（input，5x5），设定一个卷积核（kernel，3x3），卷积核在方阵上移动，对应位的数字相乘求和，得到一个数，总之卷积可以把原始的数据量变小

        其实卷积核是定义了一种特征，然后通过卷积可以将图像中符合这种特征的部分提取出来

        参数解释

        *   in\_channel 和 out\_channel：这两个的比值就是卷积核的数量，卷积核的数量表示对同一个输入会操作几次
        *   kernel\_size：卷积核
        *   bias：卷积后的结果加减一个常数，一般为true
        *   stride：纵向或者横向移动的步长
        *   padding：原始数据外围扩张一圈
        *   dilation：空洞卷积
        *   groups：一般为1
*   [Pooling layers](https://pytorch.org/docs/stable/nn.html#pooling-layers)
    *   概念

        分为pool（下采样）和unpool（上采样）
    *   MaxPool：最终的效果是模糊化图像，来缩小数据量。比如kernel\_size是3x3，那就会在原数据的3x3的范围内提取最大值。然后移动的步数是等于kernel\_size的，也就是说池化的范围是不重叠的。如果最后kernel超出了原数据的范围，就需要用ceil mode来决定是都保留这部分数据，默认是放弃，true的话是保留
*   [Padding Layers](https://pytorch.org/docs/stable/nn.html#padding-layers)

    对输入数据进行填充，但几乎用不到
*   [Non-linear Activations (weighted sum, nonlinearity)](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

    非线性激活函数，比较有名的比如像relu，主要是来滤掉一些数据，再像sigmoid是用来将大范围的数据压缩到\[0,1]之间

    需要注意一个inplace参数，true的话表示直接改变原数据，false的话需要用一个新的量去接收
*   [Non-linear Activations (other)](https://pytorch.org/docs/stable/nn.html#non-linear-activations-other)
*   [Normalization Layers](https://pytorch.org/docs/stable/nn.html#normalization-layers)

    正则化层，对输入采用正则化，能够加快网络的训练速度
*   [Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers)

    一些具体的网络层，比如LSTM等
*   [Transformer Layers](https://pytorch.org/docs/stable/nn.html#transformer-layers)

    就是chatgpt那种transformer
*   [Linear Layers](https://pytorch.org/docs/stable/nn.html#linear-layers)

    对输入进行y=kx+b的处理
*   [Dropout Layers](https://pytorch.org/docs/stable/nn.html#dropout-layers)
*   [Sparse Layers](https://pytorch.org/docs/stable/nn.html#sparse-layers)

    也是一些成熟的特定网络层
*   [Distance Functions](https://pytorch.org/docs/stable/nn.html#distance-functions)
*   [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
*   [Vision Layers](https://pytorch.org/docs/stable/nn.html#vision-layers)
*   [Shuffle Layers](https://pytorch.org/docs/stable/nn.html#shuffle-layers)
*   [DataParallel Layers (multi-GPU, distributed)](https://pytorch.org/docs/stable/nn.html#module-torch.nn.parallel)
*   [Utilities](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils)
*   [Quantized Functions](https://pytorch.org/docs/stable/nn.html#quantized-functions)
*   [Lazy Modules Initialization](https://pytorch.org/docs/stable/nn.html#lazy-modules-initialization)

## 神经网络实例
### Cifar10 model structure
![cifar10 model structure](https://www.researchgate.net/profile/Yiren-Zhou-6/publication/312170477/figure/fig2/AS:448817725218817@1484017892180/Structure-of-CIFAR10-quick-model.png#pic_center)
首先原图的尺寸是3x32x32，现在想要变成32x32x32，就需要通过下面的公式来计算得到需要的padding和stride
![输入图片描述](Pytorch%E4%BD%BF%E7%94%A8_md_files/25c719b0-c008-11ed-81d7-993a58520a16_20230311202753.jpeg?v=1&type=image&token=V1:lzgBva3-9q9JaHUPHlvIxpCp4Eg4wT1MbCeZnKcEH2s)
中间就按照卷积和池化来设计就行，最后三个构成了一个全连接层，使用两个线性层来实现的
利用sequential可以简化代码
利用summaryWriter可以看到网络的结构，如下
![输入图片描述](Pytorch%E4%BD%BF%E7%94%A8_md_files/1972ca80-c00b-11ed-81d7-993a58520a16_20230311204901.jpeg?v=1&type=image&token=V1:_O7XSb7oAXtvknvD6TIL8xbb_XHB9Pny2XoH6bOiw7I)
###  [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions) 和 [Optimizer](https://pytorch.org/docs/stable/optim.html)
前面搭建的神经网络只能计算一次输入到输出，但这时的输出并不一定准确，这时候需要计算输出和期望之间的差距，就要用到loss函数，得到loss之后，利用反向传播算法，会得到一个梯度，接下来可以利用不同的优化器去迭代优化这个loss（会去更新各个层里面的参数），这个过程也叫训练
```  
import torch  
import torchvision  
from torch import nn  
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear  
from torch.optim.lr_scheduler import StepLR  
from torch.utils.data import DataLoader  
  
dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),  
                                       download=True)  
  
dataloader = DataLoader(dataset, batch_size=1)  
  
class Tudui(nn.Module):  
    def __init__(self):  
        super(Tudui, self).__init__()  
        self.model1 = Sequential(  
            Conv2d(3, 32, 5, padding=2),  
            MaxPool2d(2),  
            Conv2d(32, 32, 5, padding=2),  
            MaxPool2d(2),  
            Conv2d(32, 64, 5, padding=2),  
            MaxPool2d(2),  
            Flatten(),  
            Linear(1024, 64),  
            Linear(64, 10)  
        )  
  
    def forward(self, x):  
        x = self.model1(x)  
        return x  

loss = nn.CrossEntropyLoss()  
tudui = Tudui()  
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)  
for epoch in range(20):  
    running_loss = 0.0  
  for data in dataloader:  
        imgs, targets = data  
        outputs = tudui(imgs)  
        result_loss = loss(outputs, targets)  
        optim.zero_grad()  
        result_loss.backward()  
        optim.step()  
        running_loss = running_loss + result_loss  
    print(running_loss)
```
## 现有神经网络
1. 使用模型
 `torchvision.models.vgg16(*, weights: Optional[VGG16_Weights] = None, progress: bool = True, **kwargs: Any)`
其中`Optional[VGG16_Weights]`可以用来设定预训练的程度
2. 修改模型
- 添加网络层
`vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))`
- 修改网络层
`vgg16_false.classifier[6] = nn.Linear(4096, 10)`
- 模型保存和加载
	 ```
	torch.save(vgg16, "vgg16_method1.pth") # 保存方式1,模型结构+模型参数
	model = torch.load("vgg16_method1.pth") 
	 
	vgg16 = torchvision.models.vgg16(pretrained=False) # 保存方式2，模型参数（官方推荐）
	vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
	torch.save(vgg16.state_dict(), "vgg16_method2.pth") 
	 ``` 
## 完整的训练
train.py(C:\Users\djh19\Desktop\pytorch-tutorial\src\train.py)
- [谷歌gpu训练](https://colab.research.google.com/)
需要在修改-笔记本设置-硬件加速，修改成GPU，还可以使用TPU
### 使用gpu训练
- 方法一
只要把模型，数据和loss function调用cuda即可
train_gpu_1.py(C:\Users\djh19\Desktop\pytorch-tutorial\src\train_gpu_1.py)
- 方法二
	train_gpu_2.py(C:\Users\djh19\Desktop\pytorch-tutorial\src\train_gpu_2.py)
	```
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 保证安全
	module = module.to(device)
	```
	如果有多个GPU，可以用`torch.device("cuda:id")`来选择GPU




