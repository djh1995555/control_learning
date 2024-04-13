
1.  安装Anaconda3，记住要勾选添加路径到环境变量
2.  创建虚拟环境  
    可以通过anaconda prompt打开命令行，创建环节conda create -n pytorch python=3.6
3.  安装pytorch
	-   添加清华源  
	    conda config --add channels  [https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/)  
	    conda config --add channels  [https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/)  
	    conda config --set show_channel_urls yes  
	    conda config --add channels  [https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch)
	-   安装pytorch  
	    可以去官网安装，选择对应的配置即可，如下  
	    `conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`  
	    其中-c的部分要去掉，这表示从官网下载，不过有时候源里的资源更新不及时，会出现found conflict
        也可以尝试官方的wheel资源
        `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`  
        还不行的话，可以尝试旧的安装命令如下  
	    `conda install pytorch torchvision torchaudio cudatoolkit=11.3`
	-   验证  
	    在对应环境下进入python  
	    返回true表示安装成功了  
	    `import torch`  
	    `torch.cuda.is_available()`

4.  安装notebook  
- 方法1：
    在pytorch环境下安装jupyter notebook  
    本来anaconda装好之后是自带jupyter notebook的，但是只在base环境下，所以需要在pytorch环境下再安装一遍
	-   命令行安装  
	    `conda install nb_conda`
	-   anaconda安装  
	    可以直接在anaconda navigatior里面选择对应的环境，直接安装，更不容易出错
	-   常见问题  
	    如果出现AttributeError: type object 'IOLoop' has no attribute 'initialized'，是tornado的版本问题
		```
		conda uninstall nb_conda # 卸载jupyter notebook
		conda uninstall tornado # 卸载tornado
		conda install tornado=4.5 # 安装tornado
		conda install nb_conda # 安装jupyter notebook
		```
- 方法2：
    方法1就是给每个虚拟环境装一个jupyter，这样从对应的环境中打开jupyter就会进入对应的环境，但是不太优雅。方法2就是用自带的jupyter，每个虚拟环境安装一个ipykernel，用自带的jupyter可以在新建文件时选择对应的ipykernel，也就可以使用对应的虚拟环境，更简洁
    ```
    conda create -n my-conda-env    # creates new virtual env
    conda activate my-conda-env     # activate environment in terminal
    conda install ipykernel      # install Python kernel in new conda env
    ipython kernel install --user --name=my-conda-env-kernel  # configure Jupyter to use Python kernel
    jupyter notebook      # run jupyter from system
    ```
# Python安装方法
1. pip安装
`pip install module`
2. pip自身操作
	- pip 自身的升级
	`python -m pip install --upgrade pip`
	- pip安装/卸载/升级
	`pip install 包名 #安装`
	`pip uninstall 包名 #卸载`
	`pip install --upgrade 包名 #升级`
	- pip查看已安装的包
	`pip list`
	- pip检查哪些包需要更新：
	`pip list --outdated`
	- pip查看某个包的详细信息：
	`pip show 包名`
	- pip安装指定版本的包：
	`pip install 包名 ==版本号`
	`pip install numpy ==1.20.3`
	`pip install 'matplotlib>3.4'`
	`pip install 'matplotlib>3.4.0,<3.4.3' #可通过使用==, >=, <=, >, <来指定版本号`
3. pip安装慢
	指定源下载
	`python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/`
	`pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple/`
	其他源
	- 阿里云：https://mirrors.aliyun.com/pypi/simple/
	- 清华：https://pypi.tuna.tsinghua.edu.cn/simple
	- 豆瓣：https://pypi.douban.com/simple/

