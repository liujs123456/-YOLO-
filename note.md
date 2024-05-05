# 配置环境

1. 创建虚拟环境：conda create -n yolo python=3.6

2. 激活目标环境：conda activate yolo

3. 安装所有依赖包：pip install -r requirements.txt 

# 文件说明

1. train.py：使用数据集训练模型，默认使用CPU进行测试，会耗费大量时间，如果使用GPU训练需要安装pytorch的gpu版本

2. test.py：在数据集上对模型进行测试，默认使用CPU进行测试

3. test_costom.py：检测costom文件夹中的自定义图片，并保存到det_results/custom文件夹中,。
    需要检测图片时，只需要把图片放入到costom文件夹中，运行test_costom.py即可

4. models/yolo.py：定义了YOLO网络结构，需要重点学习，知道大概的模型结果以及预测结果是什么

5. weight文件下存放训练好的模型参数，可以直接使用训练好的./weights/voc/yolo/yolo_69.6.pth模型进行预测


# 训练数据集准备

1. 模型使用PASCAL VOC数据集进行训练，使用命令下载（或者复制网址到浏览器下载）
    curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

2. 数据集解压后生成VOCdevkit文件夹，并放在当前项目目录下


