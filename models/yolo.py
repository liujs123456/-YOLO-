import torch
import torch.nn as nn
import numpy as np

from .basic import Conv, SPP
from backbone import build_resnet

from .loss import compute_loss

"""
    YOLO模型按照以下步骤进行预测：

    1.前向传播：模型接收输入图片，经过backbone（ResNet-18）、neck（SPP模块及1x1卷积层）和detection head（多个卷积层）处理，生成特征图。

    2.预测层：特征图经过最后一个1x1卷积层（self.pred），输出一个张量，形状为 [batch_size, height, width, 1 + num_classes + 4]。
    这里假设batch_size=1，则简化为 [height, width, 1 + num_classes + 4]。

        每个位置的预测：对于特征图上的每一个像素位置（对应原图的一个小区域），模型都会输出一组预测结果。

            1. 第1个通道：物体存在概率（Objectness Score）：第0个通道（索引为0）表示该位置是否存在物体（无论何种类别），值域为(0, 1)，
            通过sigmoid函数归一化至(0, 1)之间，数值越接近1表示模型越确信该位置存在物体。

            2. 第2到1+num_classes个通道：类别概率（Class Scores）：接下来num_classes个通道（索引为1到num_classes）代表该位置可能存在各目标类别的概率分布，
            同样通过sigmoid函数归一化。每个通道对应一个类别，数值越接近1表示模型越确信该位置存在对应类别的物体。

            3. 最后4个通道：边界框偏移量（Bounding Box Regression）：最后4个通道（索引为num_classes+1到num_classes+4）表示该位置对应的边界框相对于网格单元中心的偏移量
            （txtytwth），其中tx和ty表示中心点的偏移，tw和th表示宽高相对于网格单元的缩放比例。这些偏移量未经解码时是相对值，需经过decode_boxes方法转换为绝对坐标。

    3.解码边界框：对预测的txtytwth偏移量进行解码，结合网格坐标矩阵，得到边界框在原图上的绝对坐标（x1y1x2y2形式）。

    4.后处理：

        筛选：对每个位置的预测结果，应用置信度阈值（conf_thresh）筛选，仅保留物体存在概率高于阈值的预测。
        类别分配：为每个保留的预测框，根据类别概率最高的类别分配类别标签。
        非最大抑制（NMS）：对同一类别的多个预测框，根据其重叠程度（IoU）和得分，执行非最大抑制，去除冗余预测，保留每个类别中最具代表性的边界框。
    
    5.输出：经过上述过程，模型针对输入图片中的多个类别对象生成了一系列预测边界框，每个边界框带有相应的类别标签、得分（物体存在概率与类别概率的乘积）以及实际坐标（x1y1x2y2）。这些信息以列表或数组形式返回，通常包含以下内容：

        bboxes：一个二维数组，每一行是一个边界框的坐标（x1, y1, x2, y2）。
        scores：一个一维数组，与bboxes对应，存储每个边界框的得分。
        labels：一个一维数组，与bboxes对应，存储每个边界框所属的类别标签。

"""

# YOLO
class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5):
        super(myYOLO, self).__init__()
        self.device = device                           # cuda或者是cpu
        self.num_classes = num_classes                 # 类别的数量
        self.trainable = trainable                     # 训练的标记
        self.conf_thresh = conf_thresh                 # 得分阈值
        self.nms_thresh = nms_thresh                   # NMS阈值
        self.stride = 32                               # 网络的最大步长
        self.grid_cell = self.create_grid(input_size)  # 网格坐标矩阵
        self.input_size = input_size                   # 输入图像大小
        
        # backbone: resnet18
        # 使用预训练权重作为特征提取器
        self.backbone, feat_dim = build_resnet('resnet18', pretrained=trainable)

        # neck: SPP
        self.neck = nn.Sequential(
            SPP(),
            Conv(feat_dim*4, feat_dim, k=1),
        )

        # detection head
        self.convsets = nn.Sequential(
            Conv(feat_dim, feat_dim//2, k=1),
            Conv(feat_dim//2, feat_dim, k=3, p=1),
            Conv(feat_dim, feat_dim//2, k=1),
            Conv(feat_dim//2, feat_dim, k=3, p=1)
        )

        # pred
        # 一个1x1卷积层，将特征图转换为输出通道数为1 + num_classes + 4的预测结果
        # 其中，第一个通道表示物体存在概率（objectness），接下来num_classes个通道表示各类别概率，最后四个通道表示边界框的偏移量
        self.pred = nn.Conv2d(feat_dim, 1 + self.num_classes + 4, 1)
    

        if self.trainable:
            self.init_bias()


    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :1], bias_value)
        nn.init.constant_(self.pred.bias[..., 1:1+self.num_classes], bias_value)


    def create_grid(self, input_size):
        """ 
            用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
        """
        # 输入图像的宽和高
        w, h = input_size, input_size
        # 特征图的宽和高
        ws, hs = w // self.stride, h // self.stride
        # 生成网格的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

        # 将xy两部分的坐标拼起来：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # [H, W, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)
        
        return grid_xy


    def set_grid(self, input_size):
        """
            用于重置G矩阵。
        """
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)


    def decode_boxes(self, pred):
        """
            将txtytwth转换为常用的x1y1x2y2形式。
            
            功能：
            1.坐标解码：首先将物体存在概率和类别概率通过sigmoid函数转换为概率值。
            接着，将预测的txtytwth偏移量分别通过sigmoid函数（txty）和exp函数（twth）进行解码。
            解码后的txty与网格坐标相加得到边界框中心坐标，中心坐标与twth相乘得到边界框的实际宽度和高度。
            
            2.转换为x1y1x2y2格式：根据解码后的中心坐标和宽高计算出边界框左上角和右下角坐标
            （x1y1x2y2），便于后续处理和可视化。
        """
        output = torch.zeros_like(pred)
        # 得到所有bbox 的中心点坐标和宽高
        pred[..., :2] = torch.sigmoid(pred[..., :2]) + self.grid_cell
        pred[..., 2:] = torch.exp(pred[..., 2:])

        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        output[..., :2] = pred[..., :2] * self.stride - pred[..., 2:] * 0.5
        output[..., 2:] = pred[..., :2] * self.stride + pred[..., 2:] * 0.5
        
        return output


    def nms(self, bboxes, scores):
        """"
            Pure Python NMS baseline.
            
            功能：
            1.计算交并比：对于同一类别的多个预测边界框，计算它们之间的交并比（Intersection over Union, IoU）。
            若IoU大于给定的非最大抑制阈值，说明两个边界框重叠严重，应保留其中一个。
            
            2.保留最优边界框：按照类别分数（通常是物体存在概率与类别概率的乘积）降序排列边界框。
            依次遍历排序后的边界框，对每个框，检查其与其他剩余框的IoU。若IoU大于阈值，则剔除与当前框重叠严重的其他框。
            最终保留下来的边界框即为经过非最大抑制处理的结果。
        """
        x1 = bboxes[:, 0]  #xmin
        y1 = bboxes[:, 1]  #ymin
        x2 = bboxes[:, 2]  #xmax
        y2 = bboxes[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []                                             
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 滤除超过nms阈值的检测框
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        Input:
            bboxes: [HxW, 4]
            scores: [HxW, num_classes]
        Output:
            bboxes: [N, 4]
            score:  [N,]
            labels: [N,]
        """

        labels = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), labels)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels


    @torch.no_grad()
    def inference(self, x):
        # backbone主干网络
        feat = self.backbone(x)

        # neck网络
        feat = self.neck(feat)

        # detection head网络
        feat = self.convsets(feat)

        # 预测层
        pred = self.pred(feat)

        # 对pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
        # [B, H*W, 1]
        conf_pred = pred[..., :1]
        # [B, H*W, num_cls]
        cls_pred = pred[..., 1:1+self.num_classes]
        # [B, H*W, 4]
        txtytwth_pred = pred[..., 1+self.num_classes:]

        # 测试时，默认batch是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        conf_pred = conf_pred[0]            #[H*W, 1]
        cls_pred = cls_pred[0]              #[H*W, NC]
        txtytwth_pred = txtytwth_pred[0]    #[H*W, 4]

        # 每个边界框的得分
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)
        
        # 解算边界框, 并归一化边界框: [H*W, 4]
        bboxes = self.decode_boxes(txtytwth_pred) / self.input_size
        bboxes = torch.clamp(bboxes, 0., 1.)
        
        # 将预测放在cpu处理上，以便进行后处理
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()
        
        # 后处理
        bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels


    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone主干网络
            feat = self.backbone(x)

            # neck网络
            feat = self.neck(feat)

            # detection head网络
            feat = self.convsets(feat)

            # 预测层
            pred = self.pred(feat)

            # 对pred 的size做一些view调整，便于后续的处理
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
            # [B, H*W, 1]
            conf_pred = pred[..., :1]
            # [B, H*W, num_cls]
            cls_pred = pred[..., 1:1+self.num_classes]
            # [B, H*W, 4]
            txtytwth_pred = pred[..., 1+self.num_classes:]

            # 计算损失
            (
                conf_loss,
                cls_loss,
                bbox_loss,
                total_loss
            ) = compute_loss(pred_conf=conf_pred, 
                             pred_cls=cls_pred,
                             pred_txtytwth=txtytwth_pred,
                             targets=targets
                             )

            return conf_loss, cls_loss, bbox_loss, total_loss            
