import argparse
import torch
import numpy as np
import cv2
import os
import time

from utils.misc import load_weight
from data.voc0712 import VOCDetection, VOC_CLASSES
from data.coco import COCODataset, coco_class_index, coco_class_labels
from data.transform import BaseTransform

from models.build import build_yolo


parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-d', '--dataset', default='custom',
                    help='voc, coco-val custom.')
parser.add_argument('--root', default='./',
                    help='data root')
# parser.add_argument('-size', '--input_size', default=416, type=int,
#                     help='输入图像尺寸')
parser.add_argument('-size', '--input_size', default=608, type=int,
                    help='输入图像尺寸')

parser.add_argument('-v', '--version', default='yolo',
                    help='yolo')
parser.add_argument('--weight', default='./weights/voc/yolo/yolo_epoch_121_69.1.pth',
                    type=str, help='模型权重的路径')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='得分阈值')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS 阈值')
parser.add_argument('-vs', '--visual_threshold', default=0.3, type=float,
                    help='用于可视化的阈值参数')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('--save', action='store_true', default=True, 
                    help='save vis results.')

args = parser.parse_args()


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, 
              bboxes, 
              scores, 
              labels, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset_name='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(labels[i])
            if dataset_name == 'coco-val':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        

def detect_picture(args, model, device, img_path, transform, class_colors=None, class_names=None, class_indexs=None):
    save_path = os.path.join('det_results/', args.dataset)
    os.makedirs(save_path, exist_ok=True)
    
    # img_path = ""
    print('Testing image: ',img_path)
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    # 预处理图像，并将其转换为tensor类型
    x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
    x = x.unsqueeze(0).to(device)

    t0 = time.time()
    # 前向推理
    bboxes, scores, labels = model(x)
    print("detection time used ", time.time() - t0, "s")
    
    # 将预测的输出映射到原图的尺寸上去
    scale = np.array([[w, h, w, h]])
    bboxes *= scale

    # 可视化检测结果
    img_processed = visualize(
        img=img,
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        vis_thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        dataset_name=args.dataset
        )
    # cv2.imshow('detection', img_processed)
    # cv2.waitKey(0)

    # 保存可视化结果
    if args.save:
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img_processed)

if __name__ == '__main__':
    # 是否使用cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 输入图像的尺寸
    input_size = args.input_size

    class_names = VOC_CLASSES
    class_indexs = None
    num_classes = 20
    
    # 用于可视化，给不同类别的边界框赋予不同的颜色，为了便于区分。
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # 构建模型
    model = build_yolo(args, device, input_size, num_classes, trainable=False)

    # 加载已训练好的模型权重
    model = load_weight(model, args.weight)
    model.to(device).eval()
    print('Finished loading model!')

    val_transform = BaseTransform(input_size)

    folder_path = os.path.join(args.root, 'costom')
    file_list = os.listdir(folder_path)
    img_paths = [f for f in file_list if os.path.isfile(os.path.join(folder_path, f))]

    for img_path in img_paths:
        # 开始测试
        img_path = folder_path = os.path.join(args.root, 'costom', img_path)
        detect_picture(args=args,
            model=model, 
            device=device, 
            img_path=img_path,
            transform=val_transform,
            class_colors=class_colors,
            class_names=class_names,
            class_indexs=class_indexs,
            )
    # # 开始测试
    # test(args=args,
    #      model=model, 
    #      device=device, 
    #      testset=dataset,
    #      transform=val_transform,
    #      class_colors=class_colors,
    #      class_names=class_names,
    #      class_indexs=class_indexs,
    #     )
