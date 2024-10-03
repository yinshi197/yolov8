# 				YOLOV8目标检测模型训练

**步骤：**

1.收集数据集：使用开源已标记数据集、爬取网络图片、使用数据增强生成数据集

2.标注数据集：常用工具常用的标注工具有很多， 比如 LabelImg 、 LabelMe 、 VIA 等，在线标注数据集的工具 Make Sense 

```
<object-class-id> <x> <y> <width> <height>
```

3.划分数据集

原始数据集

```
Moon_Cake
	├─images
	   └─all
	└─labels
	   └─all

```

```
├── yolov8_dataset
	└── train #训练集
		└── images (folder including all training images)
		└── labels (folder including all training labels)
	└── test #测试集
		└── images (folder including all testing images)
		└── labels (folder including all testing labels)
	└── valid #验证集
		└── images (folder including all testing images)
		└── labels (folder including all testing labels)
```

4.配置训练环境

```
train: /path/to/images/train
val: /path/to/images/val
test: /path/to/images/test  # 可选

nc: 10  # 类别数
names: ['class1', 'class2', ...]  # 类别名称
```

5.训练模型

```
yolo detect train data=datasets/Stundet.yaml model=yolov8n.pt epochs=100 imgsz=640
```
