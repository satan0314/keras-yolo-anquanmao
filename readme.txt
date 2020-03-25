实现步骤：
1.	图像采集（取到图片）：录取视频，然后由命令将视频做成图片，并对图片进行筛选，并用LableImg标注生成xml文件。
2.	生成txt文件：分别运行make_txt.py和voc_annotation.py数据集制作完成
3.	修改参数文件yolo3.cfg：根据自己的数据集分类数目，修改filters和classes
4.	权重文件：下载权重文件，并转为keras适用的h5文件。
5.	训练：修改train.py，训练模型，生成h5文件。
6.	检测：为了可用于视频，编写yolo_camera.py 用于现场摄像头识别
