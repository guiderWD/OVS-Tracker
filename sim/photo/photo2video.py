import cv2
import os

# 视频参数
fps = 15
output_video = 'agents_animation.avi'
image_folder = '10agent'  # 存放图片的文件夹
image_files = []

# 收集所有图像文件
for i in range(1, 377):  # 001 到 376
    filename = f'agents_count_{i:03d}.png'
    image_files.append(os.path.join(image_folder, filename))

# 读取第一张图像以获取尺寸
first_image = cv2.imread(image_files[0])
height, width, layers = first_image.shape

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 遍历所有图像并写入视频
for image_file in image_files:
    image = cv2.imread(image_file)
    video_writer.write(image)

# 释放视频写入对象
video_writer.release()
print(f'视频已成功创建: {output_video}')
