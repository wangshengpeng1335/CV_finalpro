import torch
import torchvision.transforms as transforms
from PIL import Image
from model import UNET

# 加载模型
model = UNET(in_channels=3, out_channels=1)
checkpoint = torch.load('my_checkpoint.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    # 添加其他必要的预处理步骤
])

# 加载测试图像
test_image = Image.open('test_image.jpg')  # 替换为您自己的测试图像路径
test_image = transform(test_image).unsqueeze(0)

# 运行图像分割
with torch.no_grad():
    output = model(test_image)

# 对输出进行后处理，获取分割结果
segmentation_map = output.argmax(1).squeeze().cpu().numpy()

# 计算IOU
def calculate_iou(pred, target):
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    iou = intersection / union
    return iou

# 加载真实标签图像
ground_truth = Image.open('ground_truth.jpg')  # 替换为您自己的真实标签图像路径
ground_truth = transform(ground_truth).squeeze().numpy()

# 将分割结果和真实标签转为二值化图像
pred_binary = (segmentation_map > 0).astype(int)
gt_binary = (ground_truth > 0).astype(int)

# 计算IOU
iou = calculate_iou(pred_binary, gt_binary)
print("IOU:", iou)
