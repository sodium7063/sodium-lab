import timm
import torch
from torchvision import transforms
from PIL import Image


def get_vit_model_and_transforms():
    # 加载预训练的ViT模型
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.eval()

    # 定义图像转换过程
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return model, transform


def test_vit():
    # 获取模型和transforms
    model, transform = get_vit_model_and_transforms()

    # 创建一个示例图像 (这里用随机数据代替)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 使用模型进行推理
    with torch.no_grad():
        output = model(dummy_input)

    print("Model output shape:", output.shape)
    print("Number of classes:", output.size(1))

    # 打印模型结构
    print("\nModel Architecture:")
    print(model.__class__.__name__)


if __name__ == "__main__":
    test_vit()
