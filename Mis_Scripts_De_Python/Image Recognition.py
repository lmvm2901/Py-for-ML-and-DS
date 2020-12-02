from torchvision import models
from torchvision import transforms
from PIL import Image
import torch

img = Image.open("C:/Users/Lmvm_/Desktop/Carpeta/Coursera/Py for ML and Data "
                 "Science/Mis_Scripts_De_Python/Imagenes/bobby.jpg")
dir(models)

alexnet = models.AlexNet()

resnet = models.resnet101(pretrained=True)
resnet

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

resnet.eval()
out = resnet(batch_t)

with open("C:/Users/Lmvm_/Desktop/Carpeta/Coursera/Py for ML and Data "
          "Science/Mis_Scripts_De_Python/Txts/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]
_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())


_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
