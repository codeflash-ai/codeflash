from PIL import Image
import urllib
from torchvision import transforms
import torch

from codeflash.tracer import Tracer
from codeflash.model import AlexNet

def load_model():
    model = AlexNet()
    model.load_state_dict(torch.load('alexnet-owt-7be5be79.pth'))
    model.eval()
    return model

def download_image():
    url, filename = ('https://github.com/pytorch/hub/raw/master/images/dog.jpg', 'dog.jpg')
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

download_image()
model = load_model()
input_image = Image.open('dog.jpg')
preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
    print('YESS')
with Tracer():
    with torch.no_grad():
        output = model(input_batch)
print(output[0].shape)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities.shape)