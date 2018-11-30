import torch
from torch.nn.functional import softmax

from alexnet import AlexNet
from utils import cifar10_loader, device, cifar10_classes

torch.random.manual_seed(128)
batch_size = 1
testloader = cifar10_loader(train=False, batch_size=batch_size)

net = AlexNet()
net.load_state_dict(torch.load("model.h5"))
net.eval()


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        inputs, labels = images.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.topk(outputs.data, 5)
        print(predicted)
        indexes = predicted.numpy()[0].tolist()
        print(indexes)
        print(softmax(outputs).numpy()[0][indexes])
        print([cifar10_classes[i] for i in indexes])
