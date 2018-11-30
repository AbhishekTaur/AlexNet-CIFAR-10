import torch
import torchvision
from torchvision import transforms as transforms

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar10_classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def cifar10_loader(train=True, batch_size=128):
    set = torchvision.datasets.CIFAR10(root='~/data', train=train,
                                       download=True, transform=transform)
    loader = torch.utils.data.DataLoader(set, batch_size=batch_size,
                                         shuffle=True, num_workers=4)
    return loader

def config_matplotlib():
    import matplotlib
    params = {
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': False,
    }
    matplotlib.rcParams.update(params)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
