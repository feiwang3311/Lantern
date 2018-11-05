import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.utils.model_zoo as model_zoo
import inputs
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
  'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
  'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

  def __init__(self, inplanes, squeeze_planes,
         expand1x1_planes, expand3x3_planes):
    super(Fire, self).__init__()
    self.inplanes = inplanes
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    self.squeeze_activation = nn.ReLU(inplace=True)
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                   kernel_size=1)
    self.expand1x1_activation = nn.ReLU(inplace=True)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                   kernel_size=3, padding=1)
    self.expand3x3_activation = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.squeeze_activation(self.squeeze(x))
    return torch.cat([
      self.expand1x1_activation(self.expand1x1(x)),
      self.expand3x3_activation(self.expand3x3(x))
    ], 1)


class SqueezeNet(nn.Module):

  def __init__(self, num_classes=10):
    super(SqueezeNet, self).__init__()
    self.num_classes = num_classes
    self.firstConv = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      Fire(96, 16, 64, 64),
      Fire(128, 16, 64, 64),
      Fire(128, 32, 128, 128),
      nn.MaxPool2d(kernel_size=2, stride=2),
      Fire(256, 32, 128, 128),
      Fire(256, 48, 192, 192),
      Fire(384, 48, 192, 192),
      Fire(384, 64, 256, 256),
      nn.MaxPool2d(kernel_size=2, stride=2),
      Fire(512, 64, 256, 256),
    )
    # Final convolution is initialized differently form the rest
    self.final_conv = nn.Conv2d(512, self.num_classes, kernel_size=4)
    # self.classifier = nn.Sequential(
    #   # nn.Dropout(p=0.5),
    #   final_conv,
    #   # nn.ReLU(inplace=True),
    #   # nn.AvgPool2d(4, stride=1)
    # )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if m is self.final_conv:
          weight_init.normal(m.weight, mean=0.0, std=0.01)
        else:
          weight_init.kaiming_uniform(m.weight)
        if m.bias is not None:
          weight_init.constant(m.bias, 0)

  def forward(self, x):
    before = time.time()
    x = self.firstConv(x)
    # after = time.time()
    x = self.features(x)
    x = self.final_conv(x)
    return x.view(x.size(0), self.num_classes), before

class Test(nn.Module):
  def __init__(self, num_classes = 10):
    super(Test, self).__init__()
    self.num_classes = num_classes
    self.features = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
    )
    self.fire1_squeeze = nn.Conv2d(96, 16, kernel_size = 1, stride = 1)
    self.squeeze_activation = nn.ReLU(inplace=True)
    self.fire1_expand1 = nn.Conv2d(16, 64, kernel_size = 1, stride = 1)
    self.expand1x1_activation = nn.ReLU(inplace=True)
    self.fire1_expand2 = nn.Conv2d(16, 64, kernel_size = 3, stride = 1, padding = 1)
    self.expand3x3_activation = nn.ReLU(inplace=True)
    # self.fire1 = Fire(96, 16, 64, 64)
    self.fire2 = Fire(128, 16, 64, 64)
    self.fire3 = Fire(128, 32, 128, 128)
    torch.manual_seed(42)
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        weight_init.kaiming_uniform(m.weight)
        if m.bias is not None:
          weight_init.constant(m.bias, 0)
  def forward(self, x):
    printHead(10, x, "input")
    x = self.features(x)
    printHead(10, x, "after conv1")
    # x = self.fire1(x)
    x = self.squeeze_activation(self.fire1_squeeze(x))
    printHead(10, x, "after fire1_squeeze")
    x1 = self.expand1x1_activation(self.fire1_expand1(x))
    printHead(10, x1, "after fire1_expand1")
    x2 = self.expand3x3_activation(self.fire1_expand2(x))
    printHead(10, x2, "after fire1_expand2")
    x = torch.cat([x1, x2], 1)
    printHead(10, x, "after fire1")
    # print(x.max(), x.min(), x.mean(), x.median())
    x = self.fire2(x)
    printHead(10, x, "after fire2")
    # print(x.max(), x.min(), x.mean(), x.median())
    x = self.fire3(x)
    printHead(10, x, "after fire3")
    # print(x.max(), x.min(), x.mean(), x.median())

def printHead(n, tensor, name):
  print(name, tensor.shape)
  tensor = tensor.view(tensor.numel())
  smin = float(tensor.min().data)
  smax = float(tensor.max().data)
  if abs(smin) < abs(smax):
    print("Max Abs: {0:0.5f}".format(smax), end = " || ")
  else:
    print("Max Abs: {0:0.5f}".format(smin), end = " || ")
  for i in range(n):
    print("{0:0.5f}".format(float(tensor[i].data)), end=" ")
  print()

def printHead2(n, tensor, name):
  print(name, tensor.shape)
  tensor = tensor.reshape(-1)
  amax = 0
  for i in range(tensor.shape[0]):
    if abs(tensor[i]) > amax:
      amax = tensor[i]
  print("{0:0.5f}".format(amax), end = " || ")
  for i in range(n):
    print("{0:0.5f}".format(tensor[i]), end=" ")
  print("\n")

if __name__ == '__main__':
  torch.manual_seed(42)
  model = SqueezeNet()
  optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0)
  batch = inputs.Batch('../cifar10_data/cifar-10-batches-py/data_batch_1', 64)

  tloss = 0.0
  for i in range(batch.total_size // batch.batch_size):
    (input_x, input_y) = batch.batch()
    optimizer.zero_grad()
    res = model(Variable(torch.from_numpy(input_x)))
    loss = F.nll_loss(F.log_softmax(res, dim=1), Variable(torch.from_numpy(input_y)))
    tloss += loss.data[0]
    loss.backward()
    optimizer.step()
    if (i + 1) % (batch.total_size // batch.batch_size // 10) == 0:
      print('epoch %d: step %d, training loss %f' % (1, i + 1, tloss / (i)))

  exit(0)

  (input_x, input_y) = batch.batch()
  # printHead2(10, input_x, "input")
  # printHead(10, model.features[0].weight, "conv1 kernel")
  res = model(Variable(torch.from_numpy(input_x)))
  # printHead(10, res, "result")
  loss = F.nll_loss(F.log_softmax(res, dim=1), Variable(torch.from_numpy(input_y)))
  print("loss is {}".format(loss.data.item() * 64))
  loss.backward()
  # for pp in model.parameters():
  #   printHead(10, pp.grad, "unknown")
  optimizer.step()
  for pp in model.parameters():
    printHead(10, pp.data, "unknown")
