import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import MaxPool2d, Conv2d, Linear, Flatten
from model import NN_CIFAR10
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# 訓練設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 準備數據
train_data = torchvision.datasets.CIFAR10(root = "dataset", train = True, transform = torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root = "dataset", train = False, transform = torchvision.transforms.ToTensor())

# 長度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("length of training data: {}".format(train_data_size))
print("length of testing data: {}".format(test_data_size))

# 數據加載
train_dataloader = DataLoader(train_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

# 搭建網路
class NN_CIFAR10(nn.Module):
    def __init__(self):
        super(NN_CIFAR10, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, padding = 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding = 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding = 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        output = self.model(x)
        return output
nn_test = NN_CIFAR10()
nn_test = nn_test.to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# optimizer
learning_rate = 1e-2
optim = torch.optim.SGD(nn_test.parameters(), lr = learning_rate)
scheduler = StepLR(optim, step_size=10, gamma=0.5)

# 總訓練數
total_train_step = 0
# 總測試數
total_test_step = 0
# 輪數
epoch = 40
# tensorboard
writer = SummaryWriter("CIFAR10_logs")

# 訓練
# nn_test.train()
for i in range(epoch):
    print("-----------round: {}".format(i + 1))
    
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = nn_test(imgs)
        loss = loss_fn(output, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1

        # loss(train)輸出
        if total_train_step % 100 == 0:     
            print("train time: {}, loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 學習率輸出
    print("learning rate: {}".format(optim.param_groups[0]['lr']))
    scheduler.step()

    # 測試
    # nn_test.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = nn_test(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            total_accuracy += (output.argmax(1) == targets).sum()

    # loss(test)輸出
    print("total test loss: {}".format(total_test_loss))
    print("total accuracy: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accurcy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 保存
    # torch.save(nn_test, "./model/CIFAR10_train{}.pth".format(i))
    # print("save successful")
writer.close()