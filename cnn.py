import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas
from torch.utils.data import Dataset
import os


# Custom Label
label_mapping = {
    0: 0,  # T-shirt/top -> Upper
    1: 1,  # Trouser -> Lower
    2: 1,  # Pullover -> Lower
    3: 0,  # Dress -> Upper
    4: 0,  # Coat -> Upper
    5: 2,  # Sandal -> Feet
    6: 0,  # Shirt -> Upper
    7: 2,  # Sneaker -> Feet
    8: 3,  # Bag -> Bag
    9: 2,  # Ankle boot -> Feet
}

Word_mapping = {
    0: "Upper",  # 4000
    1: "Lower",  # 2000
    3: "Bag",  # 1000
    2: "Feet"  # 3000
}

Num_mapping = {
    0: 4000,
    1: 2000,
    3: 1000,
    2: 3000
}


def CalSpace(msg):
    return 14 - len(msg)


def MakeSpace(msg):
    req = CalSpace(msg)
    res = ""

    for _ in range(req):
        res = res + " "

    return res


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def ViewData(data, prefix=""):
    x, y = next(iter(data))
    print("{1} Feature Shape {0}".format(x.shape, prefix))
    print("{1} Label Shape {0}".format(y.shape, prefix))


class NewTrainDataset(Dataset):
    def __init__(self, train_dataframe_feature, train_dataframe_label):
        self.dataf = train_dataframe_feature
        self.datal = train_dataframe_label

    def __getitem__(self, index):
        return self.dataf[index], label_mapping[self.datal[index]]

    def __len__(self):
        return len(self.datal)


class NewTestDataset(Dataset):
    def __init__(self, test_dataframe_feature, test_dataframe_label):
        self.dataf = test_dataframe_feature
        self.datal = test_dataframe_label

    def __getitem__(self, index):
        return self.dataf[index], label_mapping[self.datal[index]]

    def __len__(self):
        return len(self.datal)


class ConvNet(nn.Module):
    def __init__(self, hidden1, hidden2, num_out):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3)
        self.mxpl1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=4)
        self.mxpl2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(
            in_features=250, out_features=hidden1, bias=True)
        self.linear2 = nn.Linear(
            in_features=hidden1, out_features=hidden2, bias=True)
        self.linear3 = nn.Linear(
            in_features=hidden2, out_features=num_out, bias=False)

    def forward(self, data):
        out = nn.Tanh()(self.mxpl1(self.conv1(data)))
        out = nn.Tanh()(self.mxpl2(self.conv2(out)))

        out = out.view(-1, 250)

        out = nn.ReLU()(self.linear1(out))
        out = nn.ReLU()(self.linear2(out))
        out = self.linear3(out)  # RelU not Req

        return out

# Layer conv => maxpool > activation(TanH) > conv => maxpool => activation(TanH) => Linear => activation (RelU)
# => Linear => activation (RelU) => Linear => activation (Not req , Cross Entropy takes care , SoftMax)


if __name__ == "__main__":

    train_data = datasets.FashionMNIST(
        root='data',
        download=True,
        train=True,
        transform=ToTensor()
    )

    print("Train Dataset Loaded")

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=False,
        transform=ToTensor()
    )

    print("Test Datset Loaded")

    train_dataframe_feature = pandas.DataFrame(
        data=(torch.flatten(train_data.data, start_dim=1)).numpy())
    print("Train Dataset Feature Shape {0}".format(
        train_dataframe_feature.shape))

    test_dataframe_feature = pandas.DataFrame(
        data=torch.flatten(input=test_data.data, start_dim=1).numpy())
    print("Test Dataframe Feature Shape {0}".format(
        test_dataframe_feature.shape))

    train_dataframe_label = pandas.DataFrame(data=train_data.targets.numpy())
    print("Train Dataframe Label Shape {0}".format(
        train_dataframe_label.shape))

    test_dataframe_label = pandas.DataFrame(data=test_data.targets.numpy())
    print("Test Dataframe Label Shape {0}".format(test_dataframe_label.shape))

    print("Target Labels {0}".format(train_data.classes))

    print("Creating new datasets")

    new_train_dataset = NewTrainDataset(train_dataframe_feature=train_data.data.numpy(
    ), train_dataframe_label=train_data.targets.numpy())
    new_test_dataset = NewTestDataset(test_dataframe_feature=test_data.data.numpy(
    ), test_dataframe_label=test_data.targets.numpy())

    batch_size = 200
    learning_rate = 1e-2
    conv_input_size = 125
    conv_hidden_size_1 = 512
    conv_hidden_size_2 = 512
    num_class_out = 4
    epoches = 9

    FILE_PATH = "./model/label10_model.pth"

    new_train_dataset_dataloader = DataLoader(
        dataset=new_train_dataset, num_workers=2, shuffle=True, batch_size=batch_size)
    new_test_dataset_dataloader = DataLoader(
        dataset=new_test_dataset, num_workers=2, shuffle=False, batch_size=batch_size)

    ViewData(new_train_dataset_dataloader, "NewTrain")
    ViewData(new_test_dataset_dataloader, "NewTest")

    print("Device used {0}".format(device.type))

    model = ConvNet(hidden1=conv_hidden_size_1,
                    hidden2=conv_hidden_size_2, num_out=num_class_out)
    model.to(device=device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    if os.path.exists(FILE_PATH):
        print("Loading Model")
        model.load_state_dict(torch.load(FILE_PATH))
    else:
        print("Saving Model")
        losses = []

        model.train(mode=True)
        for epoch in range(epoches):
          temp =[]
          for i, (features, labels) in enumerate(new_train_dataset_dataloader):
              features = torch.as_tensor(
                  data=features, dtype=torch.float32, device=device)
              labels = torch.as_tensor(
                  data=labels, dtype=torch.long, device=device)
              features = features.unsqueeze(dim=1)

              pred_y = model(features)

              loss = loss_func(pred_y, labels)
              loss.backward()
              optimizer.step()
              optimizer.zero_grad()
              temp.append(loss.item())

              if i % 100 == 0:
                  print("Epoch [{0}/{1}] Iter [{2}/{3}] Loss [{4:.5f}]".format(
                      epoch+1, epoches, i+100, 60000/batch_size, loss.item()))
          losses.append(temp)

        model.train(mode=False)

        for index in range(len(losses)):
          lossesv = losses[index]
          print("\n\nEpoch           {0}".format(index+1))
          print("------------------------")

          avg = 0.0
          for value in lossesv:
            avg += value / batch_size
          
          print("Average loss      {0:.5f}".format(avg))


        torch.save(model.state_dict(), FILE_PATH)

    print("\n\nTesting Model")
    with torch.no_grad():
        model.eval()
        test_tot_loss = 0.0
        correct = 0.0
        correctFE = [0.0 for i in range(4)]
        boolCrt = []
        for i, (features, labels) in enumerate(new_test_dataset_dataloader):
            features = torch.as_tensor(
                features, dtype=torch.float32, device=device)
            labels = torch.as_tensor(
                data=labels, dtype=torch.long, device=device)

            features = features.squeeze(dim=0)
            features = features.unsqueeze(dim=1)

            pred_y = model(features)
            loss = loss_func(pred_y, labels)
            test_tot_loss += (loss.item() / batch_size)

            preVal, preValIndexes = torch.max(pred_y, dim=1)

            for index in range(len(preVal)):
                if torch.eq(preValIndexes[index], labels[index]):
                    boolCrt.append(preValIndexes[index])

        for vi in boolCrt:
            correctFE[vi] += 1
            correct += 1

        print("\nOverall Accuracy   [{0:.5f}]".format(correct/10000))
        print("Name          Accuracy")
        print("--------------------------")
        for index, value in enumerate(correctFE):
            print("{0}{1}[{2:.5f}]".format(Word_mapping[index], MakeSpace(Word_mapping[index]), value/Num_mapping
                                         [index]))