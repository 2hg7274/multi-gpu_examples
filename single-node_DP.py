import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Set Hyper Parameters and other varibales to train the model
batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.5
seed = 1
log_interval = 200


# Define Neural Networks Model.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)


    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x.view(-1, 784)))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        output = F.log_softmax(h6, dim=1)
        # print("\tIn Model: input size", x.size(), "output size", output.size())
        return output
    


#Define Train function and Test function to validate.
def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print("Outside: input size", data.size(), "output size", output.size())

        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            # CUDA Memory Allocated
            for i in range(torch.cuda.device_count()):
                print(f"GPU{i} Memory Usages: ", round(torch.cuda.memory_allocated(i)/1024**2, 1), "MB")
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    


def main():
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'GPU Device: {torch.cuda.device_count()}')
    kwargs = {'num_workers': 4*torch.cuda.device_count(), 'pin_memory': True} if torch.cuda.is_available() else {}
    print("set vars and device done")

    #Prepare Data Loader for Training and Validation
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform), 
        batch_size=batch_size, 
        shuffle=True, 
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,transform=transform), 
        batch_size=test_batch_size, 
        shuffle=True, 
        **kwargs)
    

    model = Net()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    # Train and Test the model and save it.
    for epoch in range(1, epochs+1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    torch.save(model, './model_output/model_DP.pt')

if __name__ == "__main__":
    main()