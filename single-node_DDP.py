import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os



def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)



class Trainer:
    def __init__(self, model: torch.nn.Module, 
                 train_data: DataLoader, val_data: DataLoader, 
                 optimizer: torch.optim.Optimizer, gpu_id: int, save_every: int):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.log_interval = 50

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.nll_loss(output, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss


    def _run_train_epoch(self, epoch, log_interval):
        self.model.train()
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        for batch_idx, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)

            if batch_idx % log_interval == 0:
                print('[GPU{}]Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.gpu_id, epoch, batch_idx * len(source), len(self.train_data.dataset),
                    100. * batch_idx / len(self.train_data), loss.item()))


    def _run_test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for source, target in self.val_data:
                source, target = source.to(self.gpu_id), target.to(self.gpu_id)
                output = self.model(source)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.val_data.dataset)

        print('\n[GPU{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            self.gpu_id, test_loss, correct, len(self.val_data.dataset), 100. * correct / len(self.val_data.dataset)))


    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = './model_output/model_DDP.pt'
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    
    def train(self, max_epochs: int):
        for epoch in range(1, max_epochs+1):
            self._run_train_epoch(epoch, self.log_interval)
            self._run_test()
            if self.gpu_id == 0 and epoch%self.save_every==0:
                self._save_checkpoint(epoch)



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

        return output

class MNISTDataset(Dataset):
    def __init__(self, split, transform):
        self.split = split
        self.transform = transform
        self.data = datasets.MNIST('./data', train=self.split, transform=self.transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]


def main(rank: int, world_size: int, 
         save_every: int, total_epoch: int, batch_size: int, test_batch_size: int,
         seed: int, lr: float, momentum: float):
    ddp_setup(rank, world_size)
    torch.manual_seed(seed)
    print(f'GPU Device: {torch.cuda.device_count()}')
    kwargs = {'num_workers': 4*torch.cuda.device_count(), 'pin_memory': True} if torch.cuda.is_available() else {}
    print("set vars and device done")

    #Prepare Data Loader for Training and Validation
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = MNISTDataset(True, transform)
    test_dataset = MNISTDataset(False, transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=DistributedSampler(train_dataset), 
        **kwargs)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size, 
        shuffle=False,
        sampler=DistributedSampler(test_dataset), 
        **kwargs)
    
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    trainer = Trainer(model, train_loader, test_loader, optimizer, rank, save_every)
    trainer.train(total_epoch)
    destroy_process_group()




if __name__=="__main__":
    batch_size = 128
    test_batch_size = 1000
    epochs = 20
    lr = 0.01
    momentum = 0.5
    seed = 1
    save_every= 2

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, save_every, epochs, batch_size, test_batch_size, seed, lr, momentum), nprocs=world_size)