import os
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy
)


def fsdp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def clean_up():
    destroy_process_group()



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
    

class Trainer:
    def __init__(self, model: torch.nn.Module,
                 train_data: DataLoader, test_data: DataLoader,
                 optimizer: torch.optim.Optimizer, gpu_id: int, save_every: int):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = FSDP(self.model)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.log_interval = 50

    def _run_batch(self, data, targets):
        ddp_loss = torch.zeros(2).to(self.gpu_id)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, targets)
        loss.backward()
        self.optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

        return ddp_loss
    
    def _run_train_epoch(self, epoch):
        self.model.train()
        self.train_data.sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(self.train_data):
            data, target = data.to(self.gpu_id), target.to(self.gpu_id)
            ddp_loss = self._run_batch(data, target)
  
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if self.gpu_id == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0]/ddp_loss[1]))

        
    def _run_test(self):
        self.model.eval()
        ddp_loss = torch.zeros(3).to(self.gpu_id)
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.gpu_id), target.to(self.gpu_id)
                output = self.model(data)
                ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
                ddp_loss[2] += len(data)

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if self.gpu_id == 0:
            test_loss = ddp_loss[0] / ddp_loss[2]
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, int(ddp_loss[1]), int(ddp_loss[2]), 100.*ddp_loss[1]/ddp_loss[2]
            ))

    
    def _save_checkpoint(self, epoch):
        dist.barrier()
        ckp = self.model.state_dict()
        self.PATH = './model_output/model_FSDP.pt'
        torch.save(ckp, self.PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {self.PATH}")

    def train(self, max_epochs: int):
        for epoch in range(1, max_epochs+1):
            self._run_train_epoch(epoch)
            self._run_test()
            if self.gpu_id==0 and epoch%self.save_every==0:
                self._save_checkpoint(epoch)
            


def fsdp_main(rank: int, world_size: int,
              save_every: int, total_epoch: int, batch_size: int, test_batch_size: int,
              lr: float, momentum: float):
    fsdp_setup(rank, world_size)
    kwargs = {'num_workers': 4*torch.cuda.device_count(), 'pin_memory': True} if torch.cuda.is_available() else {}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = MNISTDataset(True, transform)
    test_dataset = MNISTDataset(False, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_dataset,rank=rank,num_replicas=world_size,shuffle=True),
        **kwargs)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=DistributedSampler(test_dataset,rank=rank,num_replicas=world_size),
        **kwargs)
    
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100)
    torch.cuda.set_device(rank)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        test_data=test_loader,
        optimizer=optimizer,
        gpu_id=rank,
        save_every=save_every)
    trainer.train(total_epoch)
    
    clean_up()


if __name__=="__main__":
    batch_size = 128
    test_batch_size = 1000
    epochs = 20
    lr = 0.01
    momentum = 0.5
    save_every= 1

    world_size = torch.cuda.device_count()
    mp.spawn(fsdp_main,
             args=(world_size,
                   save_every,
                   epochs,
                   batch_size,
                   test_batch_size,
                   lr,
                   momentum),
            nprocs=world_size,
            join=True)