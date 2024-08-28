import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch import nn, optim

# AI-driven hyperparameter tuning
def auto_tune_hyperparameters():
    # Example: AI algorithm to select optimal hyperparameters
    learning_rate = 0.01  # Dynamically adjusted
    batch_size = 64       # Determined by AI
    return learning_rate, batch_size

# AI-driven dynamic learning rate adjustment
class AI_LR_Scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.best_loss = float('inf')
    
    def step(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.9  # Decrease learning rate if no improvement

# AI-driven intelligent checkpointing
def save_checkpoint(model, optimizer, epoch, loss, best_loss, filepath='checkpoint.pth'):
    if loss < best_loss:
        print(f"Saving checkpoint at epoch {epoch} with loss {loss:.4f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, filepath)

# Initialize distributed environment
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# Main training loop
def train(rank, world_size):
    setup_distributed(rank, world_size)
    
    learning_rate, batch_size = auto_tune_hyperparameters()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate)
    scheduler = AI_LR_Scheduler(optimizer)

    best_loss = float('inf')
    for epoch in range(10):
        total_loss = 0.0
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        save_checkpoint(ddp_model, optimizer, epoch, avg_loss, best_loss)
        best_loss = min(best_loss, avg_loss)

        print(f"Rank {rank}, Epoch {epoch}, Loss: {avg_loss:.4f}")

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 2  # Example for 2 GPUs
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
