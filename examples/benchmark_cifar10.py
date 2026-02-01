import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
import hydra
from hydra.core.config_store import ConfigStore
from typing import Dict, Any

from src.conf.config import DistillationConfig
from src.infra.model_wrapper import PyTorchModelWrapper
from src.infra.loader import DataLoaderFactory
from src.infra.loss import KLDivergenceLoss, AttentionTransferLoss
from src.engine.trainer import Trainer

# Register config
cs = ConfigStore.instance()
cs.store(name="cifar_config", node=DistillationConfig)

class StudentResNet(nn.Module):
    """Simplified ResNet for student (just to demonstrate architectural diff)."""
    def __init__(self):
        super().__init__()
        # Using a very small model for speed in demonstration
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

@hydra.main(version_base=None, config_name="cifar_config")
def main(cfg: DistillationConfig):
    print(f"Running Benchmark with Batch Size: {cfg.data.batch_size}")

    # 1. Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root=cfg.data.data_root, train=True, download=True, transform=transform_train)
    train_loader = DataLoaderFactory.create(trainset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    testset = datasets.CIFAR10(root=cfg.data.data_root, train=False, download=True, transform=transform_test)
    test_loader = DataLoaderFactory.create(testset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    # 2. Models
    # Teacher: Pretrained ResNet18
    print("Preparing Teacher Model...")
    teacher = resnet18(weights='DEFAULT')
    teacher.fc = nn.Linear(512, 10)

    # Quick Fine-tune Teacher (1 Epoch) using standard CE Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)
    teacher.train()
    optimizer_t = optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9)
    criterion_t = nn.CrossEntropyLoss()

    print("🚀 Fine-tuning Teacher for 1 Epoch (to make it a valid teacher)...")
    from tqdm import tqdm
    for inputs, targets in tqdm(train_loader, desc="Teacher Fine-tune"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_t.zero_grad()
        outputs = teacher(inputs)
        loss = criterion_t(outputs, targets)
        loss.backward()
        optimizer_t.step()

    # Validate Teacher
    teacher.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = teacher(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f"✅ Teacher Accuracy after fine-tuning: {100.*correct/total:.2f}%")

    teacher_wrapper = PyTorchModelWrapper(teacher)

    # Student: ResNet9 (Simplified but deeper)
    class ResNet9(nn.Module):
        def __init__(self, in_channels=3, num_classes=10):
            super().__init__()
            self.conv1 = self.conv_block(in_channels, 64)
            self.conv2 = self.conv_block(64, 128, pool=True)
            self.res1 = nn.Sequential(self.conv_block(128, 128), self.conv_block(128, 128))
            self.conv3 = self.conv_block(128, 256, pool=True)
            self.conv4 = self.conv_block(256, 512, pool=True)
            self.res2 = nn.Sequential(self.conv_block(512, 512), self.conv_block(512, 512))
            self.classifier = nn.Sequential(
                nn.MaxPool2d(4),
                nn.Flatten(),
                nn.Linear(512, num_classes)
            )

        def conv_block(self, in_channels, out_channels, pool=False):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            if pool: layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            return self.classifier(out)

    student = ResNet9()
    student_wrapper = PyTorchModelWrapper(student)

    # 3. Loss & Opt
    kl_loss = KLDivergenceLoss(temperature=cfg.loss.temperature, alpha=cfg.loss.alpha)
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) # Optimized LR for ResNet9

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, epochs=cfg.train.epochs, steps_per_epoch=len(train_loader)
    )

    # 4. Trainer with MLOps
    trainer = Trainer(
        student=student_wrapper,
        teacher=teacher_wrapper,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=kl_loss,
        scheduler=scheduler,  # Passed OneCycleLR
        config=cfg,
        project_name="cifar10-benchmark"
    )

    trainer.fit(epochs=cfg.train.epochs)

    # 5. Evaluation
    print("\nStarting Evaluation...")
    from src.engine.evaluator import Evaluator
    evaluator = Evaluator(
        model=student_wrapper,
        val_loader=test_loader,
        config=cfg
    )

    metrics = evaluator.evaluate()
    print(f"Final Student Metrics: {metrics}")

    # 6. Inference Preview (Ultralytics Style)
    print("\nGenerating Inference Preview...")
    from src.utils.plotting import plot_images

    # Get a batch
    data_iter = iter(test_loader)
    images, targets = next(data_iter)

    # Predict
    device = trainer.device
    student.eval()
    with torch.no_grad():
        outputs = student(images.to(device))
        preds = outputs.argmax(dim=1)

    # Plot
    # CIFAR-10 Classes
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    save_path = f"{trainer.output_dir}/val_batch0.jpg"
    plot_images(images, targets, preds.cpu(), classes, save_path)

    # Ensure Result Plots are up to date
    from src.utils.plotting import plot_results
    plot_results(f"{trainer.output_dir}/results.csv", trainer.output_dir)

if __name__ == "__main__":
    main()
