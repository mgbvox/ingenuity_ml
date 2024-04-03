import os

import lightning as L
from torch import optim, nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.dropout = nn.Dropout2d()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = F.relu(x, inplace=False)
        # x = self.dropout(x)
        return x


class ClassifierHead(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_classes: int):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.layer_in = nn.Linear(n_input, n_hidden)
        self.layer_out = nn.Linear(n_hidden, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer_in(self.flatten(x))
        x = F.relu(x)
        x = self.layer_out(x)
        # x = F.relu(x)
        return x


class MNISTClassifier(L.LightningModule):
    def __init__(self, classifier_module: nn.Module):
        super().__init__()
        self.classifier = classifier_module

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.classifier(x)
        loss = F.cross_entropy(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        print(loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    n_layer = 5

    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=7, persistent_workers=True)
    classifier_module = nn.Sequential(
        ConvBlock(1, 64),
        *[ConvBlock(64, 64, kernel_size=3) for _ in range(n_layer)],
        ClassifierHead(n_input=64 * 28 * 28, n_hidden=120, n_classes=10)
    )

    model = MNISTClassifier(classifier_module)
    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=10)
    trainer.fit(model=model, train_dataloaders=train_loader)
    


if __name__ == "__main__":
    main()
