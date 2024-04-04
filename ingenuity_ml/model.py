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
        return x


class MNISTClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            ConvBlock(1, 64),
            *[ConvBlock(64, 64, kernel_size=3) for _ in range(5)],
            ClassifierHead(n_input=64 * 28 * 28, n_hidden=120, n_classes=10)
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.classifier(x)
        loss = F.cross_entropy(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        # print(loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self.classifier(x).argmax(1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def train(batch_size: int = 256, epochs: int = 10):

    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=7,
        persistent_workers=True,
    )

    model = MNISTClassifier()
    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=epochs)
    trainer.fit(model=model, train_dataloaders=train_loader)

    return model, dataset, train_loader, trainer


if __name__ == "__main__":
    train()
