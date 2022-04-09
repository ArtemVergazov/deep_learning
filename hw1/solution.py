# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys

# List any extra packages you need here
PACKAGES_TO_INSTALL = ["gdown==4.4.0",]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

import torch
# Your code here...
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

# %load_ext tensorboard
# from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18

import numpy as np
import matplotlib.pyplot as plt

import random
import os.path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(0xC0FFEE)


def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.

    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    # Your code here
    path = os.path.join(path, kind)

    # Inspired by https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html#transforms,
    #             https://pytorch.org/vision/stable/transforms.html
    if kind == 'train':
        tf = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(.5, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    dataset = ImageFolder(
        path,
        transform=tf,
    )
    
    shuffle = kind == 'train'
    
    return DataLoader(dataset, batch_size=64, shuffle=shuffle, pin_memory=True)


def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    # Your code here
    # Inspired by http://cs231n.stanford.edu/reports/2017/pdfs/931.pdf
    #             https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet18(pretrained=False, progress=True, num_classes=200)
    model.fc = nn.Sequential(
        nn.Dropout(.5),
        nn.Linear(512, 200),
    )
    return model.to(device)


def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    # Your code here
    optimizer = SGD(model.parameters(), lr=1e-2, momentum=.9)
    return optimizer


def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    # Your code here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model(batch.to(device))
    

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    # Your code here
    # Inspired by https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/361c46b0182e04464775765192096219/optimization_tutorial.ipynb#scrollTo=i9lSncaIXYCW
    
    total_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    running_correct = 0
    running_loss = 0.

    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = predict(model, X)
            _, pred_classes = pred.max(1)

            loss = loss_fn(pred, y)
            running_correct += (pred_classes == y).sum().item()
            running_loss += float(loss.item())

    accuracy = running_correct / total_samples
    mean_loss = running_loss / num_batches

    print(f"\nTest Error: \n Accuracy: {accuracy}, Avg loss: {mean_loss}\n")

    return accuracy, mean_loss
    

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    # Your code here
#     writer = SummaryWriter('runs/tiny_image_net_experiment1')
#     %tensorboard --logdir="./runs/tiny_image_net_experiment1"

    total_samples = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    batch_size = total_samples / num_batches

    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 30

    scheduler = OneCycleLR(
        optimizer,
        max_lr=.02,
        steps_per_epoch=num_batches,
        epochs=epochs,
        div_factor=10,
        final_div_factor=10,
    )

    for epoch in range(epochs):

        running_loss = 0.
        running_correct = 0

        batch_count = 0

        model.train()

        for batch, (X, y) in enumerate(train_dataloader):

            batch_count += 1

            # Compute prediction and loss
            X = X.to(device)
            y = y.to(device)

            # logits = predict(model, X)
            # pred = nn.Softmax(dim=1)(logits)
            pred = predict(model, X)

            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())

            _, pred_classes = pred.max(1)
            running_correct += (pred_classes == y).sum().item()

            # Print training progress.
            if batch % 100 == 0:
                print(f'\nEpoch {epoch}/{epochs} Batch {batch}/{num_batches} loss: {running_loss / batch_count} accuracy: {running_correct / batch_count / batch_size}\n')

        model.eval()

        print('\ntrain stats:')
        print(f"epoch: {epoch}/{epochs} loss: {running_loss / batch_count}")
        print(f"epoch: {epoch}/{epochs} accuracy: {running_correct / batch_count / batch_size}\n")

#         writer.add_scalar('loss/training', running_loss / batch_count, epoch)
#         writer.add_scalar('accuracy/training', running_correct / batch_count / batch_size, epoch)

        # Validation.
        val_accuracy, val_loss = validate(val_dataloader, model)

        print('\nval stats:')
        print(f"epoch: {epoch}/{epochs} loss: {val_loss}")
        print(f"epoch: {epoch}/{epochs} accuracy: {val_accuracy}\n")

#         writer.add_scalar('loss/validation', val_loss, epoch)
#         writer.add_scalar('accuracy/validation', val_accuracy, epoch)

        # scheduler.step(val_loss)

#         torch.save(model.state_dict(), f'model_3_weights_epoch_{epoch}.pth')
    
#     writer.close()
    

def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here
    model.load_state_dict(torch.load(checkpoint_path))
    

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    md5_checksum = "0c39a127027c893946dab5a851b926d1"
    google_drive_link = "https://drive.google.com/file/d/1eAf16xpXCQJbZVoSobmQ8No_eidoVhKs/view?usp=sharing"

    return md5_checksum, google_drive_link
