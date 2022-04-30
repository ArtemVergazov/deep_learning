import scipy
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import numpy as np
from scipy import linalg
from scipy.stats import entropy



class GANLoss(nn.Module):
    """
    GAN loss calculator

    Variants:
      - non_saturating
      - hinge
    """
    def __init__(self, loss_type):
        super(GANLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, fake_scores, real_scores=None):
        if real_scores is None:
            # TODO: calculate generator loss (2 points)
            if self.loss_type == 'non_saturating':
                loss = -fake_scores.sigmoid().log().mean()
            else:  # hinge loss
                loss = -fake_scores.mean()

        else:
            # TODO: calculate discriminator loss (2 points)
            if self.loss_type == 'non_saturating':
                loss = -(real_scores.sigmoid().log() + (1 - fake_scores.sigmoid()).log()).mean()
            else:  # hinge loss
                loss = (1 - real_scores).relu().mean() + (1 + fake_scores).relu().mean()

        return loss


class ValLoss(nn.Module):
    """
    Calculates FID and IS
    """
    def __init__(self):
        super().__init__()
        self.inception_v3 = models.inception_v3(pretrained=True)
        self.inception_v3.eval()

        for p in self.inception_v3.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocess data
        x = F.interpolate(x, size=(299, 299), mode='bilinear')
        x = (x - 0.5) * 2

        # N x 3 x 299 x 299
        x = self.inception_v3.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception_v3.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception_v3.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception_v3.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception_v3.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception_v3.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception_v3.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception_v3.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception_v3.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception_v3.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception_v3.Mixed_7c(x)
        # Adaptive average pooling
        x = self.inception_v3.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.inception_v3.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)

        return x

    @torch.no_grad()
    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        # N x 2048
        x = self.inception_v3.fc(x)
        # N x 1000 (num_classes)
        x = F.softmax(x, dim=1)

        return x

    def calc_data(self, real_inputs: list, fake_inputs: list):
        real_features = []
        for real_inputs_batch in real_inputs:
            real_features_batch = self._features(real_inputs_batch)
            real_features.append(real_features_batch.detach().cpu().numpy())            
        real_features = np.concatenate(real_features)

        fake_features = []
        fake_probs = []

        for fake_inputs_batch in fake_inputs:
            fake_features_batch = self._features(fake_inputs_batch)
            fake_probs_batch = self._classifier(fake_features_batch)

            fake_features.append(fake_features_batch.detach().cpu().numpy())
            fake_probs.append(fake_probs_batch.detach().cpu().numpy())

        fake_features = np.concatenate(fake_features)
        fake_probs = np.concatenate(fake_probs)

        return real_features, fake_features, fake_probs

    # This method incorporates the algorithm from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py.
    @staticmethod
    def calc_fid(real_features, fake_features):

        mu1 = real_features.mean(axis=0)
        mu2 = fake_features.mean(axis=0)

        sigma1 = np.cov(real_features, rowvar=False)
        sigma2 = np.cov(fake_features, rowvar=False)

        diff = mu1 - mu2

        # Product might be almost singular.
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        if not np.isfinite(covmean).all():
            eps = 1e-6
            print(f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates')
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

        # Numerical error might give slight imaginary component.
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    # This method incorporates the algorithm from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py.
    @staticmethod
    def calc_is(fake_probs):
        
        # Compute the mean kl-div.
        py = np.mean(fake_probs, axis=0)
        scores = []
        for i in range(fake_probs.shape[0]):
            pyx = fake_probs[i, :]
            scores.append(entropy(pyx, py))
        return np.exp(np.mean(scores))

    def forward(self, real_images: list, fake_images: list) -> torch.Tensor:
        real_features, fake_features, fake_probs = self.calc_data(real_images, fake_images)

        fid = self.calc_fid(real_features, fake_features)

        inception_score = self.calc_is(fake_probs)

        return fid, inception_score
