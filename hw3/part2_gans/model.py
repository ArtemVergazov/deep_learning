import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm



class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        # TODO
        self.num_features = num_features
        self.embed_features = embed_features

        self.gamma = spectral_norm(nn.Linear(embed_features, num_features))
        # Bias is already initialized in nn.Linear.
        # self.bias = spectral_norm(nn.Linear(embed_features, num_features))
        self.batch_norm = nn.BatchNorm2d(num_features, affine=True)  # TODO: True or False?

    def forward(self, inputs, embeds):
        gamma = self.gamma(embeds) # TODO 
        # bias = self.bias(embeds) # TODO

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        # assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = self.batch_norm(inputs) # TODO: apply batchnorm

        return outputs * gamma[..., None, None]# + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_channels = embed_channels
        self.batchnorm = batchnorm

        self.skip_connection = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)) \
            if in_channels != out_channels \
            else None

        self.abn1 = AdaptiveBatchNorm(in_channels, embed_channels) if batchnorm else None
        self.abn2 = AdaptiveBatchNorm(out_channels, embed_channels) if batchnorm else None
        
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample else None
        self.downsample = nn.AvgPool2d(2) if downsample else None

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        # TODO

        if self.upsample is not None:
            inputs = self.upsample(inputs)

        if self.skip_connection is not None:
            inputs_skip = self.skip_connection(inputs)
        else:
            inputs_skip = inputs.clone()
        
        if self.abn1 is not None:
            inputs = self.abn1(inputs, embeds)
            
        inputs = self.conv1(self.relu1(inputs))
        
        if self.abn2 is not None:
            inputs = self.abn2(inputs, embeds)
            
        inputs = self.conv2(self.relu2(inputs))
        outputs = inputs + inputs_skip
        
        if self.downsample is not None:
            outputs = self.downsample(outputs)

        return outputs


class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.output_size = 4 * 2**num_blocks
        # TODO

        self.min_channels = min_channels
        self.max_channels = max_channels
        self.noise_channels = noise_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.use_class_condition = use_class_condition

        if use_class_condition:
            self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=noise_channels)
            self.map = spectral_norm(nn.Linear(noise_channels * 2, max_channels * 4 * 4))
            embedding_channels = noise_channels * 2
            batchnorm = True
        else:
            self.map = spectral_norm(nn.Linear(noise_channels, max_channels * 4 * 4))
            embedding_channels = None
            batchnorm=False

        in_channels = max_channels
        self.preactres_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.preactres_blocks.append(PreActResBlock(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                embed_channels=embedding_channels,
                batchnorm=batchnorm,
                upsample=True,
                downsample=False,
            ))
            in_channels //= 2
         
        self.bn = nn.BatchNorm2d(min_channels, affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv = spectral_norm(nn.Conv2d(min_channels, 3, kernel_size=3, padding=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise, labels):
        # TODO
        
        if self.use_class_condition:
            embeds = torch.squeeze(self.embedding(labels), 1)
            embeds = torch.cat([noise, embeds], dim=1)
            x = self.map(embeds)
        else:
            embeds=None
            x = self.map(noise)
        
        x = x.reshape(x.shape[0], self.max_channels, 4, 4)
        
        for i in range(self.num_blocks):
            x = self.preactres_blocks[i](x, embeds)
                                   
        x = self.conv(self.relu(self.bn(x)))
        outputs = self.sigmoid(x)

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()
        # TODO

        self.min_channels = min_channels
        self.max_channels = max_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.use_projection_head = use_projection_head
        
        self.conv = spectral_norm(nn.Conv2d(3, min_channels, kernel_size=1))
        
        if use_projection_head:
            self.embedding = spectral_norm(nn.Embedding(num_embeddings=num_classes, embedding_dim=max_channels))
        
        self.psi = spectral_norm(nn.Linear(max_channels, 1))

        in_channels = min_channels
        self.preactres_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.preactres_blocks.append(PreActResBlock(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                embed_channels=None,
                batchnorm=False,
                upsample=False,
                downsample=True,
            ))
            in_channels *= 2
                                   
        self.relu = nn.ReLU()

    def forward(self, inputs, labels):

        x = self.conv(inputs)
        
        for i in range(self.num_blocks):
            x = self.preactres_blocks[i](x, embeds=None)
                                   
        x = torch.sum(self.relu(x), [2, 3])
        
        scores = self.psi(x)
        scores = torch.squeeze(scores, 1)
        
        if self.use_projection_head:
            embeds = self.embedding(labels)
            embeds = torch.squeeze(embeds, 1)
            scores = scores + torch.sum(embeds * x, dim=1)

        assert scores.shape == (inputs.shape[0],)
        return scores
