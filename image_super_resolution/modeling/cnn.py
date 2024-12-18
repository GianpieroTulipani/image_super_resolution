import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation
from torch import Tensor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        """
        Implements a Residual Block with upsampling, downsampling, and feature extraction.
        
        Args:
            in_channels (int): Number of input channels for the block.
            out_channels (int): Number of output channels after the block.
            kernel_size (int): The kernel size for convolutional layers.
            stride (int): The stride for convolutional layers.
        """
        super(ResidualBlock, self).__init__()

        # Define upsampling and downsampling layers
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1
        )
        
        self.downsample = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.squeeze_excitation = SqueezeExcitation(
            input_channels=in_channels,
            squeeze_channels=in_channels // 16
        )
        
        self.relu = nn.ReLU()

    def feature_extraction(self, feature_map: Tensor) -> Tensor:
        """
        Apply a series of convolutional layers with ReLU activation.
        
        Args:
            feature_map (Tensor): Input feature map to be processed.
        
        Returns:
            Tensor: Refined feature map after applying convolution and ReLU.
        """
        feature_map = self.relu(self.conv(feature_map))
        feature_map = self.relu(self.conv(feature_map))
        return self.relu(self.conv(feature_map))

    def forward(self, input_features: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass for the residual block.
        
        Args:
            input_features (Tensor): Input feature map to be processed by the block.
        
        Returns:
            tuple[Tensor, Tensor]: Returns the upsampled features and the residual features with attention.
        """
        # Upsample the input features
        upsampled_features = self.upsample(input_features)

        # Downsample the upsampled features
        downsampled_features = self.downsample(upsampled_features)

        # Compute the residual difference
        residual = input_features - downsampled_features

        # Perform feature extraction
        refined_features = self.feature_extraction(residual)
        channels_attention = self.squeeze_excitation(refined_features)
        
        return upsampled_features, (residual + channels_attention)


class LaplacianPyramidAttentionNetwork(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Implements the Laplacian Pyramid Attention Network for multi-scale image processing.
        
        Args:
            in_channels (int): Number of input channels for the network.
            out_channels (int): Number of output channels after the network.
        """
        super(LaplacianPyramidAttentionNetwork, self).__init__()

        # Initial convolution layers for feature extraction
        self.conv1 = nn.Conv2d(3, 4 * out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4 * out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Downsampling layer
        self.downsample_2x = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        
        self.upsample_8x = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=10,
            stride=8,
            padding=1
        )
        self.upsample_2x = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.channels_compression_8x = nn.Conv2d(
            in_channels=(2 * in_channels),
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )

        self.channels_compression_4x = nn.Conv2d(
            in_channels=(4 * in_channels),
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )
        
        self.channels_compression_2x = nn.Conv2d(
            in_channels=(6 * in_channels),
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )

        # Residual blocks for various scales
        self.residual_block_2x = ResidualBlock(in_channels, out_channels, kernel_size=4, stride=2)
        self.residual_block_4x = ResidualBlock(in_channels, out_channels, kernel_size=6, stride=4)
        self.residual_block_8x = ResidualBlock(in_channels, out_channels, kernel_size=10, stride=8)

        # Reconstruction layer to convert features back to an RGB image
        self.reconstruction_layer = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

    def extract_features(self, input_image: Tensor) -> Tensor:
        """
        Extract features from the input image using convolution layers.
        
        Args:
            input_image (Tensor): Input image to be processed.
        
        Returns:
            Tensor: Extracted feature map.
        """
        features = self.relu(self.conv1(input_image))
        features = self.relu(self.conv2(features))
        return self.relu(self.conv3(features))

    def reconstruct_image(self, features: Tensor) -> Tensor:
        """
        Reconstruct the RGB image from feature maps.
        
        Args:
            features (Tensor): Feature maps to be converted into an RGB image.
        
        Returns:
            Tensor: Reconstructed RGB image.
        """
        features = self.relu(self.conv3(features))
        return self.reconstruction_layer(features)

    def large_residual_block(self, residual_block, channels_compression, input_residual: Tensor, num_blocks: int) -> tuple[Tensor, Tensor]:
        """
        Applies a series of residual blocks for feature refinement and compression.
        
        Args:
            residual_block: The residual block to be used.
            channels_compression: The compression layer applied after residual blocks.
            input_residual (Tensor): The input residual feature map.
            num_blocks (int): The number of residual blocks to apply.
        
        Returns:
            tuple[Tensor, Tensor]: The concatenated output of refined features and the residual.
        """
        concat_list = []
        
        for i in range(num_blocks):
            refined, residual = residual_block(input_residual)
            if i == 0:
                combined = refined
            else:
                combined = combined + refined
            concat_list.append(refined)
        concat = torch.cat(concat_list, dim=1)

        return self.relu(channels_compression(concat)), residual

    def forward(self, input_image: Tensor) -> Tensor:
        """
        Forward pass for the Laplacian Pyramid Attention Network.
        
        Args:
            input_image (Tensor): The input image to be processed by the network.
        
        Returns:
            Tensor: The reconstructed image after processing through the network.
        """
        features = self.extract_features(input_image)

        upsampled_rgb_8x = self.upsample_8x(features)

        refined_8x, residual_8x = self.large_residual_block(
            residual_block=self.residual_block_8x,
            channels_compression=self.channels_compression_8x,
            input_residual=features,
            num_blocks=2
        )
        refined_8x = refined_8x + upsampled_rgb_8x
        upsampled_rgb_4x = self.downsample_2x(refined_8x)

        refined_4x, residual_4x = self.large_residual_block(
            residual_block=self.residual_block_4x,
            channels_compression=self.channels_compression_4x,
            input_residual=residual_8x,
            num_blocks=4
        )
        refined_4x = refined_4x + upsampled_rgb_4x
        upsampled_rgb_2x = self.downsample_2x(refined_4x)
        
        refined_2x, residual_2x = self.large_residual_block(
            residual_block=self.residual_block_2x,
            channels_compression=self.channels_compression_2x,
            input_residual=residual_4x,
            num_blocks=6
        )

        refined_2x = refined_2x + upsampled_rgb_2x
        output_2x = refined_2x + self.upsample_2x(residual_2x)
        reconstructed_image = self.reconstruct_image(output_2x)

        return reconstructed_image