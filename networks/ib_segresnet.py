# Obtained from the MONAI github repository.

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from monai.networks.nets import SegResNet
from networks.IB_3D_Kernels import IB_filters_3D

__all__ = ["IB_SegResNet_k3", "IB_SegResNet_k5"]

class ResBlock(nn.Module):
    """
    ResBlock employs skip connection and two convolution blocks and is used
    in SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: Union[Tuple, str],
        kernel_size: int = 3,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_conv_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)
        self.conv2 = get_conv_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)

    def forward(self, x):

        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x

class IB_ResBlock(nn.Module):
    """
    ResBlock employed with IB layers.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: Union[Tuple, str],
        kernel_size: int = 3,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        conv_On_filters = None,
        conv_Off_filters = None,
        IB_Stride: int = 2,
        IB_Padding: int = 2,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.conv_On_filters = conv_On_filters
        self.conv_Off_filters = conv_Off_filters
        self.IB_Stride = IB_Stride
        self.IB_Padding = IB_Padding

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm3 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels//2)
        self.norm4 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels//2)
        self.act = get_act_layer(act)
        self.conv1 = get_conv_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels//2)
        self.conv2 = get_conv_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels//2)
        self.conv3 = get_conv_layer(spatial_dims, in_channels=in_channels//2, out_channels=in_channels//2)
        self.conv4 = get_conv_layer(spatial_dims, in_channels=in_channels//2, out_channels=in_channels//2)

        self.down_conv = get_conv_layer(spatial_dims, in_channels // 2, in_channels, stride=2)

    def forward(self, x):

        # On and Off surround modulation
        sm_on = self.surround_modulation_DoG_on(x)
        sm_off = self.surround_modulation_DoG_off(x)

        identity = self.down_conv(x)

        x = self.norm1(identity)
        x = self.act(x)
        x = self.conv1(x)
        x = x + sm_on
        x = self.norm3(x)
        x = self.act(x)
        on_out = self.conv3(x)

        x = self.norm2(identity)
        x = self.act(x)
        x = self.conv2(x)
        x = x + sm_off
        x = self.norm4(x)
        x = self.act(x)
        off_out = self.conv4(x)
    
        x = torch.cat((on_out, off_out), 1)

        x += identity

        return x

    def surround_modulation_DoG_on(self, input):
        output = F.conv3d(input, weight=self.conv_On_filters, stride=self.IB_Stride, padding=self.IB_Padding)
        return F.relu(output, inplace=True)

    def surround_modulation_DoG_off(self, input):
        output = F.conv3d(input, weight=self.conv_Off_filters, stride=self.IB_Stride, padding=self.IB_Padding)
        return F.relu(output, inplace=True)

class IB_SegResNet_k5(nn.Module):
    """
    SegResNet with IB blocks in the second encoder.
    The model supports 3D inputs.
    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        norm_name: deprecating option for feature normalization type.
        num_groups: deprecating option for group norm. parameters.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.
            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: Optional[float] = None,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
        device_id: int = 0,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.device = torch.device('cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu')

        self.conv_On_filters = IB_filters_3D(radius=2.0, gamma=2. / 3., in_channels=init_filters, 
                                                    out_channels=init_filters, off=False).to(self.device)
        self.conv_Off_filters = IB_filters_3D(radius=2.0, gamma=2. / 3., in_channels=init_filters, 
                                                    out_channels=init_filters, off=True).to(self.device)

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i in range(len(blocks_down)):
            layer_in_channels = filters * 2 ** i
            if i == 1:
                down_layer = nn.Sequential(
                    IB_ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act, 
                                conv_On_filters=self.conv_On_filters, 
                                conv_Off_filters=self.conv_Off_filters),
                    *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(blocks_down[i]-1)],
                )
            else:
                pre_conv = (
                    get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                        if i > 0
                        else nn.Identity()
                    )
                down_layer = nn.Sequential(
                    pre_conv,
                    *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(blocks_down[i])],
                )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: List[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x

class IB_SegResNet_k3(nn.Module):
    """
    SegResNet with IB blocks in the second encoder.
    The model supports 3D inputs.
    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        norm_name: deprecating option for feature normalization type.
        num_groups: deprecating option for group norm. parameters.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.
            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: Optional[float] = None,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
        device_id: int = 0,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.device = torch.device('cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu')

        self.conv_On_filters = IB_filters_3D(radius=1.0, gamma=1. / 2., in_channels=init_filters, 
                                                    out_channels=init_filters, off=False).to(self.device)
        self.conv_Off_filters = IB_filters_3D(radius=1.0, gamma=1. / 2., in_channels=init_filters, 
                                                    out_channels=init_filters, off=True).to(self.device)

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i in range(len(blocks_down)):
            layer_in_channels = filters * 2 ** i
            if i == 1:
                down_layer = nn.Sequential(
                    IB_ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act, 
                                conv_On_filters=self.conv_On_filters, 
                                conv_Off_filters=self.conv_Off_filters, IB_Padding=1),
                    *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(blocks_down[i]-1)],
                )
            else:
                pre_conv = (
                    get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                        if i > 0
                        else nn.Identity()
                    )
                down_layer = nn.Sequential(
                    pre_conv,
                    *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(blocks_down[i])],
                )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: List[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x