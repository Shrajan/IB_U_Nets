import torch
import torch.nn as nn
from torch.nn import init

class Conv3x3(nn.Module):
  """
  3x3 conv + batch normalization (+ nonlinearity).
  if downsample = True, a stride of 2 is used
  and the input shape will be halved by convolution.
  """
  def __init__(self, in_channels, out_channels, downconv=False, nonlin=None, dropout_rate=0.0):
    super(Conv3x3, self).__init__()
    stride = 2 if downconv else 1
    self.nonlin = nonlin
    self.conv_block = nn.Sequential(
      nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
      nn.BatchNorm3d(num_features=out_channels)
    )
    self.dropout_rate = dropout_rate
    if dropout_rate > 0.0:
      self.do = nn.Dropout(p=dropout_rate)

  def forward(self, x):
    x = self.conv_block(x)
    if self.nonlin is not None:
      x = self.nonlin(x)
    if self.dropout_rate > 0.0 and self.training:
      x = self.do(x)
    return x

class TConv(nn.Module):
  """
  Transposed conv + batch normalization (+ nonlinearity).
  Output shape will be input shape * 2.
  """
  def __init__(self, in_channels, out_channels, nonlin=None):
    super(TConv, self).__init__()
    self.conv_block = nn.Sequential(
      nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=False),
      nn.BatchNorm3d(num_features=out_channels)
    )
    self.nonlin = nonlin

  def forward(self, x):
    x = self.conv_block(x)
    if self.nonlin is not None:
      x = self.nonlin(x)
    return x

class TConvHW(nn.Module):
  """
  Transposed conv + batch normalization (+ nonlinearity).
  Output shape will be input shape * 2.
  """
  def __init__(self, in_channels, out_channels, nonlin=None):
    super(TConvHW, self).__init__()
    self.conv_block = nn.Sequential(
      nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2,2,1), stride=(2,2,1), bias=False),
      nn.BatchNorm3d(num_features=out_channels)
    )
    self.nonlin = nonlin

  def forward(self, x):
    x = self.conv_block(x)
    if self.nonlin is not None:
      x = self.nonlin(x)
    return x


class Encoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, nonlin, downconv=False, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.conv1 = Conv3x3(in_channels=in_channels, out_channels=mid_channels, nonlin=nonlin, downconv=downconv, dropout_rate=dropout_rate)
        self.conv2 = Conv3x3(in_channels=mid_channels, out_channels=out_channels, nonlin=nonlin, downconv=False, dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, nonlin, dropout_rate=0.0, upconv = True):
        super(Decoder, self).__init__()
        self.conv1 = Conv3x3(in_channels=in_channels, out_channels=out_channels, nonlin=nonlin, dropout_rate=dropout_rate)
        self.conv2 = Conv3x3(in_channels=out_channels, out_channels=out_channels, nonlin=nonlin, dropout_rate=dropout_rate)
        self.tconv = TConv(in_channels=out_channels, out_channels=out_channels, nonlin=nonlin)
        self.upconv = upconv

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.upconv:
          x = self.tconv(x)
        return x

class Bottom(nn.Module):
    def __init__(self, in_channels=256, mid_channels=256, out_channels=512, downconv=True, nonlin=nn.LeakyReLU(0.2), dropout_rate=0.0):
        super(Bottom, self).__init__()
        self.conv1 = Conv3x3(in_channels=in_channels, out_channels=mid_channels, downconv=downconv, nonlin=nonlin, dropout_rate=dropout_rate)
        self.conv2 = Conv3x3(in_channels=mid_channels, out_channels=out_channels, nonlin=nonlin, dropout_rate=dropout_rate)
        self.tconv = TConvHW(in_channels=out_channels, out_channels=out_channels, nonlin=nonlin)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.tconv(x)
        return x


class UNet(nn.Module):
    """
    Classical 3D UNet.
    """
    def __init__(self, input_channels=1, output_channels=1, init_filters=16, dropout_rate=0.5):
        super(UNet, self).__init__()
        self.nonlin = nn.ReLU(inplace=True)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.init_filters = init_filters
        self.dropout_rate = dropout_rate

        self.enc0 = Encoder(in_channels=self.input_channels, mid_channels=self.init_filters, out_channels=self.init_filters*2, nonlin=self.nonlin, downconv=False, dropout_rate=self.dropout_rate)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.enc1 = Encoder(in_channels=self.init_filters*2, mid_channels=self.init_filters*2, out_channels=self.init_filters*4, nonlin=self.nonlin, downconv=False, dropout_rate=self.dropout_rate)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.enc2 = Encoder(in_channels=self.init_filters*4, mid_channels=self.init_filters*4, out_channels=self.init_filters*8, nonlin=self.nonlin, downconv=False, dropout_rate=self.dropout_rate)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.bottom = Bottom(in_channels=self.init_filters*8, mid_channels=self.init_filters*8, out_channels=self.init_filters*16, downconv=False, nonlin=self.nonlin, dropout_rate=self.dropout_rate)

        self.dec2 = Decoder(in_channels=self.init_filters*(8 + 16), out_channels=self.init_filters*8, nonlin=self.nonlin, dropout_rate=self.dropout_rate)
        self.dec1 = Decoder(in_channels=self.init_filters*(4 + 8), out_channels=self.init_filters*4, nonlin=self.nonlin, dropout_rate=self.dropout_rate)
        self.dec0 = Decoder(in_channels=self.init_filters*(2 + 4), out_channels=self.init_filters*2, nonlin=self.nonlin, dropout_rate=self.dropout_rate, upconv = False)
        
        self.last = nn.Conv3d(in_channels=self.init_filters*2, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.weight_init()

    def forward(self, x):
        # Encoding.
        x0 = self.enc0(x)
        p0 = self.maxpool1(x0)
        x1 = self.enc1(p0)
        p1 = self.maxpool2(x1)
        x2 = self.enc2(p1)
        p2 = self.maxpool3(x2)

        # Bottom.
        x = self.bottom(p2)

        # Decoding.
        x = torch.cat((x2, x), 1)
        x = self.dec2(x)
        x = torch.cat((x1, x), 1)
        x = self.dec1(x)
        x = torch.cat((x0, x), 1)
        x = self.dec0(x)

        x = self.last(x)
        return x

    def weight_init(self):
      self.param_count_G = 0
      for module in self.modules():
        if (isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear)):
          init.orthogonal_(module.weight)
          self.param_count_G += sum([p.data.nelement() for p in module.parameters()])
      print("{} params initialized for model.".format(self.param_count_G))

    def n_params(self):
      return self.param_count_G

