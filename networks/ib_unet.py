import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from networks.unet import Conv3x3, Bottom, TConv, Encoder, Decoder, TConvHW
from networks.IB_3D_Kernels import IB_filters_3D

class IB_Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, nonlin, downconv=False, dropout_rate=0.0):
        super(IB_Block, self).__init__()
        self.nonlin = nonlin
        self.conv1 = Conv3x3(in_channels=in_channels, out_channels=out_channels, nonlin=nonlin, downconv=downconv, dropout_rate=dropout_rate)
        self.conv2 = Conv3x3(in_channels=in_channels, out_channels=out_channels, nonlin=nonlin, downconv=downconv, dropout_rate=dropout_rate)
        self.conv3 = Conv3x3(in_channels=out_channels, out_channels=out_channels, nonlin=nonlin, downconv=downconv, dropout_rate=dropout_rate)
        self.conv4 = Conv3x3(in_channels=out_channels, out_channels=out_channels, nonlin=nonlin, downconv=downconv, dropout_rate=dropout_rate)

    def forward(self, inputs, sm_on, sm_off):
        output1 = self.conv1(inputs)
        output2 = self.conv2(inputs)

        input3 = sm_on + output1
        input4 = sm_off + output2

        output3 = self.conv3(input3)
        output4 = self.conv4(input4)

        outputs = torch.cat((output3, output4), 1)
        return outputs

####################################################################################
#--------------------------------3D IB UNET k=5----------------------------------#
####################################################################################

class IB_UNet_k5(nn.Module):
    """
    Classical 3D UNet with IB filters k=5.
    """
    def __init__(self, input_channels=1, output_channels=1, init_filters=16, dropout_rate=0.5, device_id=0):
        super(IB_UNet_k5, self).__init__()
        self.nonlin = nn.ReLU(inplace=True)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.init_filters = init_filters
        self.dropout_rate = dropout_rate
        self.device = torch.device('cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu')

        self.conv_On_filters = IB_filters_3D(radius=2.0, gamma=2. / 3., in_channels=self. init_filters*2, 
                                                    out_channels=self. init_filters*2, off=False).to(self.device)
        self.conv_Off_filters = IB_filters_3D(radius=2.0, gamma=2. / 3., in_channels=self. init_filters*2, 
                                                    out_channels=self. init_filters*2, off=True).to(self.device)

        self.enc0 = Encoder(in_channels=self.input_channels, mid_channels=self. init_filters, out_channels=self. init_filters*2, nonlin=self.nonlin, downconv=False, dropout_rate=self.dropout_rate)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.enc1 = IB_Block(in_channels=self. init_filters*2, out_channels=self. init_filters*2, nonlin=self.nonlin, downconv=False, dropout_rate=self.dropout_rate)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.enc2 = Encoder(in_channels=self. init_filters*4, mid_channels=self. init_filters*4, out_channels=self. init_filters*8, nonlin=self.nonlin, downconv=False, dropout_rate=self.dropout_rate)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.bottom = Bottom(in_channels=self. init_filters*8, mid_channels=self. init_filters*8, out_channels=self. init_filters*16, downconv=False, nonlin=self.nonlin, dropout_rate=self.dropout_rate)

        self.dec2 = Decoder(in_channels=self. init_filters*(8 + 16), out_channels=self. init_filters*8, nonlin=self.nonlin, dropout_rate=self.dropout_rate)
        self.dec1 = Decoder(in_channels=self. init_filters*(4 + 8), out_channels=self. init_filters*4, nonlin=self.nonlin, dropout_rate=self.dropout_rate)
        self.dec0 = Decoder(in_channels=self. init_filters*(2 + 4), out_channels=self. init_filters*2, nonlin=self.nonlin, dropout_rate=self.dropout_rate, upconv = False)
        
        self.last = nn.Conv3d(in_channels=self. init_filters*2, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.weight_init()

    def forward(self, x):
        # Encoding.
        x0 = self.enc0(x)
        p0 = self.maxpool1(x0)

        # On and Off surround modulation
        sm_on = self.surround_modulation_DoG_on(x0)
        sm_off = self.surround_modulation_DoG_off(x0)
        
        x1 = self.enc1(p0, sm_on, sm_off)
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

    def surround_modulation_DoG_on(self, input):
        output = F.conv3d(input, weight=self.conv_On_filters, stride=2, padding=2)
        return F.relu(output, inplace=True)

    def surround_modulation_DoG_off(self, input):
        output = F.conv3d(input, weight=self.conv_Off_filters, stride=2, padding=2)
        return F.relu(output, inplace=True)

####################################################################################
#--------------------------------3D IB UNET k=3----------------------------------#
####################################################################################

class IB_UNet_k3(nn.Module):
    """
    Classical 3D UNet with IB filters k=3.
    """
    def __init__(self, input_channels=1, output_channels=1, init_filters=16, dropout_rate=0.5, device_id=0):
        super(IB_UNet_k3, self).__init__()
        self.nonlin = nn.ReLU(inplace=True)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.init_filters = init_filters
        self.dropout_rate = dropout_rate
        self.device = torch.device('cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu')

        self.conv_On_filters = IB_filters_3D(radius=1.0, gamma=1. / 2., in_channels=self. init_filters*2, 
                                                    out_channels=self. init_filters*2, off=False).to(self.device)
        self.conv_Off_filters = IB_filters_3D(radius=1.0, gamma=1. / 2., in_channels=self. init_filters*2, 
                                                    out_channels=self. init_filters*2, off=True).to(self.device)

        self.enc0 = Encoder(in_channels=self.input_channels, mid_channels=self. init_filters, out_channels=self. init_filters*2, nonlin=self.nonlin, downconv=False, dropout_rate=self.dropout_rate)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.enc1 = IB_Block(in_channels=self. init_filters*2, out_channels=self. init_filters*2, nonlin=self.nonlin, downconv=False, dropout_rate=self.dropout_rate)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.enc2 = Encoder(in_channels=self. init_filters*4, mid_channels=self. init_filters*4, out_channels=self. init_filters*8, nonlin=self.nonlin, downconv=False, dropout_rate=self.dropout_rate)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.bottom = Bottom(in_channels=self. init_filters*8, mid_channels=self. init_filters*8, out_channels=self. init_filters*16, downconv=False, nonlin=self.nonlin, dropout_rate=self.dropout_rate)

        self.dec2 = Decoder(in_channels=self. init_filters*(8 + 16), out_channels=self. init_filters*8, nonlin=self.nonlin, dropout_rate=self.dropout_rate)
        self.dec1 = Decoder(in_channels=self. init_filters*(4 + 8), out_channels=self. init_filters*4, nonlin=self.nonlin, dropout_rate=self.dropout_rate)
        self.dec0 = Decoder(in_channels=self. init_filters*(2 + 4), out_channels=self. init_filters*2, nonlin=self.nonlin, dropout_rate=self.dropout_rate, upconv = False)
        
        self.last = nn.Conv3d(in_channels=self. init_filters*2, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.weight_init()

    def forward(self, x):
        # Encoding.
        x0 = self.enc0(x)
        p0 = self.maxpool1(x0)

        # On and Off surround modulation
        sm_on = self.surround_modulation_DoG_on(x0)
        sm_off = self.surround_modulation_DoG_off(x0)
        
        x1 = self.enc1(p0, sm_on, sm_off)
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

    def surround_modulation_DoG_on(self, input):
        output = F.conv3d(input, weight=self.conv_On_filters, stride=2, padding=1)
        return F.relu(output, inplace=True)

    def surround_modulation_DoG_off(self, input):
        output = F.conv3d(input, weight=self.conv_Off_filters, stride=2, padding=1)
        return F.relu(output, inplace=True)
