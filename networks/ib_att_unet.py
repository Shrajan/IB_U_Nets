import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from networks.attention_unet import (weights_init_kaiming, weights_init_normal, weights_init_orthogonal, 
                            weights_init_xavier, init_weights,_GridAttentionBlockND, GridAttentionBlock3D, 
                            UnetConv3, UnetGridGatingSignal3, UnetUp3, UnetUp3_all )
from networks.IB_3D_Kernels import IB_filters_3D

class IB_Block(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1)):
        super(IB_Block, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv4 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv4 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

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
#---------------------------3D IB ATTENTION UNET k=5-----------------------------#
####################################################################################

class IB_Attention_UNet_k5(nn.Module):

    def __init__(self, input_channels = 1, output_channels =1,  init_filters = 32, is_deconv=True, nonlocal_mode='concatenation', 
                       device_id=0, attention_dsample=(2,2,2), is_batchnorm=True):
        super(IB_Attention_UNet_k5, self).__init__()
        self.is_deconv = is_deconv
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.nonlocal_mode = nonlocal_mode
        self.init_filters =  init_filters
        self.is_batchnorm = is_batchnorm
        self.device = torch.device('cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu')

        self.conv_On_filters = IB_filters_3D(radius=2.0, gamma=2. / 3., in_channels= init_filters, 
                                                    out_channels= init_filters, off=False).to(self.device)
        self.conv_Off_filters = IB_filters_3D(radius=2.0, gamma=2. / 3., in_channels= init_filters, 
                                                    out_channels= init_filters, off=True).to(self.device)

        filters = [ self.init_filters,  self.init_filters *2,  self.init_filters * 4,  self.init_filters * 8,  self.init_filters * 16]

        # downsampling
        self.conv1 = UnetConv3(self.input_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = IB_Block(filters[0], filters[0], self.is_batchnorm) 
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)
        self.gating = UnetGridGatingSignal3(filters[4], filters[3], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = GridAttentionBlock3D(in_channels=filters[1], gating_channels=filters[3],
                                                    inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode)
        self.attentionblock3 = GridAttentionBlock3D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode)
        self.attentionblock4 = GridAttentionBlock3D(in_channels=filters[3], gating_channels=filters[3],
                                                    inter_channels=filters[3], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode)

        # upsampling
        self.up_concat4 = UnetUp3_all(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = UnetUp3_all(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = UnetUp3_all(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = UnetUp3_all(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], self.output_channels, 1)

        self.param_count_G = 0
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
                self.param_count_G += sum([p.data.nelement() for p in m.parameters()])
            elif isinstance(m, nn.Linear):
                self.param_count_G += sum([p.data.nelement() for p in m.parameters()])
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
        print("{} params initialized for model.".format(self.param_count_G))
            

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        # On and Off surround modulation
        sm_on = self.surround_modulation_DoG_on(conv1)
        sm_off = self.surround_modulation_DoG_off(conv1)

        conv2 = self.conv2(maxpool1, sm_on, sm_off)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        g_conv3, att3 = self.attentionblock3(conv3, gating)
        g_conv2, att2 = self.attentionblock2(conv2, gating)

        # Upscaling Part (Decoder)
        up4 = self.up_concat4(g_conv4, center)
        up3 = self.up_concat3(g_conv3, up4)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    def surround_modulation_DoG_on(self, input):
        output = F.conv3d(input, weight=self.conv_On_filters, stride=2, padding=2)
        return F.relu(output, inplace=True)

    def surround_modulation_DoG_off(self, input):
        output = F.conv3d(input, weight=self.conv_Off_filters, stride=2, padding=2)
        return F.relu(output, inplace=True)

####################################################################################
#---------------------------3D IB ATTENTION UNET k=3-----------------------------#
####################################################################################

class IB_Attention_UNet_k3(nn.Module):

    def __init__(self, input_channels = 1, output_channels =1,  init_filters = 32, is_deconv=True, nonlocal_mode='concatenation', 
                       device_id=0, attention_dsample=(2,2,2), is_batchnorm=True):
        super(IB_Attention_UNet_k3, self).__init__()
        self.is_deconv = is_deconv
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.nonlocal_mode = nonlocal_mode
        self.init_filters =  init_filters
        self.is_batchnorm = is_batchnorm
        self.device = torch.device('cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu')

        device_id = 'cuda:' + str(opt.device_id)
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

        self.conv_On_filters = IB_filters_3D(radius=1.0, gamma=1. / 2., in_channels= init_filters, 
                                                    out_channels= init_filters, off=False).to(self.device)
        self.conv_Off_filters = IB_filters_3D(radius=1.0, gamma=1. / 2., in_channels= init_filters, 
                                                    out_channels= init_filters, off=True).to(self.device)

        filters = [ self.init_filters,  self.init_filters *2,  self.init_filters * 4,  self.init_filters * 8,  self.init_filters * 16]

        # downsampling
        self.conv1 = UnetConv3(self.input_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = IB_Block(filters[0], filters[0], self.is_batchnorm) 
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)
        self.gating = UnetGridGatingSignal3(filters[4], filters[3], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = GridAttentionBlock3D(in_channels=filters[1], gating_channels=filters[3],
                                                    inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode)
        self.attentionblock3 = GridAttentionBlock3D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode)
        self.attentionblock4 = GridAttentionBlock3D(in_channels=filters[3], gating_channels=filters[3],
                                                    inter_channels=filters[3], sub_sample_factor=attention_dsample, mode=self.nonlocal_mode)

        # upsampling
        self.up_concat4 = UnetUp3_all(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = UnetUp3_all(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = UnetUp3_all(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = UnetUp3_all(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], self.output_channels, 1)

        self.param_count_G = 0
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
                self.param_count_G += sum([p.data.nelement() for p in m.parameters()])
            elif isinstance(m, nn.Linear):
                self.param_count_G += sum([p.data.nelement() for p in m.parameters()])
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
        print("{} params initialized for model.".format(self.param_count_G))
            

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        # On and Off surround modulation
        sm_on = self.surround_modulation_DoG_on(conv1)
        sm_off = self.surround_modulation_DoG_off(conv1)

        conv2 = self.conv2(maxpool1, sm_on, sm_off)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        g_conv3, att3 = self.attentionblock3(conv3, gating)
        g_conv2, att2 = self.attentionblock2(conv2, gating)

        # Upscaling Part (Decoder)
        up4 = self.up_concat4(g_conv4, center)
        up3 = self.up_concat3(g_conv3, up4)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    def surround_modulation_DoG_on(self, input):
        output = F.conv3d(input, weight=self.conv_On_filters, stride=2, padding=(1,1,1))
        return F.relu(output, inplace=True)

    def surround_modulation_DoG_off(self, input):
        output = F.conv3d(input, weight=self.conv_Off_filters, stride=2, padding=(1,1,1))
        return F.relu(output, inplace=True)

