import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

def PoissonGen(inp, rescale_fac=0.5):
    rand_inp = torch.rand_like(inp)
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (1.0 + torch.abs(input))**2
        return grad

class SNN_VGG9_BNTT_DCLS(nn.Module):
    def __init__(self, in_channels, timesteps, leak_mem, img_size, num_cls):
        super(SNN_VGG9_BNTT_DCLS, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.timesteps

        print(">>>>>>>>>>>>>>>>> VGG9 with DCLS >>>>>>>>>>>>>>>>>>>>>>>")
        print("***** time step per batchnorm".format(self.batch_num))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.offset_conv1 = nn.Conv2d(in_channels, 18, kernel_size=3, stride=1, padding=1)
        self.conv1 = DeformConv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.offset_conv2 = nn.Conv2d(64, 18, kernel_size=3, stride=1, padding=1)
        self.conv2 = DeformConv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.offset_conv3 = nn.Conv2d(128, 18, kernel_size=3, stride=1, padding=1)
        self.conv3 = DeformConv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(256 * (img_size // 8) * (img_size // 8), 1024, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

    def forward(self, inp):
        batch_size = inp.size(0)
        device = inp.device  
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size, device=device)
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2, device=device)
        mem_conv3 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4, device=device)
        mem_conv4 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4, device=device)

        mem_fc1 = torch.zeros(batch_size, 1024, device=device)
        mem_fc2 = torch.zeros(batch_size, self.num_cls, device=device)

        for t in range(self.timesteps):
            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            offset1 = self.offset_conv1(out_prev)
            mem_conv1 = self.leak_mem * mem_conv1 + self.bntt1[t](self.conv1(out_prev, offset1))
            mem_thr = (mem_conv1 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()
            out_prev = self.pool1(out_prev)

            offset2 = self.offset_conv2(out_prev)
            mem_conv2 = self.leak_mem * mem_conv2 + self.bntt2[t](self.conv2(out_prev, offset2))
            mem_thr = (mem_conv2 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()
            out_prev = self.pool2(out_prev)

            offset3 = self.offset_conv3(out_prev)
            mem_conv3 = self.leak_mem * mem_conv3 + self.bntt3[t](self.conv3(out_prev, offset3))
            mem_thr = (mem_conv3 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()

            mem_conv4 = self.leak_mem * mem_conv4 + self.bntt4[t](self.conv4(out_prev))
            mem_thr = (mem_conv4 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()
            out_prev = self.pool3(out_prev)

            out_prev = out_prev.view(out_prev.size(0), -1)

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()

            mem_fc2 = self.leak_mem * mem_fc2 + self.fc2(out_prev)

        return mem_fc2

class SNN_VGG11_BNTT_DCLS(nn.Module):
    def __init__(self, in_channels, timesteps, leak_mem, img_size, num_cls):
        super(SNN_VGG11_BNTT_DCLS, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.timesteps

        print(">>>>>>>>>>>>>>>>> VGG11 with DCLS >>>>>>>>>>>>>>>>>>>>>>>")
        print("***** time step per batchnorm".format(self.batch_num))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.offset_conv1 = nn.Conv2d(in_channels, 18, kernel_size=3, stride=1, padding=1)
        self.conv1 = DeformConv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.offset_conv2 = nn.Conv2d(64, 18, kernel_size=3, stride=1, padding=1)
        self.conv2 = DeformConv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.offset_conv3 = nn.Conv2d(128, 18, kernel_size=3, stride=1, padding=1)
        self.conv3 = DeformConv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.offset_conv5 = nn.Conv2d(256, 18, kernel_size=3, stride=1, padding=1)
        self.conv5 = DeformConv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool4 = nn.AvgPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt8 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 4096, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(4096, self.num_cls, bias=bias_flag)

    def forward(self, inp):
        batch_size = inp.size(0)
        device = inp.device  

        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size, device=device)
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2, device=device)
        mem_conv3 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4, device=device)
        mem_conv4 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4, device=device)
        mem_conv5 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8, device=device)
        mem_conv6 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8, device=device)  # 新增初始化

        mem_fc1 = torch.zeros(batch_size, 4096, device=device)
        mem_fc2 = torch.zeros(batch_size, self.num_cls, device=device)

        for t in range(self.timesteps):
            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            offset1 = self.offset_conv1(out_prev)
            mem_conv1 = self.leak_mem * mem_conv1 + self.bntt1[t](self.conv1(out_prev, offset1))
            mem_thr = (mem_conv1 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()
            out_prev = self.pool1(out_prev)

            offset2 = self.offset_conv2(out_prev)
            mem_conv2 = self.leak_mem * mem_conv2 + self.bntt2[t](self.conv2(out_prev, offset2))
            mem_thr = (mem_conv2 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()
            out_prev = self.pool2(out_prev)

            offset3 = self.offset_conv3(out_prev)
            mem_conv3 = self.leak_mem * mem_conv3 + self.bntt3[t](self.conv3(out_prev, offset3))
            mem_thr = (mem_conv3 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()

            mem_conv4 = self.leak_mem * mem_conv4 + self.bntt4[t](self.conv4(out_prev))
            mem_thr = (mem_conv4 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()
            out_prev = self.pool3(out_prev)

            offset5 = self.offset_conv5(out_prev)
            mem_conv5 = self.leak_mem * mem_conv5 + self.bntt5[t](self.conv5(out_prev, offset5))
            mem_thr = (mem_conv5 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()

            mem_conv6 = self.leak_mem * mem_conv6 + self.bntt6[t](self.conv6(out_prev))
            mem_thr = (mem_conv6 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()
            out_prev = self.pool4(out_prev)

            out_prev = self.pool5(out_prev)
            out_prev = out_prev.view(out_prev.size(0), -1)

            self.dropout = nn.Dropout(0.5) 
            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_fc1 = self.dropout(mem_fc1)  
            mem_thr = (mem_fc1 / 1.0) - 1.0
            out = self.spike_fn(mem_thr)
            out_prev = out.clone()

            mem_fc2 = self.leak_mem * mem_fc2 + self.fc2(out_prev)

        return mem_fc2

if __name__ == '__main__':
    timesteps = 20
    leak_mem = 0.95
    img_size = 28
    num_cls = 5
    batch_size = 3

    model = SNN_VGG11_BNTT_DCLS(timesteps=timesteps,
                          leak_mem=leak_mem,
                          img_size=img_size,
                          in_channels=3,
                          num_cls=num_cls)

    random_images = torch.rand(batch_size, 3, img_size, img_size)

    with torch.no_grad(): 
        output = model(random_images)

    print(f"Output shape: {output.shape}")
    # print("Output (predictions):")
    print(output)