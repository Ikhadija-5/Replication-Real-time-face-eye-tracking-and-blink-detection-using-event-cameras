import torch
import torch.nn as nn
import torch.optim as optim


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GRU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=out_channels, num_layers=1, batch_first=True)

    def forward(self, x):
        # Convert input shape (batch, channels, height, width) to (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1)
        batch_size, height, width, channels = x.size()

        # Flatten height and width dimensions
        x = x.view(batch_size, height * width, channels)

        # Apply GRU
        out, _ = self.gru(x)

        # Reshape back to (batch, height, width, channels)
        out = out.view(batch_size, height, width, -1)
        out = out.permute(0, 3, 1, 2)

        return out

class RouteBlock(nn.Module):
    def __init__(self, out_channels):
        super(RouteBlock, self).__init__()
        self.out_channels = out_channels

    def forward(self, x):
        return x[:, :self.out_channels, :, :]


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class YOLOBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(YOLOBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, 512, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class GR_YOLO(nn.Module):
    def __init__(self, in_channels=1, num_classes=21):
        super(GR_YOLO, self).__init__()
        self.conv0 = ConvBlock(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool7 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.conv8 = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool9 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.conv10 = ConvBlock(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool11 = nn.MaxPool2d(kernel_size=2,stride = 1)
        self.conv12 = ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv13 = ConvBlock(1024, 256, kernel_size=1, stride=1, padding=1)
        self.GRU14 = GRU(256,256)
        self.conv15 = ConvBlock(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv16 = ConvBlock(512, 21, kernel_size=1, stride=1, padding=1)
        self.YOLO17 = YOLOBlock(21, 7)
        self.route18 = RouteBlock(256)
        self.conv19 = ConvBlock(256, 128, kernel_size=1, stride=1, padding=1)
        self.upsampling20 = UpsampleBlock(128, 128)
        self.route21 = RouteBlock(384)
        self.conv22 = ConvBlock(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv23 = ConvBlock(256, 21, kernel_size=1, stride=1, padding=1)
        self.YOLO24 = YOLOBlock(21, 7)


    def forward(self, x):
        # Define forward pass
        x = self.conv0(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.pool7(x)
        x = self.conv8(x)
        x = self.pool9(x)
        x = self.conv10(x)
        x = self.pool11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.GRU14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.YOLO17(x)
        print(x.shape)
        x = self.route18(x)
        x = self.conv19(x)
        x = self.upsampling20(x)
        x = self.route21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.YOLO24(x)
        return x
