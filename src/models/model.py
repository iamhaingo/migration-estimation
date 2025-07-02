import torch
import torch.nn as nn
import torchvision.models as models


class MultiBranchCNN(nn.Module):
    def __init__(self, output_dim=1):
        super(MultiBranchCNN, self).__init__()

        # === RGB branch using pretrained ResNet (adjust for 330x330) ===
        self.rgb_model = models.resnet50(pretrained=True)  # finetune ResNet18
        self.rgb_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # default
        self.rgb_model.fc = nn.Identity()  # remove final classification head
        rgb_output_dim = 2048

        # === VIIRS branch: small CNN from scratch for 1-channel input ===
        self.viirs_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # -> (32,)
        )
        viirs_output_dim = 32

        # === Final MLP fusion head ===
        total_input_dim = rgb_output_dim + viirs_output_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, output_dim),  # output_dim=1 for regression
        )

    def forward(self, rgb_img, viirs_img):
        rgb_feat = self.rgb_model(rgb_img)  # shape: (batch_size, 512)
        viirs_feat = self.viirs_cnn(viirs_img)  # shape: (batch_size, 32)
        x = torch.cat([rgb_feat, viirs_feat], dim=1)  # shape: (batch_size, 512+32)
        output = self.mlp(x)  # shape: (batch_size, 1)
        return output.squeeze(1)  # (batch,) for regression
