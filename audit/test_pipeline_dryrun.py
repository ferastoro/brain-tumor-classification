import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

# 1. Test Custom CNN
class BrainCustomCNN(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model_cnn = BrainCustomCNN()
x = torch.randn(2, 3, 224, 224)
out_cnn = model_cnn(x)
print("Custom CNN output shape:", out_cnn.shape)
total_params_cnn = sum(p.numel() for p in model_cnn.parameters())
print("Custom CNN parameters:", total_params_cnn)

# 2. Test EfficientNet-B0
effnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
in_features = effnet.classifier[1].in_features
effnet.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, 4)
)
out_eff = effnet(x)
print("EfficientNet output shape:", out_eff.shape)
total_params_eff = sum(p.numel() for p in effnet.parameters())
print("EfficientNet parameters:", total_params_eff)

# 3. Test Grad-CAM hook on EfficientNet-B0
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Global average pool of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.squeeze().cpu().numpy()

gradcam = GradCAM(effnet, effnet.features[-1])
cam = gradcam.generate_heatmap(x[0:1])
print("Grad-CAM generated heatmap shape:", cam.shape)
print("All model and Grad-CAM tests passed successfully!")
