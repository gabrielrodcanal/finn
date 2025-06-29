import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

class ZF(nn.Module):
    def __init__(self):
        super(ZF, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=2)  # Reduced channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1)  # Reduced channels, stride set to 1
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1)  # Reduced channels
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1)  # Reduced channels
        self.conv5 = nn.Conv2d(8, 4, kernel_size=3, padding=1, stride=1)  # Reduced channels
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc6 = nn.Conv2d(4, 16, kernel_size=1)  # Reduced units
        self.fc7 = nn.Linear(16 * 1 * 1, 16)        # Adjusted input size
        self.fc8 = nn.Linear(16, 4)                 # Reduced classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = F.relu(self.fc6(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x


model = ZF()

# Compile with mlir
module = torch_mlir.compile(model, torch.ones(1, 3, 16, 16),
                            output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
print(module)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

torch.save(model.state_dict(), 'zf_model.pth')


import torch.onnx

# Switch model to evaluation mode
model.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 3, 16, 16)

# Export to ONNX
torch.onnx.export(
    model,                      # model being run
    dummy_input,                # model input (or a tuple for multiple inputs)
    "zf_model.onnx",            # where to save the model
    export_params=True,         # store the trained parameter weights inside the model file
    opset_version=11,           # the ONNX version to export the model to
    do_constant_folding=True,   # whether to execute constant folding for optimization
    input_names=['input'],      # input tensor names
    output_names=['output'],    # output tensor names
    #dynamic_axes={              # enable variable batch size
    #    'input': {0: 'batch_size'},
    #    'output': {0: 'batch_size'}
    #}
)

print("ONNX model exported as 'zf_model.onnx'")
