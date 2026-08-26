import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # A single "linear layer": takes 3 input numbers, produces 1 output number.
        # Internally this holds a weight tensor (shape [1,3]) and a bias tensor (shape [1]),
        # both randomly initialized, both marked as learnable parameters automatically.
       #self.layer1 = nn.Linear(in_features=3, out_features=1)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(in_features=3,out_features=4)
        self.layer3 = nn.Linear(in_features=4,out_features=1)
    def forward(self, x):
        # Defines what happens when data flows through this network.
        # Here: just pass x through the one linear layer.
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        #x = self.activation(x)
       #x = self.layer1(x)
        return x

#initiate the network
model = TinyNet()
print("Model structure: \n", model)

#inspect its learnable parameters
print("\n Parameters: ")
for name,param in model.named_parameters():
    print(f" {name}: shape={param.shape}, requires_grad = {param.requires_grad}")

#run a fake input through it - 1 Sample, 3 features
fake_input = torch.tensor([[1.0,2.0,3.0]])
output = model(fake_input)
print("\n INput: ", fake_input)
print("Output: ", output)