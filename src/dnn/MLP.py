import torch
import torch.nn as nn
import torch.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, output_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out


mlp = MLP(1000, 3)
print(mlp)


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class MLP2(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, drop_out=0.1):
        super(MLP2, self).__init__()
        module_list = []
        for i in range(num_layers - 1):
            module_list.append(nn.Linear(input_dim, hidden_dim))
            module_list.append(nn.BatchNorm1d(hidden_dim))
            module_list.append(nn.ReLU())
            input_dim = hidden_dim
            # module_list.append(nn.Dropout(p=drop_out))
        module_list.append(nn.Linear(hidden_dim, output_dim))
        self.models = nn.Sequential(*module_list)

    def forward(self, x):
        return self.models(x)









