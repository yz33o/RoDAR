import torch
import torch.nn as nn


class IDM(nn.Module):
    def __init__(self, params_value=None, params_trainable=None, device=torch.device("cpu")):
        super(IDM, self).__init__()
        if params_value is None:
            params_value = dict(v0=30, T=1, a=1, b=1.5, delta=4, s0=2, )
        if params_trainable is None:
            params_trainable = dict(v0=False, T=False, a=False, b=False, delta=False, s0=False, )
        self.torch_params = dict()
        for k, v in params_value.items():
            if params_trainable[k] is True:
                self.torch_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device),
                                                               requires_grad=True)
                self.torch_params[k].retain_grad()
            else:
                self.torch_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device),
                                                          requires_grad=False)
        self.veh_length = 0.

    def forward(self, x):
        dx = x[:, 0]
        dv = x[:, 1]
        v = x[:, 2]

        s0 = self.torch_params['s0']
        v0 = self.torch_params['v0']
        T = self.torch_params['T']
        a = self.torch_params['a']
        b = self.torch_params['b']
        delta = self.torch_params['delta']

        s_star = s0 + torch.clamp_min(T * v - v * dv / (2 * torch.sqrt(a * b)), 0)
        acc = a * (1 - torch.pow(v / v0, delta) - torch.pow(s_star / (dx - self.veh_length), 2))
        return acc.view(-1, 1)


class NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(128, 128), activation='tanh', learning_rate=0.0001, device=torch.device('cpu')):
        super(NN, self).__init__()
        self.device = device
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action = nn.Linear(last_dim, output_dim)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = x.to(self.device)
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        x = self.action(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_size=(128, 128), activation='relu', device=torch.device('cpu')):
        super(Discriminator, self).__init__()
        self.device = device
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        self.layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_size:
            self.layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.output = nn.Linear(last_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.sigmoid(self.output(x))
        return x


class KnowledgeNNWithGAN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(128, 128), activation='relu', params_value=None,
                 params_trainable=None, device=torch.device('cpu'), learning_rate=0.0001):
        super(KnowledgeNNWithGAN, self).__init__()

        self.physics_model = IDM(params_value, params_trainable, device)
        self.generator = NN(input_dim, output_dim, hidden_size, activation, device=device)
        self.discriminator = Discriminator(output_dim, hidden_size, activation, device=device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        self.optimizer = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.discriminator.parameters()),
            lr=learning_rate
        )

    def forward(self, obs):
        phy = self.physics_model(obs)
        res = self.generator(obs)
        return phy + res

    def train_discriminator(self, real_data, fake_data):
        real_labels = torch.ones(real_data.size(0), 1).to(real_data.device)
        fake_labels = torch.zeros(fake_data.size(0), 1).to(fake_data.device)

        real_output = self.discriminator(real_data)
        fake_output = self.discriminator(fake_data)

        d_loss = nn.BCELoss()(real_output, real_labels) + nn.BCELoss()(fake_output, fake_labels)
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        return d_loss.item()

    def train_generator(self, fake_data):
        labels = torch.ones(fake_data.size(0), 1).to(fake_data.device)
        output = self.discriminator(fake_data)

        g_loss = nn.BCELoss()(output, labels)
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss.item()
