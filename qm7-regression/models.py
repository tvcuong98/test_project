import torch

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(276, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 1)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x

class ElasticNet(torch.nn.Module):
    def __init__(self, alpha=1., beta=0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(276, 1)
        self.alpha = alpha
        self.beta = beta

    def _weight_decay(self):
        for k, v in self.named_parameters():
            if 'weight' in k:
                v.grad.add_(torch.sign(v.data), alpha=self.alpha)
                v.grad.add_(v, alpha=self.beta)

    def forward(self, x):
        x = self.act(self.fc1(x))
        if self.training:
            self._weight_decay()
        return x