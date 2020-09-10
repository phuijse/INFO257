import torch.nn as nn

class Lenet5(nn.Module):
    
    def __init__(self):
        super(type(self), self).__init__()
        # La entrada son imágenes de 1x32x32
        self.features = nn.Sequential(nn.Conv2d(1, 6, 5),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2),
                                      nn.Conv2d(6, 16, 5),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2))
        
        self.classifier = nn.Sequential(nn.Linear(400, 120),
                                        nn.ReLU(),
                                        nn.Linear(120, 84),
                                        nn.ReLU(),
                                        nn.Linear(84, 10))

    def forward(self, x):
        z = self.features(x)
        # Esto es de tamaño Mx16x5x5
        z = z.view(z.shape[0], -1)
        # Esto es de tamaño Mx400
        return self.classifier(z)
    
    
model = Lenet5()
print(model)
    