import torch

'''
small convolutional NN model
'''
class Model(torch.nn.Module):
    def __init__(self, input_shape, n_features):
        super(Model, self).__init__()

        fc_size = (input_shape[1]//4) * (input_shape[2]//4)

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(), 

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),    
            
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),   
            
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),  

            torch.nn.Flatten(), 
            torch.nn.Linear(fc_size*128, n_features)
        ) 

        self.model_noise_pred = torch.nn.Sequential(
            torch.nn.Linear(n_features, 2*n_features),
            torch.nn.SiLU(),
            torch.nn.Linear(2*n_features, n_features),
            torch.nn.SiLU(),
            torch.nn.Linear(n_features, 1)
        )


        # orthogonal weight init
        for i in range(len(self.model)):
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 0.5)
                torch.nn.init.zeros_(self.model[i].bias)

        # smaller weight init on last layer
        torch.nn.init.orthogonal_(self.model[-1].weight, 0.01)
        torch.nn.init.zeros_(self.model[-1].bias)


        for i in range(len(self.model_noise_pred)):
            if hasattr(self.model_noise_pred[i], "weight"):
                torch.nn.init.orthogonal_(self.model_noise_pred[i].weight, 0.5)
                torch.nn.init.zeros_(self.model_noise_pred[i].bias)

        torch.nn.init.orthogonal_(self.model_noise_pred[-1].weight, 0.01)
        torch.nn.init.zeros_(self.model_noise_pred[-1].bias)


    def forward(self, x): 
        z = self.model(x)
        n = self.model_noise_pred(z)
        return z, n
     




