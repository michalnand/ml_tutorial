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

        # orthogonal weight init
        for i in range(len(self.model)):
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 0.5)
                torch.nn.init.zeros_(self.model[i].bias)

        # smaller weight init on last layer
        torch.nn.init.orthogonal_(self.model[-1].weight, 0.01)
        torch.nn.init.zeros_(self.model[-1].bias)

    def forward(self, x): 
        return self.model(x) 
     

