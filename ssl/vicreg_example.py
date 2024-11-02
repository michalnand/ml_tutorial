import numpy
import torch
import matplotlib.pyplot as plt




def _off_diagonal(x):
    mask = 1.0 - torch.eye(x.shape[0], device=x.device)
    return x*mask 

# invariance loss
def loss_mse_func(xa, xb):
    return ((xa - xb)**2).mean()
    
# variance loss
def loss_std_func(x):
    eps   = 10**-6
    std_x = torch.sqrt(x.var(dim=0) + eps)
    loss  = torch.mean(torch.relu(1.0 - std_x)) 
    return loss

# covariance loss 
def loss_cov_func(x):
    x_norm = x - x.mean(dim=0)
    cov_x = (x_norm.T @ x_norm) / (x.shape[0] - 1.0)
    
    loss = _off_diagonal(cov_x).pow_(2).sum()/x.shape[1] 
    return loss



def process_training(z_initial, n_steps, k0, k1):
    # initial parameters
    z      = torch.nn.Parameter(torch.from_numpy(z_initial).clone(), requires_grad = True)

    # optimizer
    optim  = torch.optim.Adam([z], lr=0.01)

    z_all = []

    for n in range(n_steps):
        loss = k0*loss_std_func(z)
        loss+= k1*loss_cov_func(z)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        z_all.append(z[:, 0:2].detach().cpu().numpy().copy())

    z_all = numpy.array(z_all)

    print(">>> ", z_all.shape)

    return z_all



if __name__ == "__main__":

    n_points = 1024
    n_dims   = 128

    # create initial features
    z_initial = 0.02*numpy.random.randn(n_points, n_dims)

    numpy.save("results/initial", z_initial[:, 0:2])



    k0 = 1.0
    k1 = 0.0

    k0_values = [1.0, 1.0]
    k1_values = [0.0, 500.0]
    

    for n in range(len(k0_values)):
        k0 = k0_values[n]
        k1 = k1_values[n]
        # train
        z_trained = process_training(z_initial, 20, k0, k1)

        numpy.save("results/" + str(k0) + "_" + str(k1), z_trained)


    plt.plot(z_initial[:, 0], z_initial[:, 1], 'o', color='blue')
    plt.plot(z_trained[-1, :, 0], z_trained[-1, :, 1], 'o', color='red')
    plt.show()