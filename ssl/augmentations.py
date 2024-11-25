import torch

'''
random apply or not apply augmentation
'''
def aug_random_apply(x, aug_func, p = 0.5): 
    c = (torch.rand((x.shape[0], 1, 1, 1)) < p).float()
    return (1.0 - c)*x + c*aug_func(x) 

'''
random gaussian noise
'''
def aug_noise(x, noise_max = 0.5):          
    alpha = noise_max*torch.rand((x.shape[0], 1, 1, 1), device=x.device)
    return (1.0 - alpha)*x + alpha*torch.randn_like(x), alpha[:, :, 0, 0]
   
'''
negative
'''
def aug_inverse(x): 
    return 1.0 - x

'''
random pixel dropout
''' 
def aug_dropout(x, p = 0.05): 
    mask = (torch.rand_like(x) > p).float()
    return x*mask

def aug_stack(x):
    y = aug_random_apply(x, aug_noise)
    y = aug_random_apply(y, aug_dropout)

    return y

