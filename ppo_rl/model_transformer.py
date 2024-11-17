import torch

'''
    multi layer perceptron for nonlinear transform,
    with optional residual connection
'''
class ModelMLP(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, residual = False, init_gain = 0.5):
        super(ModelMLP, self).__init__()

        self.residual = residual

        self.lin0 = torch.nn.Linear(n_inputs, n_hidden)
        self.act0 = torch.nn.SiLU()
        self.lin1 = torch.nn.Linear(n_hidden, n_outputs)

        torch.nn.init.orthogonal_(self.lin0.weight, 0.5)
        torch.nn.init.zeros_(self.lin0.bias)
        torch.nn.init.orthogonal_(self.lin1.weight, init_gain)
        torch.nn.init.zeros_(self.lin1.bias)

    def forward(self, x):
        y = self.lin0(x)
        y = self.act0(y)
        y = self.lin1(y)

        if self.residual:
            y = y + x

        return y
    
'''
    basic single head attention mechanism
    attn = q@k.T / scaling
    attn = softmax(attn)
    y    = attn@v
'''
class ModelAttention(torch.nn.Module):
    def __init__(self, n_features):
        super(ModelAttention, self).__init__()

        self.lin_q = torch.nn.Linear(n_features, n_features)
        self.lin_k = torch.nn.Linear(n_features, n_features)
        self.lin_v = torch.nn.Linear(n_features, n_features)

        torch.nn.init.xavier_uniform_(self.lin_q.weight)
        torch.nn.init.zeros_(self.lin_q.bias)
        torch.nn.init.xavier_uniform_(self.lin_k.weight)
        torch.nn.init.zeros_(self.lin_k.bias)
        torch.nn.init.xavier_uniform_(self.lin_v.weight)
        torch.nn.init.zeros_(self.lin_v.bias)
        
    def forward(self, x):
        q = self.lin_q(x)
        k = self.lin_k(x)
        v = self.lin_v(x)

        scaling = (q.shape[-1]**0.5)
        attn = torch.bmm(q, torch.transpose(k, 1, 2))/scaling

        attn = torch.softmax(attn, dim=-1)

        y = torch.bmm(attn, v)

        return y + x, attn

class ModelTransformer(torch.nn.Module):
    def __init__(self, input_shape, n_actions, n_features = 128):
        super(ModelTransformer, self).__init__()

        n_inputs = input_shape[1]   

        # small transformer model
        self.mlp0   = ModelMLP(n_inputs, 2*n_features, n_features, False)
        self.attn0  = ModelAttention(n_features) 
        self.mlp1   = ModelMLP(n_features, 2*n_features, n_features, True)
        self.attn1  = ModelAttention(n_features) 
        self.mlp2   = ModelMLP(n_features, 2*n_features, n_features, True)
        self.attn2  = ModelAttention(n_features) 
       
        # two output heads
        self.actor  = ModelMLP(n_features, 2*n_features, n_actions, False, 0.01)
        self.critic = ModelMLP(n_features, 2*n_features, 1, False, 0.1)

    '''
        state shape    = (n_batch, n_seq, n_inputs)
        logits shape   = (n_batch, n_seq, n_actions)
        values shape   = (n_batch, n_seq, 1)
    '''  
    def forward(self, state):
        # obtain features
        z = self.mlp0(state)
        z, attn0 = self.attn0(z)
        z = self.mlp1(z)
        z, attn1 = self.attn1(z)
        z = self.mlp2(z)
        z, attn2 = self.attn2(z)
       
        # obtain actor and critic outputs
        logits = self.actor(z)
        value  = self.critic(z)

        return logits, value, [attn0, attn1, attn2]