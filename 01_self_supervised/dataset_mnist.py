import torch
import torchvision
import numpy


class DatasetMnist:
    def __init__(self, train = True):
        self.input_shape = (1, 28, 28)
        self.data = torchvision.datasets.MNIST("./data/", train=train, download=True)
        

        self.class_indices = []

        for i in range(len(self.data)):
            self.class_indices.append([])

        for i in range(len(self.data)):
            _, class_id  = self.data[i]

            self.class_indices[int(class_id)].append(i)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y    = self.data[index]
        x       = numpy.array(x).astype(numpy.float32)
        x       = x/255.0


        return torch.from_numpy(x).unsqueeze(0), y
    
    def get_batch(self, batch_size):

        x = torch.zeros((batch_size, ) + self.input_shape, dtype=torch.float32)
        y = torch.zeros((batch_size, ))
        indices = numpy.random.randint(0, len(self.data), batch_size)

        for i in range(batch_size):
            x[i], y[i] = self[indices[i]]
        
        return x, y
    
    
    def get_batch_class(self, batch_size, class_id):

        class_indices = self.class_indices[class_id]

        x = torch.zeros((batch_size, ) + self.input_shape, dtype=torch.float32)

        indices = numpy.random.randint(0, len(class_indices), batch_size)

        for i in range(batch_size):
            x[i], _ = self[class_indices[indices[i]]]
        
        return x
