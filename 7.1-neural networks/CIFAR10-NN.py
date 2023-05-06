import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import seaborn as sns
from typing import List #just to help readability for funct def

#####################################################################

def create_sequential_model(dim_in:int, dim_out:int, hidden_layer_sizes:List[int]):
    print(dim_in)
    hiddens = [dim_in, *hidden_layer_sizes]
    print(hiddens)
    torch_layers = []

    #Create a linear layer and feed it through a ReLU
    for i in range(len(hiddens)-1):
        torch_layers.append(torch.nn.Linear(hiddens[i], hiddens[i+1]))
        torch_layers.append(torch.nn.ReLU())

    torch_layers.append(torch.nn.Linear(hiddens[-1], dim_out)) #create the output layer
    return torch.nn.Sequential(*torch_layers)


def create_model():
    CIFAR10_train = torchvision.datasets.CIFAR10('CIFAR10_data',download=False,train=True, transform=True)
    CIFAR10_validation = torchvision.datasets.CIFAR10('CIFAR10_data',download=False,train=False, transform=True)

    print(CIFAR10_train.data.shape)
    print(len(CIFAR10_train.classes))

    training_data = (CIFAR10_train.data.reshape((-1,32*32*3))/255.0).astype(np.float32) # flatten the dataset and normalise
    training_labels = np.asarray(CIFAR10_train.targets, dtype=np.int64)
    validation_data = (CIFAR10_validation.data.reshape((-1,32*32*3))/255.0).astype(np.float32) # flatten the dataset and normalise
    validation_labels = np.asarray(CIFAR10_validation.targets, dtype=np.int64)
    
    model = create_sequential_model(32*32*3,10, [100,100])
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    learning_rate = 1e-3 #starting learning rate that we can tweak to increase performance
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate) #model.parameters gives the weight matrices and biases to the optimiser (AKA trainable p
        
    batch_size = 256 #the number of datapoints per batch that we do
    optimisation_steps = int(1e4) #the number of batches that we train
    
    metrics = []
    for i in range(optimisation_steps):
        idx = np.random.randint(0, training_data.shape[0], size = batch_size) # random sample of batch_size indices from 0 to the number of datapoints the dataset has 
        x = training_data[idx,:] # get the datapoints at the sampled indices
        # flattened_x = torch.from_numpy(x.reshape(batch_size,-1)).as # flatten the datapoints
        y_pred = model(torch.from_numpy(x)) # predict the classes of the datapoints)
        loss = criterion(y_pred,torch.from_numpy(training_labels[idx])) # compute the loss by comparing the predicted labels vs the actual labels
        # zero the gradients held by the optimiser
        optimiser.zero_grad()
        # perform a backward pass to compute the gradients
        loss.backward()
        # update the weights
        optimiser.step()

        # Record current step 
        if i%100==99:
            if i%1000==999:
                train_pred =  model(torch.from_numpy(training_data))
                val_pred =  model(torch.from_numpy(validation_data))
                train_accuracy = torch.mean((train_pred.argmax(dim=1) == torch.from_numpy(training_labels)).float())
                val_accuracy = torch.mean((val_pred.argmax(dim=1) == torch.from_numpy(validation_labels)).float())
                # print the loss every 100 steps
                metrics.append([i,loss.item(),train_accuracy.numpy(), val_accuracy.numpy()])
            print(f'\rEpoch: {i} Loss:{round(loss.item(),2)}', end='')
    
    metrics = np.asarray(metrics)
    sns.lineplot(x=metrics[:,0],y=metrics[:,1])
    plt.xlabel('step')
    plt.ylabel('training loss')
    plt.show()
    sns.lineplot(x=metrics[:,0],y=metrics[:,2],label='training')
    sns.lineplot(x=metrics[:,0],y=metrics[:,3], label='validation')
    plt.xlabel('step')
    plt.ylabel('accuracy')
    plt.show()

####################
create_model()
#########