import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.optim as optim
import copy

def lr_finder(min_lr, max_lr, n_steps, loss, model, data_loaders, batch_size):
    
    # Save initial weights so we can restore them at the end
    torch.save(model.state_dict(), "__weights_backup")
    
    # specify optimizer
    optimizer = optim.SGD(model.parameters(), lr=min_lr)

    # We create a learning rate scheduler that increases the learning
    # rate at every batch.
    # Find the factor where min_lr r**(n_steps-1) = max_lr
    r = np.power(max_lr / min_lr, 1 / (n_steps - 1))

    def new_lr(epoch):
        """
        This should return the *factor* by which the initial learning
        rate must be multipled for to get the desired learning rate
        """
        return r ** epoch

    # This scheduler increases the learning rate by a constanct factor (r)
    # at every iteration
    lr_scheduler = LambdaLR(optimizer, new_lr)

    # Set the model in training mode
    # (so all layers that behave differently between training and evaluation,
    # like batchnorm and dropout, will select their training behavior)
    model.train()

    # Loop over the training data
    losses = {}
    train_loss = 0.0

    n_chars = len(model.chars)
    # initialize hidden state
    h = model.init_hidden(batch_size)

    for x, y in get_batches(data, batch_size, seq_length):
        counter += 1

        # One-hot encode our data and make them Torch tensors
        x = one_hot_encode(x, n_chars)
        inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

        if(train_on_gpu):
            inputs, targets = inputs.cuda(), targets.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        model.zero_grad()

        # get the output from the model
        output, h = model(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output, targets.view(batch_size*seq_length).long())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss.data.item() - train_loss)
        )

        losses[lr_scheduler.get_last_lr()[0]] = train_loss

        # Stop if the loss gets too big
        if train_loss / min(losses.values()) > 10:
            break

        if batch_idx == n_steps - 1:
            break
        else:
            # Increase the learning rate for the next iteration
            lr_scheduler.step()
    
    # Restore model to its initial state
    model.load_state_dict(torch.load('__weights_backup'))
    
    return losses
