import torch
import matplotlib.pyplot as plt
import torch.nn as nn

import numpy as np

from dataset import generate_data

from IPython.display import clear_output
def plot_all(losses, acces):
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.plot(acces, label='acc')
    plt.xlabel('Batches (It must multiple x 1000)')
    plt.ylabel('Loss and Acc')
    plt.title('Loss and Acc Averages over Epochs')
    plt.legend()
    plt.grid()
    plt.show()


def model_train(optimizer, model, seq_length_min, seq_length_max, batch_size):
    
    num_batches = 30000

    losses = []
    accs = []
    lossesaverage_list = []
    accslist_average = []
    for k in range(num_batches):
        sequence, x, target  = generate_data(seq_length_min, seq_length_max, batch_size=batch_size)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)

        optimizer.zero_grad()
        output, _ = model(x_tensor)

        lossesfn = nn.CrossEntropyLoss()
        loss = lossesfn(output.reshape(-1, 12), target_tensor.reshape(-1))
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=-1) == target_tensor).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)
        if k % 1000 == 0:
            accsaverage = np.mean(accs)
            lossesaverage = np.mean(losses)
            lossesaverage_list.append(lossesaverage)
            accslist_average.append(accsaverage)
            plot_all(lossesaverage_list, accslist_average)
            print(f"loss: {lossesaverage}, accuracy: {accsaverage}")