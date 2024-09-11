import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm.auto import tqdm
from dataset import *

def test_model(model):
    length_total = defaultdict(int)
    length_correct = defaultdict(int)

    with torch.no_grad():
        for i in tqdm(range(50000)):
            if i % 5000 == 0:
                print(f"i = {i}")
            sequence, x, target  = generate_data(1, 20, 1)

            x_tensor = torch.tensor(x, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.long)

            output, _ = model(x_tensor)

            length_total[sequence.size] += 1
            if (output.argmax(dim=-1) == target_tensor).all():
                length_correct[sequence.size] += 1

    fig, ax = plt.subplots()
    x, y = [], []
    for i in range(1, 20):
        x.append(i)
        y.append(length_correct[i] / length_total[i])
    ax.plot(x, y)