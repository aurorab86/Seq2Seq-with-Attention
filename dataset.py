import torch 
import numpy as np

def generate_data(seq_length_min, seq_length_max, batch_size):
    T = np.random.randint(seq_length_min, seq_length_max + 1)
    x = np.random.randint(0, 10, (T, batch_size))
    one_hot_x = np.zeros((T + 1, batch_size, 12), dtype=np.float32)
    one_hot_x[np.arange(T).reshape(-1, 1), np.arange(batch_size), x] = 1
    one_hot_x[-1, :, -1] = 1
    ends = np.full(batch_size, 11).reshape(1, -1)
    y = np.concatenate([x[::-1], ends], axis=0)
    return x, one_hot_x, y


def one_hot_encoding_prediction(y_t):
    y_t = y_t.detach().cpu().numpy()
    max_values_mask = (y_t == y_t.max(axis=-1, keepdims=True))
    one_hot_encoding = torch.tensor(max_values_mask.astype(np.float32), dtype=torch.float32)
    return one_hot_encoding