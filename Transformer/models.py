
import torch.nn as nn
import torch
import os
import transformer


class ModelUtils:
    '''
    A utility class to save and load model weights
    '''

    def save_model(self, save_path, model):
        root, ext = os.path.splitext(save_path)
        if not ext:
            save_path = root + '.pth'
        try:
            torch.save(model.state_dict(), save_path)
            print(f'Successfully saved to model to "{save_path}"!')
        except Exception as e:
            print(f'Unable to save model, check save path!')
            print(f'Exception:\n{e}')
            return None

    def load_model(self, load_path, model):
        try:
            model.load_state_dict(torch.load(load_path))
            print(f'Successfully loaded the model from path "{load_path}"')

        except Exception as e:
            print(
                f'Unable to load the weights, check if different model or incorrect path!')
            print(f'Exception:\n{e}')
            return None

class transf_params:
    n_layers = 11
    num_heads = 12
    model_dim = 16  # nr of features
    forward_dim = 2048
    output_dim = 1
    dropout = 0
    n_epochs = 100
    lr = 0.01


class TransformerModel(nn.Module):
    def __init__(self, params):
        super(TransformerModel, self).__init__()
        self.transf = transformer.TransformerModel(n_layers=params.n_layers,
                                                   num_heads=params.num_heads,
                                                   model_dim=params.model_dim,
                                                   forward_dim=params.forward_dim,
                                                   output_dim=16,
                                                   dropout=params.dropout)
        self.linear = nn.Linear(16, params.output_dim)

    def forward(self, x):
        transf_out = self.transf(x)
        out = self.linear(transf_out)
        return out