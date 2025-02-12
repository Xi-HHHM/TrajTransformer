import numpy as np
import torch
from transformers import AutoProcessor
import torch.nn as nn
from x_transformers import TransformerWrapper, Encoder


class FastokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = None
        self.tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast",
                                                       trust_remote_code=True)
        self._init_transformer()

    def _init_transformer(self):
        self.transformer = TransformerWrapper(
            emb_dim=768,
            num_tokens=20000,
            max_seq_len=1024,
            attn_layers=Encoder(
                dim=512,
                depth=1,
                heads=8
                )
            )

    def forward_transformer(self, token):
        # Define a CLS token and PAD token (arbitrary IDs)
        CLS_TOKEN = 101
        PAD_TOKEN = 0

        # Pad sequences to the max length
        max_length = 400
        # Add CLS token at the start of each trajectory
        trajectories = [[CLS_TOKEN] + traj for traj in token]

        padded_trajectories = [traj + [PAD_TOKEN] * (max_length - len(traj)) for traj in trajectories]
        token_tensor = torch.stack([torch.Tensor(t) for t in padded_trajectories])

        # ToDo - Be careful - there is already a cls token in the x-transformer
        cls = self.transformer(token_tensor)

        print(cls.shape)
        return cls

    def forward(self, x):
        """
        Input should be trajectories.
        Shape [Batch, Time, DoF]
        """
        assert len(x.shape) == 3, f"The input vector should have a dimension of 3 - get {x.shape}"

        # Option 1 - Encode all DoF into one vector
        token = self.tokenizer(x)
        # ToDo: Different length in a batch.
        # Option 2 - Go through the all DoF one by one and stack them (ToDo)

        # Go through the transformer and get one CLS token.
        cls = self.forward_transformer(token)

        return cls


if __name__ == "__main__":
    action_data = np.random.rand(1, 50, 14)
    token_encoder = FastokenEncoder()
    token_encoder(action_data)
