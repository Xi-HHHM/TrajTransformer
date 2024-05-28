# Transformer implementation based on https://github.com/lucidrains/x-transformers
from x_transformers.x_transformers import *


class XTrajTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        traj_length,
        traj_dim,
        attn_layers: Encoder,
        num_classes=None,
        post_emb_norm=False,
        num_register_tokens=0,
        emb_dropout=0.
    ):
        super().__init__()
        dim = attn_layers.dim

        self.pos_embedding = nn.Parameter(torch.randn(1, traj_length, dim))

        has_register_tokens = num_register_tokens > 0
        self.has_register_tokens = has_register_tokens

        if has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.patch_to_embedding = nn.Sequential(
            # LayerNorm(traj_dim),
            nn.Linear(traj_dim, dim, bias=False),
            LayerNorm(dim)
        )

        # self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers

        self.mlp_head = nn.Linear(dim, num_classes) if exists(num_classes) else nn.Identity()

    # fixme
    def forward(
        self,
        traj,
        return_embeddings=False,
        return_logits_and_embeddings=False
    ):
        b, n = traj.shape[0], traj.shape[1]  # batch size, sequence length

        x = self.patch_to_embedding(traj)

        x = x + self.pos_embedding[:, :n]

        # x = self.post_emb_norm(x)
        x = self.dropout(x)

        if self.has_register_tokens:
            r = repeat(self.register_tokens, 'n d -> b n d', b=b)
            x, ps = pack((x, r), 'b * d')

        embed = self.attn_layers(x)

        if self.has_register_tokens:
            embed, _ = unpack(embed, ps, 'b * d')

        assert at_most_one_of(return_embeddings, return_logits_and_embeddings)

        if return_embeddings:
            return embed

        pooled = embed.mean(dim=-2)
        logits = self.mlp_head(pooled)

        if not return_logits_and_embeddings:
            return logits

        return logits, embed


class XObstacleTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        obstacle_length,
        obstacle_dim,
        attn_layers: Encoder,
        num_classes=None,
        post_emb_norm=False,
        num_register_tokens=0,
        emb_dropout=0.
    ):
        super().__init__()
        dim = attn_layers.dim

        self.pos_embedding = nn.Parameter(torch.randn(1, obstacle_length, dim))

        has_register_tokens = num_register_tokens > 0
        self.has_register_tokens = has_register_tokens

        if has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.patch_to_embedding = nn.Sequential(
            LayerNorm(obstacle_dim),
            nn.Linear(obstacle_dim, dim, bias=False),
            LayerNorm(dim)
        )

        # self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.cross_attn = self.attn_layers.cross_attend

        self.mlp_head = nn.Linear(dim, num_classes) if exists(num_classes) else nn.Identity()

    # fixme
    def forward(
        self,
        obstacle,
        traj_embedding=None,
        return_embeddings=False,
        return_logits_and_embeddings=False
    ):
        b, n = obstacle.shape[0], obstacle.shape[1]  # batch size, sequence length

        x = self.patch_to_embedding(obstacle)

        x = x + self.pos_embedding[:, :n]

        #x = self.post_emb_norm(x)
        x = self.dropout(x)

        if self.has_register_tokens:
            r = repeat(self.register_tokens, 'n d -> b n d', b=b)
            x, ps = pack((x, r), 'b * d')

        if self.cross_attn:
            assert traj_embedding is not None
            assert x.shape[-1] == traj_embedding.shape[-1], f"Shape mismatch x: {x.shape}, traj_embedding: {traj_embedding.shape}"
            embed = self.attn_layers(x, context=traj_embedding)
        else:
            embed = self.attn_layers(x)

        if self.has_register_tokens:
            embed, _ = unpack(embed, ps, 'b * d')

        assert at_most_one_of(return_embeddings, return_logits_and_embeddings)

        if not exists(self.mlp_head) or return_embeddings:
            return embed

        pooled = embed.mean(dim=-2)
        logits = self.mlp_head(pooled)

        if not return_logits_and_embeddings:
            return logits

        return logits, embed


class TrajReconstructor(nn.Module):
    def __init__(self, pre_mlp_in, post_mlp_out, mlp_n_hidden, mlp_n_layers,
                 transformer_emb_dim=256,
                 transformer_depth=4,
                 transformer_heads=4,
                 transformer_register_tokens=0,
                 dropout=0.1):
        super(TrajReconstructor, self).__init__()
        self.pre_mlp_input_dim = pre_mlp_in
        self.post_mlp_output_dim = post_mlp_out
        self.dropout = dropout

        self.n_mlp_layers = mlp_n_layers
        self.n_mlp_hidden = mlp_n_hidden

        encoder = Encoder(
            dim=transformer_emb_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            layer_dropout=dropout, )

        self.encoder = XTrajTransformerWrapper(
            traj_length=51,
            traj_dim=self.pre_mlp_input_dim,
            attn_layers=encoder,
            num_classes=None,
            post_emb_norm=False,
            num_register_tokens=transformer_register_tokens,
            emb_dropout=dropout)

        self.fc = nn.Sequential()
        if self.n_mlp_layers > 1:
            for i in range(self.n_mlp_layers - 1):
                self.fc.add_module(f"fc{i}", nn.Linear(transformer_emb_dim, self.n_mlp_hidden))
                self.fc.add_module(f"relu{i}", nn.ReLU())

        self.fc.add_module("fc_last", nn.Linear(self.n_mlp_hidden, self.post_mlp_output_dim))

    def forward(self, x):
        x = self.encoder(x, return_embeddings=False)
        x = self.fc(x)
        return x
