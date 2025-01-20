# Vit implementation from scratch

import torch
from torch import nn, randn, cat
import torch.nn.functional as F


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: tuple,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 8,
        forward_expansion: float = 1.5,
        dropout: float = 0.1,
    ):
        super(VisionTransformer, self).__init__()

        self.patch_embedding = InputEmbedding(
            img_size, patch_size, num_channels, embed_dim
        )
        self.transformer = Transformer(embed_dim, num_heads, forward_expansion)
        self.fcc = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes)
                )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        cls_token_final = x[:, 0]
        out = self.fcc(cls_token_final)
        return out


# Somewhat Done
class InputEmbedding(nn.Module):
    def __init__(
        self, img_size: tuple, patch_size: int, num_channels: int, embed_dim: int
    ):
        super(InputEmbedding, self).__init__()

        # ToDo! -> Geral patch and image size (h * w)
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        self.num_patches = (img_size[0] // patch_size) ** 2
        self.patch_dim = self.num_channels * self.patch_size * self.patch_size
        self.projection = nn.Linear(self.patch_dim, self.embed_dim)

        self.cls_token = nn.Parameter(randn(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            randn(1, self.num_patches + 1, self.embed_dim)
        )
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        batch_size = x.shape[0]
        patches = self.unfold(x).transpose(1, 2)
        x = self.projection(patches)
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = cat([cls_tokens, x], dim=1)
        x += self.position_embedding
        return x


# Done
class Transformer(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, forward_expansion: int, dropout: int = 0.1
    ):
        super(Transformer, self).__init__()

        self.multi_head = MultiHeadSelfAttention(embed_dim, num_heads)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_drop_1 = nn.Dropout(dropout)

        self.feed_forward = FeedForward(embed_dim, embed_dim * forward_expansion)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.layer_drop_2 = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.multi_head(x)
        norm_1 = self.layer_norm_1(attention)
        drop_1 = self.layer_drop_1(norm_1)

        forward = self.feed_forward(drop_1)
        norm_2 = self.layer_norm_2(forward + x)
        out = self.layer_drop_2(norm_2)
        return out


# Done
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"Embeded Dimension: {embed_dim} isn't divided by the number of heads: {num_heads}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape 
        print("\nX Shape: ", x.shape)
        print("Head Dim: ", self.head_dim)

        Q = self.query(x)
        print("Q Shape: ", Q)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        QK = torch.matmul(Q, K.transpose(-2, -1))
        soft_QK = torch.softmax(QK / (self.head_dim**0.5), dim=1)
        out = torch.matmul(soft_QK, V)
        
        # Concatenate Heads
        out = out.transpose(1, 2).contiguous().view(batch_size, num_patches, emb_dim)
        return out


# Done
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.fcc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.fcc(x)


if __name__ == "__main__":


    #factory_kwargs = {"dtype": torch.float32, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}


    # Test InputEmbedding
    batch_size = 16
    num_channels = 3
    img_size = (228, 228)
    patch_size = 16
    embed_dim = 768
    num_classes = 10
    num_heads = 8

    x = randn(batch_size, num_channels, img_size[0], img_size[1])
    #model = VisionTransformer(
    #    img_size, patch_size, num_channels, embed_dim, num_classes, num_heads
    #)

    #preds = model(x)
   # print(preds.shape)

    factory_kwargs = {"dtype": torch.float32, "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

    path_emb = InputEmbedding(img_size, patch_size, num_channels, embed_dim)
    multi_head = MultiHeadSelfAttention(embed_dim, num_heads)

    x = path_emb(x)
    preds = multi_head(x)
    print(preds)


    


