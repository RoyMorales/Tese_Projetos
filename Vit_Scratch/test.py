import torch
import torch.nn as nn

class MultiHeadProjection(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadProjection, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(self, value, key, query):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        values = value.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = key.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        return values, keys, queries

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.projection = MultiHeadProjection(embed_size, num_heads)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.head_dim = embed_size // num_heads

    def forward(self, value, key, query):
        values, keys, queries = self.projection(value, key, query)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.nn.functional.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(query.shape[0], query.shape[1], -1)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_size):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_size))

    def forward(self, x):
        N = x.shape[0]
        x = self.projection(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position_embedding
        return x

class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size, 
                 patch_size, 
                 in_channels, 
                 num_classes, 
                 embed_size, 
                 num_heads, 
                 num_layers, 
                 forward_expansion, 
                 dropout):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        for layer in self.layers:
            x = layer(x, x, x)
        x = self.norm(x)
        cls_output = x[:, 0]
        out = self.fc_out(cls_output)
        return out

# Example usage
if __name__ == "__main__":
    img_size = (224, 224)
    patch_size = 16
    in_channels = 3
    num_classes = 1000
    embed_size = 768
    num_heads = 12
    num_layers = 12
    forward_expansion = 4
    dropout = 0.1

    model = VisionTransformer(
        img_size, patch_size, in_channels, num_classes, embed_size, num_heads, num_layers, forward_expansion, dropout
    )

    x = torch.randn(1, 3, 224, 224)
    preds = model(x)
    print(preds.shape)

