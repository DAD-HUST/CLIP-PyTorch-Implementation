from torch import nn

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        drop_rate,
        config,
    ):
        super().__init__()
        self.embedding_dim = config['projection_head']['image_embedding']
        self.projection_dim = config['projection_head']['projection_dim']
        self.drop_rate = config['projection_head']['drop_rate']
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x