import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """Attention-Based Multiple Instance Learning (Ilse et al., 2018).

    Takes a bag of instance features of shape (B, N, D) and predicts a bag-level label.
    Uses gated attention: a(x) = softmax(W * (tanh(V*x) ⊙ sigmoid(U*x))).

    Args:
        in_features: Dimension D of each instance feature vector.
        hidden_dim: Dimension of the attention hidden layer.
        out_features: Number of output classes.
        num_branches: Number of parallel attention branches (multi-head).
        attention_dropout: Dropout probability in the classifier head.
    """

    def __init__(
        self,
        in_features: int = 1280,
        hidden_dim: int = 256,
        out_features: int = 1,
        num_branches: int = 1,
        attention_dropout: float = 0.25,
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features

        self.attention_v = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, num_branches)

        self.classifier = nn.Sequential(
            nn.Linear(in_features * num_branches, in_features // 2),
            nn.ReLU(),
            nn.Dropout(attention_dropout),
            nn.Linear(in_features // 2, out_features),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> dict:
        """Run a forward pass over a feature bag.

        Args:
            x: Instance features of shape (B, N, D) or (N, D). A batch dimension
                is added automatically when the input is 2-D.
            return_attention: If True, the returned dict contains an ``attention``
                key with the softmax attention weights of shape (B, num_branches, N).

        Returns:
            A dictionary with:
                - ``"logits"`` (torch.Tensor): Bag-level predictions, shape (B, out_features).
                - ``"attention"`` (torch.Tensor, optional): Attention weights,
                  shape (B, num_branches, N). Only present when *return_attention* is True.

        Raises:
            ValueError: If *x* is not 2-D or 3-D, or if the feature dimension does
                not match ``in_features``.
        """
        if x.dim() not in (2, 3):
            raise ValueError(
                f"Expected 2-D (N, D) or 3-D (B, N, D) input, got {x.dim()}-D tensor."
            )

        # Ensure batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, N, D = x.shape
        if D != self.in_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.in_features}, got {D}."
            )

        # Gated attention
        a_v = self.attention_v(x)  # (B, N, hidden_dim)
        a_u = self.attention_u(x)  # (B, N, hidden_dim)
        a = self.attention_weights(a_v * a_u)  # (B, N, num_branches)
        a = torch.transpose(a, 1, 2)  # (B, num_branches, N)
        a = F.softmax(a, dim=2)  # (B, num_branches, N)

        # Aggregate
        M = torch.bmm(a, x)  # (B, num_branches, D)
        M = M.view(B, -1)  # (B, num_branches * D)

        logits = self.classifier(M)  # (B, out_features)

        out = {"logits": logits}
        if return_attention:
            out["attention"] = a
        return out
