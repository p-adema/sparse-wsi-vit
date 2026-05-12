import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from sparse_wsi_vit.models.vit_5 import register as register_vit_5

register_vit_5()


class VitDensePreEmbedded(nn.Module):
    """ViT-5: Vision Transformers for the Mid-2020s

    Runs a ViT-5 model on the pre-embedded patch embeddings from Virchow2.
    This is a terrible, terrible idea, because the tokens aren't arranged in a square
    and this isn't what ViT-5 was designed to do. But I'll fix that later.

    Args:
        in_features: Dimension D of each instance feature vector.
        hidden_dim: Dimension of the attention hidden layer.
        out_features: Number of output classes.
        downproj: Reduce dimensionality via a Linear layer at the start
    """

    def __init__(
        self,
        in_features: int = 1280,
        out_features: int = 1,
        checkpoint_activations=False,
        downproj: int | None = None,
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.vit5 = timm.create_model(
            "vit5_small",
            num_classes=out_features,
            drop_rate=0.0,
            drop_path_rate=0.1,
            # img_size=320,  # 320 * 320 = 102 400
            pre_embedded_input=True,
            ape=False,  # it's learnable, which breaks with variable size inputs.
            patch_size=1,  # each token is a patch, effectively
            rope=True,
            rope_dynamic=True,
            embed_dim=in_features if downproj is None else downproj,
            # num_heads=8,
            # depth=6,
            checkpoint_activations=checkpoint_activations,
        )
        self.down = nn.Linear(in_features, downproj) if downproj else None

        if self.down:
            nn.init.orthogonal_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> dict:
        """Run a forward pass over a feature bag.

        Args:
            x: Instance features of shape (B, N, D) or (N, D). A batch dimension
                is added automatically when the input is 2-D.
            coords: Coordinates for each feature embedding in the WSI, shape (B, N, 2)
                or (N, 2).

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
                f"Expected 2-D (N, D) or 3-D (B, N, D) input, got {x.shape=}"
            )
        if coords.dim() not in (2, 3):
            raise ValueError(
                f"Expected 2-D (N, D) or 3-D (B, N, D) input, got {coords.shape=}"
            )

        # Ensure batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.size(1) != x.size(1):
            raise ValueError(f"{x.shape=} and {coords.shape} must agree on N")

        B, N, D = x.shape
        if D != self.in_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.in_features}, got {D}."
            )

        if self.down is not None:
            x = self.down(x)

        logits = self.vit5(x, coords)  # (B, out_features)

        out = {"logits": logits}
        return out
