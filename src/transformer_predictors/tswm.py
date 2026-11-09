"""Temporal-straightening world model.

The visual encoder is the from-scratch global ResNet used for the main
PushT runs in temporal-straightening: five residual blocks starting at
3 to 32 channels, GeM pooling, then Linear, GELU, Linear, LayerNorm.
The predictor is that repo's causal ViT. Actions are concatenated onto
the visual token. There is no decoder and no variance-covariance term.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from stable_worldmodel.wm.lewm.lewm import LeWM


def _group_count(channels: int) -> int:
    return min(32, max(1, channels // 4))


class GeM(nn.Module):
    """Generalized mean pooling. p=1 is average pooling."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        ).pow(1.0 / self.p)


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and GELU.

    ``resample='down'`` halves the spatial size. This is ``resblock_v2``
    from the temporal-straightening encoder.
    """

    def __init__(self, input_dim, output_dim, kernel_size, resample=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        padding = (kernel_size - 1) // 2

        if resample == 'down':
            self.skip = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding),
            )
            self.conv1 = nn.Conv2d(
                input_dim,
                input_dim,
                kernel_size,
                padding=padding,
                bias=False,
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding),
                nn.MaxPool2d(2, stride=2),
            )
            self.norm1 = nn.GroupNorm(_group_count(input_dim), input_dim)
            self.norm2 = nn.GroupNorm(_group_count(output_dim), output_dim)
        else:
            self.skip = (
                nn.Conv2d(input_dim, output_dim, 1)
                if input_dim != output_dim
                else nn.Identity()
            )
            self.conv1 = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                padding=padding,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                output_dim, output_dim, kernel_size, padding=padding
            )
            self.norm1 = nn.GroupNorm(_group_count(output_dim), output_dim)
            self.norm2 = nn.GroupNorm(_group_count(output_dim), output_dim)

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()

    def forward(self, x):
        identity = (
            x
            if self.input_dim == self.output_dim and self.resample is None
            else self.skip(x)
        )
        residual = self.act1(self.norm1(self.conv1(x)))
        residual = self.act2(self.norm2(self.conv2(residual)))
        return identity + residual


class SmallResNetGeM(nn.Module):
    """Five-block ResNet trained from scratch, pooled to one global token.

    Channel schedule is 3, 32, 64, 128, 256, 512. GeM pooling is followed
    by Linear, GELU, Linear, LayerNorm. ``dim`` is 384 in the main
    experiments.
    """

    def __init__(self, dim=384, gem_p=3.0):
        super().__init__()
        self.emb_dim = dim
        self.latent_ndim = 1
        self.rb1 = ResidualBlock(3, 32, 3, resample='down')
        self.rb2 = ResidualBlock(32, 64, 3, resample='down')
        self.rb3 = ResidualBlock(64, 128, 3, resample='down')
        self.rb4 = ResidualBlock(128, 256, 3, resample='down')
        self.rb5 = ResidualBlock(256, 512, 3, resample='down')
        self.gem = GeM(p=gem_p)
        self.flat = nn.Flatten()
        self.projection = nn.Sequential(
            nn.Linear(512, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        out = self.projection(self.flat(self.gem(x)))
        out = out.unsqueeze(1)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], 1, -1)
        return out


class ActionEncoder(nn.Module):
    """Conv1d plus LayerNorm action embedding.

    Matches the proprioceptive embedding used as the action encoder in
    temporal-straightening. ``emb_dim`` is 10 there.
    """

    def __init__(self, input_dim, emb_dim=10, tubelet_size=1, use_layernorm=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_embed = nn.Conv1d(
            input_dim, emb_dim, kernel_size=tubelet_size, stride=tubelet_size
        )
        self.norm = nn.LayerNorm(emb_dim) if use_layernorm else nn.Identity()

    def forward(self, x):
        x = x.to(self.patch_embed.weight.dtype)
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.norm(x)


def frame_embedding(output) -> torch.Tensor:
    """One vector per image.

    ``SmallResNetGeM`` returns ``(B, 1, D)``. A Hugging Face ViT exposes
    ``last_hidden_state`` and the CLS token is used. A pooled classifier
    backbone exposes ``logits``.
    """
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state[:, 0]
    if hasattr(output, 'logits'):
        return output.logits
    if torch.is_tensor(output):
        if output.ndim == 2:
            return output
        if output.ndim == 3:
            return output[:, 0]
    raise TypeError(
        'Encoder output must be a token tensor, a ViT hidden state, '
        f'or ResNet logits. Got {type(output)!r}.'
    )


def temporal_straightening_losses(
    emb,
    pred_emb,
    tgt_emb,
    *,
    stop_grad,
    curvature,
    curvature_weight,
):
    """Prediction MSE plus temporal straightening on encoder embeddings.

    ``stop_grad`` detaches the prediction target so that loss does not
    train the encoder through the target branch. ``curvature`` is
    ``TemporalStraighteningLoss``: mean negative cosine similarity of
    consecutive latent velocities, the SWM form of the paper's ``1 - cos``.
    """
    target = tgt_emb.detach() if stop_grad else tgt_emb
    pred_loss = (pred_emb - target).pow(2).mean()
    if emb.size(1) >= 3:
        curv_loss = curvature(emb)
    else:
        curv_loss = pred_loss.new_zeros(())
    loss = pred_loss + float(curvature_weight) * curv_loss
    return {
        'pred_loss': pred_loss,
        'curvature_loss': curv_loss,
        'loss': loss,
    }


class TSWM(LeWM):
    """Global ResNet encoder, causal ViT predictor, target stop-gradient.

    ``rollout`` is LeWM's. ``predict`` concatenates the action embedding
    onto each visual token, runs the causal ViT, and returns the visual
    channels. ``stop_grad`` is read by :meth:`training_loss`.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        stop_grad=True,
        interpolate_pos_encoding=False,
        **kwargs,
    ):
        super().__init__(
            encoder,
            predictor,
            action_encoder,
            projector=projector,
            pred_proj=pred_proj,
            **kwargs,
        )
        self.stop_grad = bool(stop_grad)
        self.interpolate_pos_encoding = bool(interpolate_pos_encoding)

    def encode(self, info):
        """Encode observations and actions into embeddings.

        ``info['pixels']`` is ``(B, T, C, H, W)``. ``info['emb']`` is
        ``(B, T, D)`` after the projector. ``info['act_emb']`` is set when
        ``action`` is present.
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        batch = pixels.size(0)
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')
        kwargs = {}
        if self.interpolate_pos_encoding:
            kwargs['interpolate_pos_encoding'] = True
        output = self.encoder(pixels, **kwargs)
        emb = self.projector(frame_embedding(output))
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=batch)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict the next visual embedding.

        ``emb`` is ``(B, T, D)`` and ``act_emb`` is ``(B, T, A)``. The
        causal ViT sees them concatenated on the feature axis, which is
        how ``VWorldModel`` feeds ``ViTPredictor`` when there is one
        token per frame. The returned tensor is the visual slice,
        ``(B, T, D)``.
        """
        visual = emb.unsqueeze(2)
        action = act_emb.unsqueeze(2).expand(-1, -1, visual.size(2), -1)
        tokens = torch.cat([visual, action], dim=-1)
        batch, steps, patches, _ = tokens.shape
        flat = rearrange(tokens, 'b t p d -> b (t p) d')
        pred = self.predictor(flat)
        pred = rearrange(pred, 'b (t p) d -> b t p d', t=steps, p=patches)
        visual_pred = pred[..., : emb.size(-1)][:, :, 0]
        visual_pred = self.pred_proj(
            rearrange(visual_pred, 'b t d -> (b t) d')
        )
        return rearrange(visual_pred, '(b t) d -> b t d', b=batch)

    def training_loss(self, emb, pred_emb, tgt_emb, curvature, curvature_weight):
        return temporal_straightening_losses(
            emb,
            pred_emb,
            tgt_emb,
            stop_grad=self.stop_grad,
            curvature=curvature,
            curvature_weight=curvature_weight,
        )


__all__ = [
    'ActionEncoder',
    'SmallResNetGeM',
    'TSWM',
    'frame_embedding',
    'temporal_straightening_losses',
]
