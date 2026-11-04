"""Temporal-straightening world model on the LeWM stack.

The visual encoder is a pooled network: a Hugging Face ResNet from
``stable_worldmodel.wm.prejepa.module.create_backbone``, or a Hugging Face
ViT read at the CLS token. Dynamics are LeWM's action-conditioned
transformer. Actions stay a conditioning stream, so the embedding that is
predicted, straightened, and planned in is the visual latent. There is no
decoder and no variance-covariance regularizer.
"""

import torch
from einops import rearrange

from stable_worldmodel.wm.lewm.lewm import LeWM


def frame_embedding(output) -> torch.Tensor:
    """One vector per image from a ViT or a pooled ResNet.

    Hugging Face ViT exposes ``last_hidden_state``; the CLS token is the
    frame embedding. ``create_backbone`` replaces a ResNet classifier with
    LayerNorm and returns that pooled vector as ``logits``.
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
        'Encoder output must be a ViT hidden state, ResNet logits, '
        f'or a (B, D) tensor. Got {type(output)!r}.'
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
    """LeWM with a pooled visual encoder and a target-branch stop-gradient.

    ``predict`` and ``rollout`` are LeWM's. ``stop_grad`` is read by
    :meth:`training_loss`. ``interpolate_pos_encoding`` is passed only to
    ViT encoders; ResNet rejects that argument.
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

    def training_loss(self, emb, pred_emb, tgt_emb, curvature, curvature_weight):
        return temporal_straightening_losses(
            emb,
            pred_emb,
            tgt_emb,
            stop_grad=self.stop_grad,
            curvature=curvature,
            curvature_weight=curvature_weight,
        )


__all__ = ['TSWM', 'frame_embedding', 'temporal_straightening_losses']
