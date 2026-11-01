"""Pure scalar objectives; gradient accumulation belongs to the trainer."""

from torch import Tensor
from torch.nn import functional as F


def _logits(real: Tensor, fake: Tensor) -> tuple[Tensor, Tensor]:
    if real.numel() == 0 or fake.numel() == 0:
        raise ValueError("RaLSGAN requires nonempty real and fake logits")
    if not real.is_floating_point() or not fake.is_floating_point():
        raise ValueError("RaLSGAN logits must be floating point")
    if real.device != fake.device:
        raise ValueError("Real and fake logits must use the same device")
    return real.float(), fake.float()


def ralsgan_discriminator(real: Tensor, fake: Tensor) -> Tensor:
    """RaLSGAN D loss; caller detaches fake images before the D forward pass.

    Means span the complete batch and spatial maps. Concatenate any chunked
    forwards before calling this function; averaging per-chunk losses differs.
    """
    real, fake = _logits(real, fake)
    return (
        (real - fake.mean() - 1).square().mean() + (fake - real.mean() + 1).square().mean()
    ) * 0.5


def ralsgan_generator(real: Tensor, fake: Tensor) -> Tensor:
    """RaLSGAN G loss, preserving gradients through both fake-logit terms."""
    real, fake = _logits(real, fake)
    return (
        (real - fake.mean() + 1).square().mean() + (fake - real.mean() - 1).square().mean()
    ) * 0.5


def auxiliary_losses(
    structure_pred: Tensor, texture_pred: Tensor, structure_gt: Tensor, image_gt: Tensor
) -> dict[str, Tensor]:
    """Structure/texture supervision at each branch prediction resolution.

    Targets are detached and bilinearly resized (align_corners=False), matching
    the original branch-target interpolation. No backward or gradient mutation
    occurs here; the trainer adds each scalar exactly once with its coefficient.
    """
    pairs = ((structure_pred, structure_gt), (texture_pred, image_gt))
    for prediction, target in pairs:
        if (
            prediction.ndim != 4
            or target.ndim != 4
            or prediction.shape[0] < 1
            or prediction.shape[:2] != target.shape[:2]
            or prediction.shape[1] != 3
            or prediction.device != target.device
            or not prediction.is_floating_point()
            or not target.is_floating_point()
        ):
            raise ValueError("Auxiliary supervision expects matching nonempty RGB batches")
    return {
        name: F.l1_loss(
            prediction.float(),
            F.interpolate(
                target.detach().float(),
                size=prediction.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
        )
        for name, (prediction, target) in zip(("structure", "texture"), pairs)
    }
