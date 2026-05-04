from __future__ import annotations

import torch


def load_encoder_side_checkpoint(model, checkpoint_path: str):
    """Load encoder-side weights from a stage-1 checkpoint into *model*.

    Strips the leading ``"model."`` prefix from Lightning-saved state-dict keys
    so that the weights map correctly onto a bare :class:`CrossModalityJEPA`
    instance.  Uses ``strict=False`` to allow partial loading (e.g. when the
    decoder weights are absent from the checkpoint).

    Args:
        model: The :class:`CrossModalityJEPA` instance to load weights into.
        checkpoint_path: Path to the ``.ckpt`` or raw ``state_dict`` file.

    Returns:
        The same *model* with encoder-side weights loaded in-place.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state_dict[key[len("model."):]] = value
        else:
            cleaned_state_dict[key] = value
    model.load_state_dict(cleaned_state_dict, strict=False)
    return model
