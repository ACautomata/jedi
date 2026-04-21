from __future__ import annotations

import torch


def run_inference(model, decoder, src_volume: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        src_output = model.encode_volume(src_volume)
        pred_tgt_emb = model.predict_tgt(src_output["patch_embeddings"])
        grid_size = src_output["grid_size"]
        return decoder(pred_tgt_emb, grid_size)


def main():
    raise NotImplementedError("Wire checkpoints and CLI arguments before production inference use.")
