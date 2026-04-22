from __future__ import annotations

import argparse

import torch
import yaml
from hydra.utils import instantiate

from jedi.models import CrossModalityJEPA


def load_model_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def load_encoder_side_checkpoint(model, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state_dict[key[len("model."):]] = value
        else:
            cleaned_state_dict[key] = value
    model.load_state_dict(cleaned_state_dict, strict=False)
    return model


def load_decoder_checkpoint(decoder, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("decoder."):
            cleaned_state_dict[key[len("decoder."):]] = value
        else:
            cleaned_state_dict[key] = value
    decoder.load_state_dict(cleaned_state_dict, strict=False)
    return decoder


def build_inference_components(model_config_path: str, decoder_config_path: str, encoder_checkpoint: str, decoder_checkpoint: str):
    model_cfg = load_model_config(model_config_path)
    decoder_cfg = load_model_config(decoder_config_path)
    encoder = instantiate(model_cfg["encoder"])
    projector = instantiate(model_cfg["projector"])
    predictor = instantiate(model_cfg["predictor"])
    pred_proj = instantiate(model_cfg["pred_proj"])
    modality_embedder_cfg = model_cfg.get("modality_embedder", None)
    modality_embedder = instantiate(modality_embedder_cfg) if modality_embedder_cfg else None
    model = CrossModalityJEPA(
        encoder=encoder,
        projector=projector,
        predictor=predictor,
        pred_proj=pred_proj,
        modality_embedder=modality_embedder,
    )
    model = load_encoder_side_checkpoint(model, encoder_checkpoint)
    decoder = instantiate(decoder_cfg["decoder"])
    decoder = load_decoder_checkpoint(decoder, decoder_checkpoint)
    model.eval()
    decoder.eval()
    return model, decoder


def run_inference(model, decoder, src_volume: torch.Tensor, tgt_modality_idx: int | None = None) -> torch.Tensor:
    with torch.no_grad():
        src_output = model.encode_volume(src_volume)
        modality_tensor = torch.tensor([tgt_modality_idx]) if tgt_modality_idx is not None else None
        pred_tgt_emb = model.predict_tgt(src_output["patch_embeddings"], tgt_modality=modality_tensor)
        grid_size = src_output["grid_size"]
        return decoder(pred_tgt_emb, grid_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--decoder-config", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tgt-modality", type=int, default=None, help="Target modality index (0=t1n, 1=t1c, 2=t2w, 3=t2f)")
    args = parser.parse_args()

    model, decoder = build_inference_components(
        model_config_path=args.model_config,
        decoder_config_path=args.decoder_config,
        encoder_checkpoint=args.encoder_checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
    )
    src_volume = torch.load(args.input, map_location="cpu")
    prediction = run_inference(model, decoder, src_volume, tgt_modality_idx=args.tgt_modality)
    torch.save(prediction, args.output)


if __name__ == "__main__":
    main()
