from torch import nn


class CrossModalityJEPA(nn.Module):
    def __init__(self, encoder, projector, predictor, pred_proj, modality_embedder=None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        self.pred_proj = pred_proj
        self.modality_embedder = modality_embedder

    def encode_volume(self, volume):
        encoded = self.encoder(volume)
        patch_embeddings = self.projector(encoded["patch_embeddings"])
        return {
            "patch_embeddings": patch_embeddings,
            "cls_embedding": encoded["cls_embedding"],
            "grid_size": encoded["grid_size"],
        }

    def encode_src_tgt(self, src, tgt):
        src_output = self.encode_volume(src)
        tgt_output = self.encode_volume(tgt)
        return src_output, tgt_output

    def predict_tgt(self, src_embeddings, tgt_modality=None):
        if self.modality_embedder is not None and tgt_modality is not None:
            cond = self.modality_embedder(tgt_modality)
        else:
            cond = None
        predicted = self.predictor(src_embeddings, cond)
        return self.pred_proj(predicted)
