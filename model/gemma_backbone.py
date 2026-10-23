"""Gemma-specific loading, readout extraction, and backbone adapters."""
from pathlib import Path


def load_gemma():
    # Lazy import keeps cached training independent of live model construction.
    from . import gemma
    return gemma


def load_letters(base, identity):
    import json
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from utils import file_hash
    config = json.loads((base / "config.json").read_text())
    if file_hash(base / "tokenizer.json") != identity["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from feature export")
    tok = Tokenizer.from_file(str(base / "tokenizer.json"))
    ids = [tok.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    if any(len(x) != 1 for x in ids):
        raise ValueError("Expected single-token answer letters")
    key = "model.language_model.embed_tokens.weight"
    with safe_open(base / "model.safetensors", framework="pt", device="cpu") as source:
        view = source.get_slice(key)
        weight = torch.cat([view[x[0]:x[0] + 1] for x in ids]).contiguous()
    if weight.shape != (26, identity["hidden_width"]) or weight.dtype != torch.bfloat16:
        raise ValueError("Readout shape/dtype differs")
    text = config["text_config"]
    if not text.get("tie_word_embeddings"):
        raise ValueError("This extraction expects the audited Gemma tied LM head")
    provenance = {"tensor_key": key, "letter_token_ids": [x[0] for x in ids],
                  "selected_rows_sha256": __import__("hashlib").sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest(),
                  "weight_sha256_from_existing_feature_identity": identity["weight_sha256"],
                  "full_weight_hash_rescan": False,
                  "softcap": text["final_logit_softcapping"],
                  "readout_position": "last valid token of existing cached Bazza prompt",
                  "prompt_changed": False}
    return weight, provenance


import torch
from torch import nn
from .backbone import BackboneOutput, FeatureCache


class Gemma4FeatureCache(FeatureCache):
    """Gemma final-layer cache using the stored effective RMSNorm weights."""
    def __init__(self, directory=None, *, selection, format='layer', banks=None,
                 split_map=None, layers=(48,), max_open_shards=128):
        if tuple(layers) != (48,):
            raise ValueError('Whole-backbone replacement currently requires layers=(48,); prefix/tap fusion is separate work')
        super().__init__(directory, selection=selection, format=format, banks=banks,
                         split_map=split_map, layer=48, max_open_shards=max_open_shards)


class Gemma4(nn.Module):
    """Live, frozen Gemma. Unpadded forwards retain the audited batch policy."""
    def __init__(self, weights=None, *, quantization='bnb-nf4',
                 selection=None, encoder=None):
        super().__init__()
        self.weights = Path(weights) if weights is not None else None
        self.quantization = quantization
        self.selection = selection or {}
        self.encoder = encoder.requires_grad_(False).eval() if encoder is not None else None
        self.register_buffer('_device',torch.empty(0),persistent=False)

    def identity(self):
        from utils import file_hash
        if self.encoder is not None and self.weights is None:
            return {'injected_encoder':type(self.encoder).__qualname__}
        return {str(path):file_hash(path) for path in (Path(__file__).with_name('gemma.py'),
                self.weights/'tokenizer.json',self.weights/'config.json')}

    def prepare_batch(self, rows, split):
        from data import collate_records
        batch = collate_records(rows)
        batch.split = split
        return batch

    def _load(self):
        if self.encoder is not None:
            return
        implementation = load_gemma()
        expected = self.selection.get('identity',{})
        for name,value in (('checkpoint',implementation.CHECKPOINT),('revision',implementation.REVISION),
                           ('weight_sha256',implementation.WEIGHT_SHA256),('quantization',self.quantization)):
            if expected.get(name) != value:
                raise ValueError('Live backbone differs from selection: '+name)
        from utils import file_hash
        if expected.get('implementation_sha256') is not None and expected['implementation_sha256'] != file_hash(Path(implementation.__file__)):
            raise ValueError('Live backbone differs from selection: implementation_sha256')
        self.encoder = implementation.load_model(self.weights,device=self._device.device,
            quantization=self.quantization).requires_grad_(False).eval()

    def forward(self, batch):
        if batch.hidden is not None:
            raise ValueError('Live Gemma expects tokens, not cached hidden states')
        batch = batch.to(self._device.device)
        self._load()
        chunks = []
        for i,row in enumerate(batch.rows):
            n = len(row['ids'])
            hidden = self.encoder(batch.input_ids[i:i+1,:n],batch.attention_mask[i:i+1,:n])
            chunks.append(torch.nn.functional.pad(hidden,(0,0,0,batch.input_ids.shape[1]-n)))
        return BackboneOutput(torch.cat(chunks),{})
