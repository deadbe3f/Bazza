"""Our Python settings object. No model construction or Transformers dependency."""
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass
class Config:
    data_selection: Path
    root: Path | None = None
    project_root: Path | None = None
    data_root: Path | None = None
    cache_root: Path | None = None
    model_root: Path | None = None
    run_root: Path | None = None
    backbone_mode: str = 'cache'
    backbone_path: Path | None = None
    backbone_checkpoint: str = 'google/gemma-4-12B-it'
    backbone_revision: str | None = None
    quantization: str = 'bnb-nf4'
    feature_format: str = 'routed'
    feature_cache_path: Path | None = None
    feature_banks: dict = field(default_factory=dict)
    feature_split_map: dict = field(default_factory=lambda: {'full':'validation','retention':'validation'})
    feature_layers: tuple = (48,)
    prefetch_batches: int = 8
    max_open_shards: int = 128
    letter_readout_path: Path | None = None
    encoder_width: int = 3840
    hidden_width: int = 1024
    attention_heads: int = 4
    low_steps: int = 2
    high_cycles: int = 2
    gate_width: int = 32
    adapter_rank: int = 64
    softcap: float = 30.
    target_key: str = 'target'
    weighted_loss: bool = False
    seed: int = 20261013
    epochs: int = 1
    batch_size: int = 32
    token_budget: int = 32768
    optimizer: str = 'muon'
    learning_rate: float = 3e-5
    muon_learning_rate: float = .003
    weight_decay: float = .01
    beta2: float = .999
    gradient_clip: float = 1.
    precision: str = 'bf16'
    cpu_threads: int = 8
    schedule: str = 'wsd'
    warmup_examples: int = 640
    warmup_batch_offset: int = 32
    decay_start_examples: int = 25000
    minimum_lr_factor: float = .1
    save_every_examples: int = 1000
    log_every_steps: int = 10
    validation_panels: dict = field(default_factory=lambda: {'full':'validation'})
    validation_batch_size: int = 32
    validation_token_budget: int = 32768
    validation_every_examples: int = 8000
    initial_validation: bool = True
    keep_best_checkpoints: int = 3
    early_stopping: bool = False
    early_stop_patience: int = 8
    early_stop_min_examples: int = 24000
    early_stop_min_delta: float = .002
    test_config_path: Path | None = None
    test_split: str = 'test'
    overfit_url: str = ''
    overfit_project: str = 'Bazza refactor'

    def resolved(self, directory):
        def path(value):
            return (Path(directory) / value).resolve() if value is not None else None
        values = {key:path(getattr(self,key)) for key in ('root','project_root','data_root','cache_root',
                  'model_root','run_root','data_selection','backbone_path',
                  'feature_cache_path','letter_readout_path','test_config_path')}
        values['feature_banks'] = {name:path(value) for name,value in self.feature_banks.items()}
        values['feature_layers'] = tuple(self.feature_layers)
        result = replace(self, **values)
        if result.backbone_mode not in ('live','cache'):
            raise ValueError('backbone_mode must be live or cache')
        if result.backbone_mode == 'live' and result.backbone_path is None:
            raise ValueError('Live Gemma needs backbone_path')
        if result.backbone_mode == 'cache':
            if result.feature_format not in ('pt','layer','routed'):
                raise ValueError('feature_format must be pt, layer or routed')
            if result.feature_format == 'routed' and not result.feature_banks:
                raise ValueError('Routed cache needs feature_banks')
            if result.feature_format != 'routed' and result.feature_cache_path is None:
                raise ValueError('Feature cache needs feature_cache_path')
        if result.precision not in ('bf16','fp32') or result.schedule not in ('wsd','constant'):
            raise ValueError('Unsupported precision or schedule')
        if min(result.batch_size,result.validation_batch_size,result.prefetch_batches,result.epochs,
               result.token_budget,result.validation_token_budget,result.keep_best_checkpoints,result.log_every_steps) < 1:
            raise ValueError('Batch sizes, epochs, budgets, prefetch and retention must be positive')
        return result

    def to_dict(self):
        def plain(value):
            if isinstance(value,Path):return str(value)
            if isinstance(value,dict):return {key:plain(item) for key,item in value.items()}
            if isinstance(value,(tuple,list)):return [plain(item) for item in value]
            return value
        return plain(asdict(self))


def export_to_hf_config(config, path=None, *, include_training=False):
    """Export portable config.json metadata; this does not register an AutoModel.

    Our Config remains authoritative. No local data/cache paths are exported.
    A Transformers model/config adapter and weights are separate release work.
    """
    result = {
        'model_type':'bazza', 'architectures':['Bazza'], 'bazza_format_version':1,
        'backbone':{'checkpoint':config.backbone_checkpoint,'revision':config.backbone_revision,
                    'hidden_size':config.encoder_width,'quantization':config.quantization},
        'decision_head':{'reader':'bigru','hidden_size':config.hidden_width,
            'num_attention_heads':config.attention_heads,'low_steps':config.low_steps,
            'high_cycles':config.high_cycles,'gate_width':config.gate_width,
            'adapter_rank':config.adapter_rank,'final_logit_softcapping':config.softcap},
        'output_type':'variable_option_decision', 'backbone_frozen':True,
    }
    if include_training:
        result['bazza_training']={key:getattr(config,key) for key in (
            'learning_rate','muon_learning_rate','batch_size','token_budget','epochs','optimizer',
            'weight_decay','beta2','gradient_clip','precision','seed','schedule','warmup_examples',
            'warmup_batch_offset','decay_start_examples','minimum_lr_factor','target_key','weighted_loss')}
    if path is not None:
        import json
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        temporary.replace(path)
    return result
