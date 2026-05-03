# Copyright contributors to the TSFM project
# Vendored from https://github.com/ibm-granite/granite-tsfm
# Stripped to inference-only paths for ibm-granite/granite-timeseries-ttm-r1
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TinyTimeMixerConfig(PretrainedConfig):
    model_type = "tinytimemixer"
    attribute_map = {"hidden_size": "d_model", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        context_length: int = 64,
        patch_length: int = 8,
        num_input_channels: int = 1,
        prediction_length: int = 16,
        patch_stride: int = 8,
        prediction_channel_indices=None,
        d_model: int = 16,
        expansion_factor: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp: str = "LayerNorm",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        use_positional_encoding: bool = False,
        positional_encoding_type: str = "sincos",
        scaling: Optional[Union[str, bool]] = "std",
        loss: Optional[str] = "mse",
        init_std: float = 0.02,
        post_init: bool = False,
        norm_eps: float = 1e-5,
        adaptive_patching_levels: int = 0,
        resolution_prefix_tuning: bool = False,
        frequency_token_vocab_size: int = 5,
        head_dropout: float = 0.2,
        num_parallel_samples: int = 100,
        decoder_num_layers: int = 8,
        decoder_d_model: int = 8,
        decoder_adaptive_patching_levels: int = 0,
        decoder_raw_residual: bool = False,
        decoder_mode: str = "common_channel",
        use_decoder: bool = True,
        enable_forecast_channel_mixing: bool = False,
        categorical_vocab_size_list=None,
        prediction_filter_length=None,
        init_linear: str = "pytorch",
        init_embed: str = "pytorch",
        mask_value: int = 0,
        **kwargs,
    ):
        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.patch_length = patch_length
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.scaling = scaling
        self.head_dropout = head_dropout
        self.patch_last = True
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_type = positional_encoding_type
        self.prediction_length = prediction_length
        self.prediction_channel_indices = prediction_channel_indices
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads
        self.init_std = init_std
        self.post_init = post_init
        self.loss = loss
        self.num_parallel_samples = num_parallel_samples
        self.norm_eps = norm_eps
        self.use_decoder = use_decoder
        self.adaptive_patching_levels = adaptive_patching_levels
        self.resolution_prefix_tuning = resolution_prefix_tuning
        self.decoder_num_layers = decoder_num_layers
        self.decoder_adaptive_patching_levels = decoder_adaptive_patching_levels
        self.decoder_raw_residual = decoder_raw_residual
        self.decoder_mode = decoder_mode
        self.enable_forecast_channel_mixing = enable_forecast_channel_mixing
        self.frequency_token_vocab_size = frequency_token_vocab_size
        self.d_model = d_model
        self.patch_stride = patch_stride
        self.decoder_d_model = decoder_d_model
        self.categorical_vocab_size_list = categorical_vocab_size_list
        self.init_processing = False
        self.prediction_filter_length = prediction_filter_length
        self.init_linear = init_linear
        self.init_embed = init_embed
        self.mask_value = mask_value
        self.masked_context_length = None
        super().__init__(**kwargs)

    def check_and_init_preprocessing(self):
        self.init_processing = True
        if not hasattr(self, "num_patches"):
            ctx = self.masked_context_length if self.masked_context_length is not None else self.context_length
            self.num_patches = (max(ctx, self.patch_length) - self.patch_length) // self.patch_stride + 1
        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices = sorted(self.prediction_channel_indices)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TinyTimeMixerGatedAttention(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        return inputs * self.attn_softmax(self.attn_layer(inputs))


class TinyTimeMixerBatchNorm(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        out = inputs.transpose(1, 2)
        out = self.batchnorm(out)
        return out.transpose(1, 2)


class TinyTimeMixerNormLayer(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.norm_mlp = config.norm_mlp
        if "batch" in config.norm_mlp.lower():
            self.norm = TinyTimeMixerBatchNorm(config)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        if "batch" in self.norm_mlp.lower():
            reshaped = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
            reshaped = self.norm(reshaped)
            return reshaped.reshape(inputs.shape)
        return self.norm(inputs)


class TinyTimeMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        hidden = in_features * config.expansion_factor
        self.fc1 = nn.Linear(in_features, hidden)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(hidden, out_features)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor):
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        return self.dropout2(inputs)


class PatchMixerBlock(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.norm = TinyTimeMixerNormLayer(config)
        self.gated_attn = config.gated_attn
        self.mlp = TinyTimeMixerMLP(config.num_patches, config.num_patches, config)
        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(config.num_patches, config.num_patches)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        hidden_state = hidden_state.transpose(2, 3)
        hidden_state = self.mlp(hidden_state)
        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)
        hidden_state = hidden_state.transpose(2, 3)
        return hidden_state + residual


class FeatureMixerBlock(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.norm = TinyTimeMixerNormLayer(config)
        self.gated_attn = config.gated_attn
        self.mlp = TinyTimeMixerMLP(config.d_model, config.d_model, config)
        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(config.d_model, config.d_model)

    def forward(self, hidden: torch.Tensor):
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)
        if self.gated_attn:
            hidden = self.gating_block(hidden)
        return hidden + residual


class TinyTimeMixerLayer(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        if config.num_patches > 1:
            self.patch_mixer = PatchMixerBlock(config=config)
        self.feature_mixer = FeatureMixerBlock(config=config)
        self.num_patches = config.num_patches

    def forward(self, hidden: torch.Tensor):
        if self.num_patches > 1:
            hidden = self.patch_mixer(hidden)
        return self.feature_mixer(hidden)


class TinyTimeMixerAdaptivePatchingBlock(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig, adapt_patch_level: int):
        super().__init__()
        temp_config = copy.deepcopy(config)
        self.adaptive_patch_factor = 2 ** adapt_patch_level
        if config.d_model // self.adaptive_patch_factor <= 4:
            self.adaptive_patch_factor = 1
        if config.d_model % self.adaptive_patch_factor != 0:
            raise ValueError("d_model must be divisible by 2^adapt_patch_level")
        temp_config.num_patches = temp_config.num_patches * self.adaptive_patch_factor
        temp_config.d_model = temp_config.d_model // self.adaptive_patch_factor
        self.mixer_layers = nn.ModuleList([TinyTimeMixerLayer(temp_config) for _ in range(temp_config.num_layers)])

    def forward(self, hidden: torch.Tensor):
        hidden = hidden.reshape(
            hidden.shape[0], hidden.shape[1],
            hidden.shape[2] * self.adaptive_patch_factor,
            hidden.shape[3] // self.adaptive_patch_factor,
        )
        for mod in self.mixer_layers:
            hidden = mod(hidden)
        return hidden.reshape(
            hidden.shape[0], hidden.shape[1],
            hidden.shape[2] // self.adaptive_patch_factor,
            hidden.shape[3] * self.adaptive_patch_factor,
        )


class TinyTimeMixerBlock(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.adaptive_patching_levels = config.adaptive_patching_levels
        if self.adaptive_patching_levels > 0:
            self.mixers = nn.ModuleList([
                TinyTimeMixerAdaptivePatchingBlock(config=config, adapt_patch_level=i)
                for i in reversed(range(config.adaptive_patching_levels))
            ])
        else:
            self.mixers = nn.ModuleList([TinyTimeMixerLayer(config=config) for _ in range(config.num_layers)])

    def forward(self, hidden_state, output_hidden_states: bool = False):
        all_hidden_states = []
        embedding = hidden_state
        for mod in self.mixers:
            embedding = mod(embedding)
            if output_hidden_states and self.adaptive_patching_levels == 0:
                all_hidden_states.append(embedding)
        return embedding, all_hidden_states if output_hidden_states else None


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class TinyTimeMixerStdScaler(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.dim = getattr(config, "scaling_dim", 1)
        self.keepdim = getattr(config, "keepdim", True)
        self.minimum_scale = getattr(config, "minimum_scale", 1e-5)

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor):
        denom = observed_indicator.sum(self.dim, keepdim=self.keepdim).clamp_min(1)
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denom
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denom
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


# ---------------------------------------------------------------------------
# Patchify
# ---------------------------------------------------------------------------

class TinyTimeMixerPatchify(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.sequence_length = config.masked_context_length if config.masked_context_length is not None else config.context_length
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride
        if self.sequence_length <= self.patch_length:
            raise ValueError("sequence_length must be greater than patch_length")
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        if past_values.shape[-2] != self.sequence_length:
            raise ValueError(f"Input length {past_values.shape[-2]} != expected {self.sequence_length}")
        output = past_values[:, self.sequence_start:, :]
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        return output.transpose(-2, -3).contiguous()


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------

class TinyTimeMixerPreTrainedModel(PreTrainedModel):
    config_class = TinyTimeMixerConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False
    # transformers 5.x changed _tied_weights_keys (list) → all_tied_weights_keys (dict)
    all_tied_weights_keys = {}

    def _init_weights(self, module):
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, TinyTimeMixerBatchNorm):
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            if self.config.init_linear == "normal":
                module.weight.data.normal_(mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif self.config.init_linear in ("uniform", "xavier_uniform"):
                (nn.init.uniform_ if self.config.init_linear == "uniform" else nn.init.xavier_uniform_)(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            else:
                module.reset_parameters()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@dataclass
class TinyTimeMixerEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TinyTimeMixerEncoder(TinyTimeMixerPreTrainedModel):
    def __init__(self, config: TinyTimeMixerConfig):
        if not config.init_processing:
            config.check_and_init_preprocessing()
        super().__init__(config)
        self.use_return_dict = config.use_return_dict
        self.patcher = nn.Linear(config.patch_length, config.d_model)
        self.positional_encoder = None
        self.mlp_mixer_encoder = TinyTimeMixerBlock(config=config)

    def forward(self, past_values, output_hidden_states=False, return_dict=None, freq_token=None):
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        patches = self.patcher(past_values)
        last_hidden_state, hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)
        if not return_dict:
            return (last_hidden_state, hidden_states)
        return TinyTimeMixerEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

@dataclass
class TinyTimeMixerModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patch_input: torch.FloatTensor = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


class TinyTimeMixerModel(TinyTimeMixerPreTrainedModel):
    def __init__(self, config: TinyTimeMixerConfig):
        if not config.init_processing:
            config.check_and_init_preprocessing()
        super().__init__(config)
        self.use_return_dict = config.use_return_dict
        self.encoder = TinyTimeMixerEncoder(config)
        self.patching = TinyTimeMixerPatchify(config)
        self.scaler = TinyTimeMixerStdScaler(config)

    def forward(self, past_values, past_observed_mask=None, output_hidden_states=False, return_dict=None, freq_token=None):
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
        patched_x = self.patching(scaled_past_values)
        encoder_output = self.encoder(patched_x, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if isinstance(encoder_output, tuple):
            encoder_output = TinyTimeMixerEncoderOutput(*encoder_output)
        if not return_dict:
            return (encoder_output.last_hidden_state, encoder_output.hidden_states, patched_x, loc, scale)
        return TinyTimeMixerModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patch_input=patched_x,
            loc=loc,
            scale=scale,
        )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TinyTimeMixerDecoder(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.adapter = nn.Linear(config.d_model, config.decoder_d_model) if config.d_model != config.decoder_d_model else None
        decoder_config = copy.deepcopy(config)
        decoder_config.num_layers = config.decoder_num_layers
        decoder_config.d_model = config.decoder_d_model
        decoder_config.dropout = config.head_dropout
        decoder_config.adaptive_patching_levels = config.decoder_adaptive_patching_levels
        decoder_config.mode = config.decoder_mode
        self.decoder_block = TinyTimeMixerBlock(decoder_config)

    def forward(self, hidden_state, patch_input=None, output_hidden_states=False, static_categorical_values=None):
        decoder_input = self.adapter(hidden_state) if self.adapter is not None else hidden_state
        decoder_output, hidden_states = self.decoder_block(hidden_state=decoder_input, output_hidden_states=output_hidden_states)
        return decoder_output, hidden_states


# ---------------------------------------------------------------------------
# Prediction head
# ---------------------------------------------------------------------------

class TinyTimeMixerForPredictionHead(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.prediction_channel_indices = (
            sorted(config.prediction_channel_indices) if config.prediction_channel_indices is not None else None
        )
        self.prediction_filter_length = config.prediction_filter_length
        self.dropout_layer = nn.Dropout(config.head_dropout)
        head_d_model = config.decoder_d_model if config.use_decoder else config.d_model
        self.base_forecast_block = nn.Linear(config.num_patches * head_d_model, config.prediction_length)
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, hidden_features, past_values=None, future_values=None):
        hidden_features = self.flatten(hidden_features)
        hidden_features = self.dropout_layer(hidden_features)
        forecast = self.base_forecast_block(hidden_features).transpose(-1, -2)
        if self.prediction_channel_indices is not None:
            forecast = forecast[..., self.prediction_channel_indices]
        if self.prediction_filter_length is not None:
            forecast = forecast[:, :self.prediction_filter_length, :]
        return forecast


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

@dataclass
class TinyTimeMixerForPredictionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    backbone_hidden_state: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


class TinyTimeMixerForPrediction(TinyTimeMixerPreTrainedModel):
    def __init__(self, config: TinyTimeMixerConfig):
        config.check_and_init_preprocessing()
        super().__init__(config)
        self.config = config
        self.use_return_dict = config.use_return_dict
        self.prediction_channel_indices = config.prediction_channel_indices
        self.num_input_channels = config.num_input_channels
        self.backbone = TinyTimeMixerModel(config)
        self.use_decoder = config.use_decoder
        if config.use_decoder:
            self.decoder = TinyTimeMixerDecoder(config)
        self.head = TinyTimeMixerForPredictionHead(config=config)
        if config.post_init:
            self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
    ) -> TinyTimeMixerForPredictionOutput:
        if past_values.dim() != 3:
            raise ValueError("`past_values` must be (batch_size, sequence_length, num_input_channels)")
        ctx = self.config.masked_context_length if self.config.masked_context_length is not None else self.config.context_length
        if past_values.shape[1] > ctx:
            past_values = past_values[:, -ctx:, :]
        elif past_values.shape[1] < ctx:
            raise ValueError("past_values is shorter than TTM context_length")

        return_dict = return_dict if return_dict is not None else self.use_return_dict
        model_output = self.backbone(past_values, past_observed_mask=past_observed_mask,
                                     output_hidden_states=output_hidden_states, return_dict=return_dict)
        if isinstance(model_output, tuple):
            model_output = TinyTimeMixerModelOutput(*model_output)

        decoder_input = model_output.last_hidden_state
        hidden_states = model_output.hidden_states

        if self.use_decoder:
            decoder_output, decoder_hidden_states = self.decoder(
                hidden_state=decoder_input,
                patch_input=model_output.patch_input,
                output_hidden_states=output_hidden_states,
            )
            if decoder_hidden_states and hidden_states:
                hidden_states.extend(decoder_hidden_states)
        else:
            decoder_output = decoder_input

        y_hat = self.head(decoder_output, past_values=past_values)

        loc = model_output.loc
        scale = model_output.scale
        if self.prediction_channel_indices is not None:
            loc = loc[..., self.prediction_channel_indices]
            scale = scale[..., self.prediction_channel_indices]

        y_hat = y_hat * scale + loc

        loss_val = None
        if future_values is not None and return_loss:
            loss_val = nn.MSELoss()(y_hat, future_values)

        if not return_dict:
            return (loss_val, y_hat, model_output.last_hidden_state, decoder_output, hidden_states, loc, scale)

        return TinyTimeMixerForPredictionOutput(
            loss=loss_val,
            prediction_outputs=y_hat,
            backbone_hidden_state=model_output.last_hidden_state,
            decoder_hidden_state=decoder_output,
            hidden_states=hidden_states,
            loc=loc,
            scale=scale,
        )
