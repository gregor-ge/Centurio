import math
from functools import partial

from timm.layers import LayerNorm2d
from timm.models.regnet import RegStage
from transformers import LlavaConfig
from transformers.activations import ACT2FN
import torch
from einops import rearrange, repeat
# from einops_exts import rearrange_many
from torch import einsum, nn


class LlavaMLPProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class LlavaMultiModalAdapter(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        # if config.adapter_type == "perceiver":
        #     self.adapter = LlavaPerceiver(config)
        # elif config.adapter_type == "ldpv2":
        #     self.adapter = LDPNetV2Adapter(config)
        if config.adapter_type == "window-pool":
            self.adapter = WindowPoolProjector(config)
        elif config.adapter_type == "window-shuffel":
            self.adapter = WindowShuffelProjector(config)
        elif config.adapter_type == "multiscale-pool":
            self.adapter = MultiscalePoolProjector(config)
        elif config.adapter_type == "multiscale-shuffel":
            self.adapter = MultiscaleShuffleProjector(config)

        # elif config.adapter_type == "reg-pool2":
        #     self.adapter = TwoPoolRegNetAdapter(config)
        # elif config.adapter_type == "multiscale-reg-pool2":
        #     self.adapter = MultiScaleTwoPoolRegNetAdapter(config)
        # elif config.adapter_type == "multiscale-reg-pool22":
        #     self.adapter = MultiScaleTwoPoolRegNet2Adapter(config)
        # elif config.adapter_type == "ldp-alter1":
        #     self.adapter = LDPNetAlter1Adapter(config)
        # elif config.adapter_type == "ldp-alter2":
        #     self.adapter = LDPNetAlter2Adapter(config)
        # elif config.adapter_type == "honeybee-c":
        #     self.adapter = CAbstractorAdapter(config)
        # elif config.adapter_type == "multiscale-mlp":
        #     self.adapter = LlavaMLPMultiScaleProjector(config)
        # elif config.adapter_type == "multiscale-ldpv2":
        #     self.adapter = LDPNetV2MultiScaleAdapter(config)
        # elif config.adapter_type == "multiscale-concat-ldpv2":
        #     self.adapter = ConcatMLPMultiScaleAdapter(config)
        # elif config.adapter_type == "multiscale-concat2-ldpv2":
        #     self.adapter = Concat2MLPMultiScaleAdapter(config)
        # elif config.adapter_type == "multiscale-hd-ldpv2":
        #     self.adapter = HDMLPMultiScaleAdapter(config)
        else:
            self.adapter = LlavaMLPProjector(config)

    def forward(self, image_features):
        return self.adapter(image_features)



class WindowMLPProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.multi_scale = getattr(config, "adapter_multi_scale", 2)
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        windows = 1 + self.multi_scale**2
        hidden_states = rearrange(hidden_states, "(b h) w d -> b (h w) d", h=windows)

        return hidden_states


class WindowPoolProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.multi_scale = getattr(config, "adapter_multi_scale", 2)
        self.pool = nn.AdaptiveAvgPool2d(getattr(config, "adapter_pool", 8))
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        b, num_tokens, c = hidden_states.shape
        h = int(math.sqrt(num_tokens))

        hidden_states = rearrange(hidden_states, "b (h w) d -> b d h w", h=h, w=h)
        hidden_states = self.pool(hidden_states)
        hidden_states = rearrange(hidden_states, "b d h w -> b (h w) d")

        windows = 1 + self.multi_scale**2
        hidden_states = rearrange(hidden_states, "(b h) w d -> b (h w) d", h=windows)
        return hidden_states


class WindowShuffelProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.multi_scale = getattr(config, "adapter_multi_scale", 2)
        self.scale_factor = getattr(config, "adapter_pool", 2)
        self.pixel_unshuffel = nn.PixelUnshuffle(self.scale_factor)
        self.linear_1 = nn.Linear(config.vision_config.hidden_size*(self.scale_factor**2), config.text_config.hidden_size, bias=True)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)



    def forward(self, image_features):
        bsz, seq, embed_dim = image_features.size()
        height = width = int(seq ** 0.5)
        hidden_states = rearrange(image_features, "b (w h) d -> b d w h", w=width, h=height)
        hidden_states = self.pixel_unshuffel(hidden_states)
        hidden_states = rearrange(hidden_states, "b d w h -> b (w h) d")

        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        windows = 1 + self.multi_scale ** 2
        hidden_states = rearrange(hidden_states, "(b h) w d -> b (h w) d", h=windows)
        return hidden_states





class MultiscalePoolProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.multi_scale = getattr(config, "adapter_multi_scale", 2)
        self.pool = nn.AvgPool2d(self.multi_scale)
        self.linear_1 = nn.Linear(config.vision_config.hidden_size*2, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        b, num_tokens, c = image_features.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        image_features = rearrange(image_features, "b (h w) d -> b d h w", h=h, w=h)

        steps = 1 + self.multi_scale**2
        low_res_features = image_features[::steps]
        high_res_features = image_features[[i for i in range(image_features.size(0)) if i%steps > 0]]

        merged_features = rearrange(high_res_features, "(b m) d h w -> b d h (m w)", m=self.multi_scale)
        merged_features = rearrange(merged_features, "(b m) d h w -> b d (m h) w", m=self.multi_scale)

        merged_features = self.pool(merged_features)

        concat_features = torch.cat([low_res_features, merged_features], dim=1)
        concat_features = rearrange(concat_features, "b d h w -> b (h w) d")

        hidden_states = self.linear_1(concat_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LDPNetV2MultiScaleAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.multi_scale = getattr(config, "adapter_multi_scale", 2)
        self.pool = nn.AvgPool2d(self.multi_scale)

        inc, ouc = config.vision_config.hidden_size*2, config.text_config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
        )
        self.dwn = nn.AvgPool2d(2) #TokenDownLayer((12, 12))
        self.peg = nn.Conv2d(ouc, ouc, 3, 1, 1, bias=True, groups=ouc) #PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        image_features = rearrange(x, "b (h w) d -> b d h w", h=h, w=h)

        steps = 1 + self.multi_scale ** 2
        low_res_features = image_features[::steps]
        high_res_features = image_features[[i for i in range(image_features.size(0)) if i % steps > 0]]

        merged_features = rearrange(high_res_features, "(b m) d h w -> b d h (m w)", m=self.multi_scale)
        merged_features = rearrange(merged_features, "(b m) d h w -> b d (m h) w", m=self.multi_scale)

        merged_features = self.pool(merged_features)

        concat_features = torch.cat([low_res_features, merged_features], dim=1)
        concat_features = rearrange(concat_features, "b d h w -> b (h w) d")

        x = self.mlp(concat_features)

        # x = self.dwn(x)
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = rearrange(x, "b (h w) d -> b d h w", h=h, w=h) #x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = self.peg(x) + x
        x = rearrange(x, "b d h w -> b (h w) d") #x.flatten(2).transpose(1, 2)

        return x


class MultiscaleShuffleProjector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.multi_scale = getattr(config, "adapter_multi_scale", 2)
        self.shuffle = nn.PixelUnshuffle(self.multi_scale)

        inc, ouc = config.vision_config.hidden_size*(1+self.multi_scale**2), config.text_config.hidden_size
        #
        self.mlp = nn.Sequential(
            nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
        )

        self.dwn = nn.AvgPool2d(2) #TokenDownLayer((12, 12))
        self.peg = nn.Conv2d(ouc, ouc, 3, 1, 1, bias=True, groups=ouc) #PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        image_features = rearrange(x, "b (h w) d -> b d h w", h=h, w=h)

        steps = 1 + self.multi_scale ** 2
        low_res_features = image_features[::steps]
        high_res_features = image_features[[i for i in range(image_features.size(0)) if i % steps > 0]]

        merged_features = rearrange(high_res_features, "(b m) d h w -> b d h (m w)", m=self.multi_scale)
        merged_features = rearrange(merged_features, "(b m) d h w -> b d (m h) w", m=self.multi_scale)

        merged_features = self.shuffle(merged_features)

        concat_features = torch.cat([low_res_features, merged_features], dim=1)
        concat_features = rearrange(concat_features, "b d h w -> b (h w) d")

        x = self.mlp(concat_features)

        # x = self.dwn(x)
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = rearrange(x, "b (h w) d -> b d h w", h=h, w=h) #x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = self.peg(x) + x
        x = rearrange(x, "b d h w -> b (h w) d") #x.flatten(2).transpose(1, 2)

        return x
#
#
# class Concat2MLPMultiScaleAdapter(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#
#         self.multi_scale = getattr(config, "adapter_multi_scale", 2)
#         self.shuffle = nn.PixelUnshuffle(self.multi_scale)
#
#         inc, ouc = config.vision_config.hidden_size*(1+self.multi_scale**2), config.text_config.hidden_size
#         #
#         # self.mlp = nn.Sequential(
#         #     nn.Linear(inc, inc), nn.GELU(), nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
#         # )
#         self.mlp = nn.Sequential(
#             nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
#         )
#
#         self.dwn = nn.AvgPool2d(2) #TokenDownLayer((12, 12))
#         self.peg = nn.Conv2d(ouc, ouc, 3, 1, 1, bias=True, groups=ouc) #PosInjectLayer(ouc, ouc, stride=1)
#
#     def forward(self, x):
#         b, num_tokens, c = x.shape
#         h = int(math.sqrt(num_tokens))
#         assert h * h == num_tokens
#         image_features = rearrange(x, "b (h w) d -> b d h w", h=h, w=h)
#
#         steps = 1 + self.multi_scale ** 2
#         low_res_features = image_features[::steps]
#         high_res_features = image_features[[i for i in range(image_features.size(0)) if i % steps > 0]]
#
#         merged_features = rearrange(high_res_features, "(b m) d h w -> b d h (m w)", m=self.multi_scale)
#         merged_features = rearrange(merged_features, "(b m) d h w -> b d (m h) w", m=self.multi_scale)
#
#         merged_features = self.shuffle(merged_features)
#
#         concat_features = torch.cat([low_res_features, merged_features], dim=1)
#         concat_features = rearrange(concat_features, "b d h w -> b (h w) d")
#
#         x = self.mlp(concat_features)
#
#         # x = self.dwn(x)
#         b, num_tokens, c = x.shape
#         h = int(math.sqrt(num_tokens))
#         assert h * h == num_tokens
#         x = rearrange(x, "b (h w) d -> b d h w", h=h, w=h) #x.permute(0, 2, 1).reshape(b, -1, h, h)
#         x = self.dwn(x)
#         x = self.peg(x) + x
#         x = rearrange(x, "b d h w -> b (h w) d") #x.flatten(2).transpose(1, 2)
#
#         return x
#
# class HDMLPMultiScaleAdapter(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#
#         self.multi_scale = getattr(config, "adapter_multi_scale", 2)
#         self.shuffle = nn.PixelUnshuffle(self.multi_scale)
#
#         inc, ouc = config.vision_config.hidden_size*self.multi_scale, config.text_config.hidden_size
#
#         self.mlp = nn.Sequential(
#             nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
#         )
#         self.down = nn.AvgPool2d(2)
#         self.up = nn.Upsample(scale_factor=2, mode='area')
#
#         self.peg1 = nn.Conv2d(inc, inc, 3, 1, 1, bias=True, groups=inc) #PosInjectLayer(ouc, ouc, stride=1)
#         self.cnn = nn.Conv2d(inc, inc, 1, 1, 1, bias=True)
#
#         self.dwn = nn.AvgPool2d(2) #TokenDownLayer((12, 12))
#         self.peg2 = nn.Conv2d(ouc, ouc, 3, 1, 1, bias=True, groups=ouc) #PosInjectLayer(ouc, ouc, stride=1)
#
#     def forward(self, x):
#         b, num_tokens, c = x.shape
#         h = int(math.sqrt(num_tokens))
#         assert h * h == num_tokens
#         image_features = rearrange(x, "b (h w) d -> b d h w", h=h, w=h)
#
#         steps = 1 + self.multi_scale ** 2
#         low_res_features = image_features[::steps]
#         high_res_features = image_features[[i for i in range(image_features.size(0)) if i % steps > 0]]
#
#         merged_features = rearrange(high_res_features, "(b m) d h w -> b d h (m w)", m=self.multi_scale)
#         merged_features = rearrange(merged_features, "(b m) d h w -> b d (m h) w", m=self.multi_scale)
#
#         low_res_features = self.up(low_res_features)
#         concat_features = torch.cat([low_res_features, merged_features], dim=1)
#
#         concat_features = self.cnn(concat_features)
#         concat_features = self.peg1(concat_features) + concat_features
#         concat_features = self.down(concat_features)
#
#         concat_features = rearrange(concat_features, "b d h w -> b (h w) d")
#
#         x = self.mlp(concat_features)
#
#         # x = self.dwn(x)
#         b, num_tokens, c = x.shape
#         h = int(math.sqrt(num_tokens))
#         assert h * h == num_tokens
#         x = rearrange(x, "b (h w) d -> b d h w", h=h, w=h) #x.permute(0, 2, 1).reshape(b, -1, h, h)
#         x = self.dwn(x)
#         x = self.peg2(x) + x
#         x = rearrange(x, "b d h w -> b (h w) d") #x.flatten(2).transpose(1, 2)
#
#         return x
#
# class LDPNetAlter1Adapter(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         inc, ouc = config.vision_config.hidden_size, config.text_config.hidden_size
#
#         self.mlp = nn.Sequential(
#             nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
#         )
#         self.dwn = nn.AvgPool2d(2) #TokenDownLayer((12, 12))
#         self.peg = nn.Conv2d(ouc, ouc, 3, 1, 1, bias=True, groups=ouc) #PosInjectLayer(ouc, ouc, stride=1)
#
#     def forward(self, x):
#         x = self.mlp(x)
#
#         # x = self.dwn(x)
#         b, num_tokens, c = x.shape
#         h = int(math.sqrt(num_tokens))
#         assert h * h == num_tokens
#         x = rearrange(x, "b (h w) d -> b d h w", h=h, w=h) #x.permute(0, 2, 1).reshape(b, -1, h, h)
#         x = self.peg(x) + x
#         x = self.dwn(x)
#         x = rearrange(x, "b d h w -> b (h w) d") #x.flatten(2).transpose(1, 2)
#
#         return x
#
#
# class MultiScaleTwoPoolRegNetAdapter(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.multi_scale = getattr(config, "adapter_multi_scale", 2)
#         self.shuffle = nn.PixelUnshuffle(self.multi_scale)
#         encoder_hidden_size = config.vision_config.hidden_size * self.multi_scale
#         output_hidden_size = config.text_config.hidden_size
#
#         # RegBlock = ResBlock + SE
#         RegBlock = partial(
#             RegStage,
#             stride=1,
#             dilation=1,
#             act_layer=nn.SiLU,  # GELU has no in_place kwarg
#             norm_layer=LayerNorm2d,
#         )
#
#         s1 = RegBlock(
#             1,
#             output_hidden_size,
#             output_hidden_size,
#         )
#         sampler1 = nn.AvgPool2d(2)
#         s2 = RegBlock(
#             1,
#             output_hidden_size,
#             output_hidden_size,
#         )
#         sampler2 = nn.AvgPool2d(2)
#
#         self.up = nn.Upsample(scale_factor=2, mode='area')
#         self.net = nn.Sequential(s1, sampler1, s2, sampler2)
#         self.mlp = nn.Sequential(
#             nn.Linear(encoder_hidden_size, output_hidden_size),
#             nn.GELU(),
#             nn.Linear(output_hidden_size, output_hidden_size)
#         )
#         self.out = nn.Sequential(
#             nn.Linear(output_hidden_size, output_hidden_size)
#         )
#
#     def forward(self, x):
#         b, num_tokens, c = x.shape
#         h = int(math.sqrt(num_tokens))
#         assert h * h == num_tokens
#         image_features = rearrange(x, "b (h w) d -> b d h w", h=h, w=h)
#
#         steps = 1 + self.multi_scale ** 2
#         low_res_features = image_features[::steps]
#         high_res_features = image_features[[i for i in range(image_features.size(0)) if i % steps > 0]]
#
#         merged_features = rearrange(high_res_features, "(b m) d h w -> b d h (m w)", m=self.multi_scale)
#         merged_features = rearrange(merged_features, "(b m) d h w -> b d (m h) w", m=self.multi_scale)
#
#         low_res_features = self.up(low_res_features)
#         x = torch.cat([low_res_features, merged_features], dim=1)
#         x = rearrange(x, "b d h w -> b (h w) d")
#         x = self.mlp(x)
#         hw = int(x.size(1) ** 0.5)
#         x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
#         x = self.net(x)
#         x = rearrange(x, "b d h w -> b (h w) d")
#         x = self.out(x)
#
#         return x
#
# class MultiScaleTwoPoolRegNet2Adapter(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.multi_scale = getattr(config, "adapter_multi_scale", 2)
#         self.shuffle = nn.PixelUnshuffle(self.multi_scale)
#         encoder_hidden_size = config.vision_config.hidden_size * self.multi_scale
#         output_hidden_size = config.text_config.hidden_size
#
#         # RegBlock = ResBlock + SE
#         RegBlock = partial(
#             RegStage,
#             stride=1,
#             dilation=1,
#             act_layer=nn.SiLU,  # GELU has no in_place kwarg
#             norm_layer=LayerNorm2d,
#         )
#
#         s1 = RegBlock(
#             3,
#             encoder_hidden_size,
#             encoder_hidden_size,
#         )
#         self.sampler = nn.AvgPool2d(2)
#
#         self.up = nn.Upsample(scale_factor=2, mode='area')
#         self.net = s1
#         self.mlp = nn.Sequential(
#             nn.Conv2d(encoder_hidden_size, output_hidden_size, 1),
#             nn.GELU(),
#             nn.Conv2d(output_hidden_size, output_hidden_size, 1),
#         )
#
#     def forward(self, x):
#         b, num_tokens, c = x.shape
#         h = int(math.sqrt(num_tokens))
#         assert h * h == num_tokens
#         image_features = rearrange(x, "b (h w) d -> b d h w", h=h, w=h)
#
#         steps = 1 + self.multi_scale ** 2
#         low_res_features = image_features[::steps]
#         high_res_features = image_features[[i for i in range(image_features.size(0)) if i % steps > 0]]
#
#         merged_features = rearrange(high_res_features, "(b m) d h w -> b d h (m w)", m=self.multi_scale)
#         merged_features = rearrange(merged_features, "(b m) d h w -> b d (m h) w", m=self.multi_scale)
#
#         low_res_features = self.up(low_res_features)
#         x = torch.cat([low_res_features, merged_features], dim=1)
#         x = self.net(x)
#         x = self.sampler(x)
#         x = self.mlp(x)
#         x = self.sampler(x)
#         x = rearrange(x, "b d h w -> b (h w) d")
#
#         return x
# class TwoPoolRegNetAdapter(nn.Module):
#     """C-Abstractor"""
#     def __init__(self, config):
#         super().__init__()
#         encoder_hidden_size = config.vision_config.hidden_size
#         output_hidden_size = config.text_config.hidden_size
#
#         # RegBlock = ResBlock + SE
#         RegBlock = partial(
#             RegStage,
#             stride=1,
#             dilation=1,
#             act_layer=nn.SiLU,  # GELU has no in_place kwarg
#             norm_layer=LayerNorm2d,
#         )
#
#         s1 = RegBlock(
#             1,
#             output_hidden_size,
#             output_hidden_size,
#         )
#         sampler1 = nn.AvgPool2d(2)
#         s2 = RegBlock(
#             1,
#             output_hidden_size,
#             output_hidden_size,
#         )
#         sampler2 = nn.AvgPool2d(2)
#
#         self.net = nn.Sequential(s1, sampler1, s2, sampler2)
#         self.mlp = nn.Sequential(
#             nn.Linear(encoder_hidden_size, output_hidden_size),
#             nn.GELU(),
#             nn.Linear(output_hidden_size, output_hidden_size)
#         )
#         self.out = nn.Sequential(
#             nn.Linear(output_hidden_size, output_hidden_size)
#         )
#
#     def forward(self, x):
#         # x: [B, L, dim]
#         # x = x[:, 1:]  # drop cls token and 2d forward
#         hw = int(x.size(1) ** 0.5)
#         x = self.mlp(x)
#         x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
#         x = self.net(x)
#         x = rearrange(x, "b d h w -> b (h w) d")
#         x = self.out(x)
#
#         return x
#
# class LDPNetAlter2Adapter(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         inc, ouc = config.vision_config.hidden_size, config.text_config.hidden_size
#
#         self.dwn = nn.AvgPool2d(2) #TokenDownLayer((12, 12))
#         self.peg = nn.Conv2d(inc, inc, 3, 1, 1, bias=True, groups=inc) #PosInjectLayer(ouc, ouc, stride=1)
#         self.mlp = nn.Sequential(
#             nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
#         )
#
#     def forward(self, x):
#         b, num_tokens, c = x.shape
#         h = int(math.sqrt(num_tokens))
#         assert h * h == num_tokens
#         x = rearrange(x, "b (h w) d -> b d h w", h=h, w=h) #x.permute(0, 2, 1).reshape(b, -1, h, h)
#         x = self.peg(x) + x
#         x = self.dwn(x)
#         x = rearrange(x, "b d h w -> b (h w) d") #x.flatten(2).transpose(1, 2)
#         x = self.mlp(x)
#
#         return x
#
# # Honeybee https://github.com/kakaobrain/honeybee/blob/main/honeybee/projectors.py; adapted by me
# class CAbstractorAdapter(nn.Module):
#     """C-Abstractor"""
#     def __init__(self, config):
#         super().__init__()
#         encoder_hidden_size = config.vision_config.hidden_size
#         hidden_size = config.vision_config.hidden_size
#         output_hidden_size = config.text_config.hidden_size
#         depth = getattr(config, "adapter_layers", 6) // 2
#
#         # RegBlock = ResBlock + SE
#         RegBlock = partial(
#             RegStage,
#             stride=1,
#             dilation=1,
#             act_layer=nn.SiLU,  # GELU has no in_place kwarg
#             norm_layer=LayerNorm2d,
#         )
#
#         s1 = RegBlock(
#             depth,
#             encoder_hidden_size,
#             hidden_size,
#         )
#         sampler = nn.AvgPool2d(2)
#         s2 = RegBlock(
#             depth,
#             hidden_size,
#             hidden_size,
#         )
#
#         self.net = nn.Sequential(s1, sampler, s2)
#         self.readout = nn.Sequential(
#             nn.Linear(config.vision_config.hidden_size, output_hidden_size),
#             nn.GELU(),
#             nn.Linear(output_hidden_size, output_hidden_size)
#         )
#
#     def forward(self, x):
#         # x: [B, L, dim]
#         # x = x[:, 1:]  # drop cls token and 2d forward
#         hw = int(x.size(1) ** 0.5)
#         x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
#         x = self.net(x)
#         x = rearrange(x, "b d h w -> b (h w) d")
#         x = self.readout(x)
#
#         return x
#
# ## MobileVLM2  https://github.com/Meituan-AutoML/MobileVLM/blob/main/mobilevlm/model/vision_projector.py
# class LDPNetV2Adapter(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         inc, ouc = config.vision_config.hidden_size, config.text_config.hidden_size
#         pool = getattr(config, "adapter_pool", 8)
#         self.mlp = nn.Sequential(
#             nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
#         )
#         self.dwn = nn.AdaptiveAvgPool2d(pool) #TokenDownLayer((12, 12))
#         self.peg = nn.Conv2d(ouc, ouc, 3, 1, 1, bias=True, groups=ouc) #PosInjectLayer(ouc, ouc, stride=1)
#
#     def forward(self, x):
#         x = self.mlp(x)
#
#         # x = self.dwn(x)
#         b, num_tokens, c = x.shape
#         h = int(math.sqrt(num_tokens))
#         assert h * h == num_tokens
#         x = rearrange(x, "b (h w) d -> b d h w", h=h, w=h) #x.permute(0, 2, 1).reshape(b, -1, h, h)
#         x = self.dwn(x)
#         # x = x.flatten(2).transpose(1, 2)
#
#         # x = self.peg(x)
#         # b, num_tokens, c = x.shape
#         # h = int(math.sqrt(num_tokens))
#         # assert h * h == num_tokens
#         # cnn_feat = x.transpose(1, 2).view(b, -1, h, h)
#         x = self.peg(x) + x
#         x = rearrange(x, "b d h w -> b (h w) d") #x.flatten(2).transpose(1, 2)
#
#         return x
#
# # class TokenDownLayer(nn.Module):
# #     def __init__(self, shape) -> None:
# #         super().__init__()
# #         self.dwn = nn.Sequential(
# #             nn.AdaptiveAvgPool2d(shape)
# #         )
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         b, num_tokens, c = x.shape
# #         h = int(math.sqrt(num_tokens))
# #         assert h * h == num_tokens
# #         x = x.permute(0, 2, 1).reshape(b, -1, h, h)
# #         x = self.dwn(x)
# #         x = x.flatten(2).transpose(1, 2)
# #         return x
#
# # class PosInjectLayer(nn.Module):
# #     # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
# #     def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
# #         super().__init__()
# #         self.peg = nn.Sequential(
# #             nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
# #         )
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         b, num_tokens, c = x.shape
# #         h = int(math.sqrt(num_tokens))
# #         assert h * h == num_tokens
# #         cnn_feat = x.transpose(1, 2).view(b, c, h, h)
# #         x = self.peg(cnn_feat) + cnn_feat
# #         x = x.flatten(2).transpose(1, 2)
# #         return x
#
#
#
#
#
#
# ### PERCEIVER based on Idefics
# class LlavaPerceiver(nn.Module):
#     def __init__(self, config: LlavaConfig):
#         super().__init__()
#
#         dim = getattr(config, "adapter_ps_dim", 768)
#         depth = getattr(config, "adapter_layers", 1)
#         latents = getattr(config, "adapter_ps_queries", 64)
#
#         self.ps = PerceiverResampler(depth=depth, latents=latents)
#         self.linear_in = nn.Linear(config.vision_config.hidden_size, dim, bias=True)
#         self.linear_out = nn.Linear(dim, config.text_config.hidden_size, bias=True)
#
#     def forward(self, x):
#         x_in = self.linear_in(x)
#         x_ps = self.ps(x_in)
#         x_out = self.linear_out(x_ps)
#         return x_out
#
# def FeedForward(dim, mult=4):
#     inner_dim = int(dim * mult)
#     return nn.Sequential(
#         nn.LayerNorm(dim),
#         nn.Linear(dim, inner_dim, bias=False),
#         nn.GELU(),
#         nn.Linear(inner_dim, dim, bias=False),
#     )
#
# class PerceiverAttention(nn.Module):
#     def __init__(self, *, dim, dim_head=64, heads=8, qk_layer_norms=True):
#         super().__init__()
#         self.qk_layer_norms = qk_layer_norms
#         self.scale = dim_head**-0.5
#         self.heads = heads
#         inner_dim = dim_head * heads
#
#         self.norm_media = nn.LayerNorm(dim)
#         self.norm_latents = nn.LayerNorm(dim)
#         if self.qk_layer_norms:
#             self.q_layer_norm = nn.LayerNorm(dim_head)
#             self.k_layer_norm = nn.LayerNorm(dim_head)
#
#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)
#
#     def forward(self, x, latents):
#         """
#         Args:
#             x (torch.Tensor): image features
#                 shape (b, T, n1, D)
#             latent (torch.Tensor): latent features
#                 shape (b, T, n2, D)
#         """
#         x = self.norm_media(x)
#         latents = self.norm_latents(latents)
#
#         h = self.heads
#
#         q = self.to_q(latents)
#         kv_input = torch.cat((x, latents), dim=-2)
#         k, v = self.to_kv(kv_input).chunk(2, dim=-1)
#         q = rearrange(q, "b n (h d) -> b h n d", h=h)
#         k = rearrange(k, "b n (h d) -> b h n d", h=h)
#         v = rearrange(v, "b n (h d) -> b h n d", h=h)
#
#         if self.qk_layer_norms:
#             q = self.q_layer_norm(q)
#             k = self.k_layer_norm(k)
#
#         q = q * self.scale
#
#         # attention
#         sim = einsum("... i d, ... j d  -> ... i j", q, k)
#         sim = sim - sim.amax(dim=-1, keepdim=True).detach()
#         attn = sim.softmax(dim=-1)
#
#         out = einsum("... i j, ... j d -> ... i d", attn, v)
#         out = rearrange(out, "b h n d -> b n (h d)", h=h)
#         return self.to_out(out)
#
#
# class PerceiverResampler(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim=768,
#         depth=6,
#         dim_head=64,
#         heads=12,
#         num_latents=32,
#         ff_mult=4,
#         **kwargs
#     ):
#         super().__init__()
#         self.latents = nn.Parameter(torch.randn(num_latents, dim))
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(
#                 nn.ModuleList(
#                     [
#                         PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
#                         FeedForward(dim=dim, mult=ff_mult),
#                     ]
#                 )
#             )
#
#         self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         """
#         Args:
#             x (torch.Tensor): image features
#                 shape (b, v, other_dim)
#         Returns:
#             shape (b, n, dim) where n is self.num_latents
#         """
#
#         # blocks
#         latents = self.latents.expand(x.shape[0], -1, -1)
#         for attn, ff in self.layers:
#             latents = attn(x, latents) + latents
#             latents = ff(latents) + latents
#         return self.norm(latents)