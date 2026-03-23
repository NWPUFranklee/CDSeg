# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom
from safetensors.torch import load_file
from einops import rearrange
import numpy as np
import os, sys
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
# Make the local dinov3 package importable as top-level `dinov3`.
# The repository contains `cat_seg/dinov3/dinov3/...`. Adding
# the parent folder `cat_seg/dinov3` to sys.path lets `import dinov3` work.
_dinov3_parent = os.path.join(os.path.dirname(__file__), "dinov3")
if _dinov3_parent not in sys.path:
    sys.path.insert(0, _dinov3_parent)

_pvt_parent = os.path.join(os.path.dirname(__file__), "PVTV2")
if _pvt_parent not in sys.path:
    sys.path.insert(0, _pvt_parent)

from dinov3.models.vision_transformer import vit_base, vit_large, vit_small
from transformers import AutoImageProcessor, AutoModel
from .PVTV2.pvtv2 import pvt_v2_b1, pvt_v2_b3, pvt_v2_b2
# Helper: load safetensors into a model (best-effort, non-strict)
try:
    from safetensors.torch import load_file as _safetensors_load
except Exception:
    _safetensors_load = None

def _load_safetensors_into_model(model: nn.Module, path: str, device: str | None = None):
    """Try to load a safetensors file into model. This is best-effort and uses strict=False.

    It will attempt to load with and without common prefixes like 'model.' or 'backbone.'.
    If `safetensors` is not installed or file missing, this is a no-op.
    """
    if _safetensors_load is None:
        # safetensors not installed
        return False
    if not os.path.isfile(path):
        return False

    # load on cpu first
    try:
        state = _safetensors_load(path, device="cpu")
    except Exception:
        return False

    def try_load(state_dict):
        # move tensors to model device if requested
        if device is not None:
            state_dict = {k: v.to(device) if hasattr(v, 'to') else v for k, v in state_dict.items()}
        try:
            model.load_state_dict(state_dict, strict=False)
            return True
        except Exception:
            return False

    # try direct
    if try_load(state):
        return True

    # try common prefix strips
    prefixes = ["model.", "backbone.", "module."]
    for p in prefixes:
        new = {k[len(p):] if k.startswith(p) else k: v for k, v in state.items()}
        if try_load(new):
            return True

    # give up but log first few keys to help debugging
    try:
        sample_keys = list(state.keys())[:10]
        print(f"safetensors load: could not auto-match keys; sample keys: {sample_keys}")
    except Exception:
        pass
    return False

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1)
        )

    def forward(self, x, x_dion):
        token_dim = x[:, :1, :]
        x = rearrange(x[:, 1:, :], "B (H W) C -> B C H W", H=24)
        weighting_dion = F.adaptive_avg_pool2d(x_dion, 1)
        weighting_dion = self.fc1(weighting_dion)
        y = x * weighting_dion
        y = torch.cat([token_dim, rearrange(y, "B C H W-> B (H W) C ", H=24)], dim=1)
        return y
    

@META_ARCH_REGISTRY.register()
class CATSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        # self.image_encoder = vit_base()
        # self.image_encoder1 = vit_base()
        # self.image_encoder2 = vit_base()
        # self.image_encoder3 = vit_base()
        # Load safetensors from the package directory (not the current working dir).

        

        safetensors_path = os.path.join(os.path.dirname(__file__), "dinov3rs")
        # # weights = load_file(safetensors_path)
        # self.image_encoder = pixio_vitb16(pretrained=safetensors_path)
        self.image_encoder = AutoModel.from_pretrained(
            safetensors_path,
            ignore_mismatched_sizes=True
        )
        print(self.image_encoder.parameters)
        peft_config = LoraConfig(
            r=4, 
            lora_alpha=32, 
            target_modules=["k_proj", "q_proj", "v_proj"],  # 这里的名称需匹配 DINO 内部层名
            lora_dropout=0.05,
            bias="none"
        )
        self.image_encoder = get_peft_model(self.image_encoder, peft_config)
        
        # for p in self.image_encoder.parameters():
        #     p.requires_grad=True
        # self.image_encoder1 = AutoModel.from_pretrained(
        #     safetensors_path,
        #     ignore_mismatched_sizes=True 
        #     # device_map="cuda:0", 
        # )
        # self.image_encoder2 = AutoModel.from_pretrained(
        #     safetensors_path,
        #     ignore_mismatched_sizes=True 
        #     # device_map="cuda:0", 
        # )
        # self.image_encoder3 = AutoModel.from_pretrained(
        #     safetensors_path,
        #     ignore_mismatched_sizes=True 
        #     # device_map="cuda:0", 
        # )
        # loaded_ok = False
        # try:
        #     loaded_ok = _load_safetensors_into_model(self.image_encoder, safetensors_path, device=None)
        #     if loaded_ok:
        #         print(f"Loaded image encoder weights from {safetensors_path}")
        # except Exception:
        #     loaded_ok = False
        # if not loaded_ok:
        #     # fallback: don't crash if weights missing; model stays randomly initialized
        #     # You can set PYTHONPATH or place model.safetensors at repo root if you prefer the old behavior.
        #     pass
        # Try to load local safetensors weights if available
        # safetensors_path = os.path.join(os.path.dirname(__file__), "model.safetensors")
        # try:
        #     # pass device as 'cpu' for initial load; tensors will be moved on demand
        #     loaded = _load_safetensors_into_model(self.image_encoder, safetensors_path, device=None)
        #     if loaded:
        #         print(f"Loaded image encoder weights from {safetensors_path}")
        # except Exception:
        #     # best-effort: don't crash if loading fails
        #     pass
        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        # self.annother_backbone = pvt_v2_b1()
        # file_path = os.path.join(os.path.dirname(__file__), "PVTV2/pvt_v2_b1.pth")
        # checkpoint = torch.load(file_path)
        # model_state_dict = self.annother_backbone.state_dict()
        # pretrained_weights = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        # model_state_dict.update(pretrained_weights)
        # self.annother_backbone.load_state_dict(model_state_dict)
        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        # self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        # self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        self.upsample3 = nn.ConvTranspose2d(512, self.proj_dim, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(512, self.proj_dim, kernel_size=1, stride=1)
        self.upsample5 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.upsample6 = nn.ConvTranspose2d(320, 256, kernel_size=2, stride=2)

        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))
        
        # self.linear = nn.Linear(1024, 512)
        # self.linear１ = nn.Linear(512, 512)
        # self.linear２ = nn.Linear(512, 512)
        # self.linear３ = nn.Linear(512, 512)
        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.linear１ = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.linear２ = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.linear３ = nn.Sequential(
            nn.Linear(1024, 512),
        )
        # self.cov = nn.Conv2d(768, self.proj_dim, kernel_size=1, stride=1, bias=False)
        self.shared_token = nn.Parameter(torch.zeros(1, 1, self.proj_dim))
        self.linear4 = nn.Sequential(
            nn.Linear(1024, 768),
        )
        self.linear5 = nn.Sequential(
            nn.Linear(1024, 768),
        )
        self.linear6 = nn.ConvTranspose2d(self.proj_dim*2, self.proj_dim, kernel_size=1, stride=1)
        self.linear7 = nn.ConvTranspose2d(self.proj_dim*2, self.proj_dim, kernel_size=1, stride=1)
        self.linear8 = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.linearcat1 = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.linearcat2 = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.linearcat3 = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.linearcat4 = nn.Sequential(
            nn.Linear(1024, 512),
        )
        # shapes = [(2, 577, 1024), (2, 577, 1024), (2, 577, 1024), (2, 577, 1024)]
        # aparams = []
        # for s in shapes:
        #     t = torch.empty(s)
        #     nn.init.normal_(t, 0, 0.02)
        #     aparams.append(nn.Parameter(t))
        # bparams = []
        # for s in shapes:
        #     t = torch.empty(s)
        #     nn.init.normal_(t, 0, 0.02)
        #     bparams.append(nn.Parameter(t))
        # self.alphaparams = nn.ParameterList(aparams)
        # self.betaparams = nn.ParameterList(bparams)
        self.weights = SqueezeAndExcitation(channel=512, reduction=16)
    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
        }
    def get_hf_intermediate_layers(self, model, pixel_values, layer_indexes):
        """
        针对 Hugging Face DINOv3 模型的中间层提取补丁
        参数:
            model: 加载好的 DINOv3 模型对象
            pixel_values: 输入图像 Tensor [B, 3, H, W]
            layer_indexes: 想要提取的层索引列表, 例如 [3, 5, 8, 11]
        返回:
            List[Tensor]: 每个 Tensor 形状为 [B, C, h, w]
        """
        # 1. 前向传播并要求输出隐藏层
        outputs = model(pixel_values, output_hidden_states=True)
        import math
        # Hugging Face 的 hidden_states 包含 (embed_outputs + all_layer_outputs)
        # 所以索引 i 实际上对应的是第 i 层的输出
        all_layers = outputs.hidden_states 
        print("all_layers length:", len(all_layers))
        selected_outs = []
        for idx in layer_indexes:
            # 获取对应层的特征: [Batch, Tokens, Dim]
            feat = all_layers[idx] 
            # 2. 剥离 [CLS] token (假设 [CLS] 在第一个位置)
            # DINOv3 通常有 1 个 CLS token，剩下的就是空间 Patch
            spatial_feat = feat[:, 5:, :] 
            # 3. 计算特征图的长宽 (Reshape)
            B, N, C = spatial_feat.shape
            h = w = int(math.sqrt(N))
            # 转换维度: [B, N, C] -> [B, C, N] -> [B, C, h, w]
            print("spatial_feat shape before reshape:", spatial_feat.shape)
            spatial_feat = spatial_feat.transpose(1, 2).reshape(B, C, h, w)
            selected_outs.append(spatial_feat)
        return selected_outs

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        self.layers = []

        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
        clip_images_resized_90 = torch.rot90(clip_images_resized, k=1, dims=(2, 3))
        clip_images_resized_180  = torch.rot90(clip_images_resized, k=2, dims=(2, 3))
        clip_images_resized_270 = torch.rot90(clip_images_resized, k=3, dims=(2, 3))

        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        clip_features1 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized_90, dense=True)
        clip_features2 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized_180, dense=True)
        clip_features3 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized_270, dense=True)

        # hidden_states = self.image_encoder(clip_images_resized, output_hidden_states=True).hidden_states
        # clip_features_dino = self.linear(hidden_states[23][:, 4:, :])
        clip_features_dino = self.linear(self.image_encoder(clip_images_resized).last_hidden_state[:, 4:, :])
        # clip_features_dino = self.linear(self.image_encoder(clip_images_resized, output_hidden_states=True).hidden_states[11][:, 4:, :])
        # clip_features_dino_90 = self.linear1(self.image_encoder(clip_images_resized_90).last_hidden_state[:, 4:, :])
        # clip_features_dino_180 = self.linear2(self.image_encoder(clip_images_resized_180).last_hidden_state[:, 4:, :])
        # clip_features_dino_270 = self.linear3(self.image_encoder(clip_images_resized_270).last_hidden_state[:, 4:, :])
        print("clip_features_dino shape:", clip_features_dino.shape)
        print("clip_features shape before adding dino:", clip_features.shape)
        # clip_features += clip_features_dino
        # clip_features =  self.linearcat1(torch.cat([clip_features, clip_features_dino], dim=-1))
        # print(self.alphaparams[0])
        # print(self.betaparams[0])
        # clip_features = clip_features + clip_features_dino
        token_dim = clip_features_dino[:, :1, :]
        clip_features_dino = clip_features_dino[:, 1:, :]
        clip_features_dino_tmp = clip_features_dino
        clip_features_dino = rearrange(clip_features_dino, "B (H W) C -> B C H W", H=24)
        # clip_features_dino_90 = torch.rot90(clip_features_dino, k=1, dims=(2, 3))
        # clip_features_dino_180 = torch.rot90(clip_features_dino, k=2, dims=(2, 3))
        # clip_features_dino_270 = torch.rot90(clip_features_dino, k=3, dims=(2, 3))
        # clip_features_dino_90 = torch.cat((rearrange(clip_features_dino_90, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features_dino_180 = torch.cat((rearrange(clip_features_dino_180, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features_dino_270 = torch.cat((rearrange(clip_features_dino_270, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features = self.weights(clip_features, clip_features_dino)
        # clip_features = clip_features + self.linear1(clip_features_dino)
        # clip_features1 = clip_features1 + self.linear1(clip_features_dino_90)
        # clip_features2 = clip_features2 + self.linear2(clip_features_dino_180)
        # clip_features3 = clip_features3 + self.linear3(clip_features_dino_270)
        
        # clip_features1 = self.linearcat2(torch.cat([clip_features1, clip_features_dino_90], dim=-1))
        # clip_features2 = self.linearcat3(torch.cat([clip_features2, clip_features_dino_180], dim=-1))
        # clip_features3 = self.linearcat4(torch.cat([clip_features3, clip_features_dino_270], dim=-1))

        # clip_features1 = self.image_encoder(clip_images_resized_90).last_hidden_state[:, 4:, :]
        # clip_features2 = self.image_encoder(clip_images_resized_180).last_hidden_state[:, 4:, :]
        # clip_features3 = self.image_encoder(clip_images_resized_270).last_hidden_state[:, 4:, :]
        # clip_features = self.image_encoder(clip_images_resized)[11]["patch_tokens_norm"]
        # clip_features1 = self.image_encoder(clip_images_resized_90)[11]["patch_tokens_norm"]
        # clip_features2 = self.image_encoder(clip_images_resized_180)[11]["patch_tokens_norm"]
        # clip_features3 = self.image_encoder(clip_images_resized_270)[11]["patch_tokens_norm"]
        # B = clip_features1.shape[0]
        # shared = self.shared_token.expand(B, -1, -1) 
        # clip_features = torch.cat([clip_features, shared], dim=1)
        # clip_features1 = torch.cat([clip_features1, shared], dim=1)
        # clip_features2 = torch.cat([clip_features2, shared], dim=1)
        # clip_features3 = torch.cat([clip_features3, shared], dim=1)
        print("clip_features shape:", clip_features.shape)
        image_features = self.linear8(torch.cat([clip_features[:, 1:, :], clip_features_dino_tmp], dim=-1))
        # image_features = clip_features[:, 1:, :] + clip_features_dino_tmp
        print("image_features shape:", image_features.shape)
        # CLIP ViT features for guidance
        
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        # res4_hidden = self.linear4(hidden_states[self.layer_indexes[0] ][:, 5:, :])
        # res5_hidden = self.linear5(hidden_states[self.layer_indexes[1] ][:, 5:, :])
        res4_hidden = self.linear4(self.image_encoder(clip_images_resized, output_hidden_states=True).hidden_states[self.layer_indexes[0] ][:, 5:, :])
        res5_hidden = self.linear5(self.image_encoder(clip_images_resized, output_hidden_states=True).hidden_states[self.layer_indexes[1] ][:, 5:, :])
        res4_hidden = rearrange(res4_hidden, "B (H W) C -> B C H W", H=24)
        res5_hidden = rearrange(res5_hidden, "B (H W) C -> B C H W", H=24)
        print("res4 shape:", res4_hidden.shape)
        # res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24) + res4_hidden
        # res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24) + res5_hidden
        res4 = self.linear6(torch.cat([rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24), res4_hidden], dim=1))
        res5 = self.linear7(torch.cat([rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24), res5_hidden], dim=1))

        # outs = self.annother_backbone(clip_images_resized)
        # res5, res4 = self.upsample5(outs[1]), self.upsample6(outs[2])
        # res3 = self.upsample3(outs[3])
        # res7, res6 = self.upsample6(outs[0]), self.upsample5(outs[1])
        # res4 = self.upsample1(res4)
        # res5 = self.upsample2(res5)
        # 假设 self.image_encoder 已是 Dinov3 模型实例，self.layer_indexes = [3, 7]
        # 在 forward 中，得到 intermediate layers（patch tokens 已 reshape 为 B,C,H,W）
        # outs = self.image_encoder.get_intermediate_layers(clip_images_resized, n=self.layer_indexes, reshape=True)
        # --- 替换旧的 self.image_encoder.get_intermediate_layers ---
        # outs = self.get_hf_intermediate_layers(
        #     self.image_encoder, 
        #     clip_images_resized, 
        #     self.layer_indexes
        # )
        # outs = self.image_encoder(clip_images_resized)[self.layer_indexes[0]]["patch_tokens_norm"], self.image_encoder(clip_images_resized)[self.layer_indexes[1]]["patch_tokens_norm"]
        # outs 是 tuple，顺序对应传入的 layer_indexes
        # res4 = rearrange(outs[0], "B (H W) C -> B C H W", H=24) # (B, C, H_patch, W_patch)
        # print("res4 shape before upsample:", res4.shape)
        # res5 = rearrange(outs[1], "B (H W) C -> B C H W", H=24) # (B, C, H_patch, W_patch)
        # 接下来与原代码一致：
        print("res5 shape before upsample:", res5.shape)
        print("res4 shape before upsample:", res4.shape)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        print("res4 after shape:", res4.shape)
        print("res5 after shape:", res5.shape)
        features = {'res5': res5, 'res4': res4, 'res3': res3}

        outputs = self.sem_seg_head([clip_features[:, 0:, :], clip_features1, clip_features2, clip_features3], features)
        # 保存原始 raw outputs（未经过 sigmoid / postprocess），以便外部工具使用（例如 CAM 分析）
        # try:
        #     self._last_raw_outputs = outputs
        # except Exception:
        #     self._last_raw_outputs = None
        # # 保存输入图像的一份拷贝（CPU, HWC, uint8）用于可视化叠加
        # try:
        #     img_tensor = batched_inputs[0]["image"].cpu()
        #     img_np = img_tensor.permute(1, 2, 0).numpy()
        #     if img_np.dtype != 'uint8':
        #         img_np = np.clip(img_np, 0, 255).astype('uint8')
        #     self._last_input_image = img_np
        # except Exception:
        #     self._last_input_image = None
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results


    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        print("ENTER CATSeg.forward, sliding_window=", self.sliding_window)
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        
        # 旋转90度
        clip_images_90 = torch.rot90(clip_images, k=1, dims=(2, 3))
        # 旋转180度
        clip_images_180  = torch.rot90(clip_images, k=2, dims=(2, 3))
        # 旋转270度
        clip_images_270 = torch.rot90(clip_images, k=3, dims=(2, 3))
        
        self.layers = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        clip_features1 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_90, dense=True)
        clip_features2 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_180, dense=True)
        clip_features3 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_270, dense=True)

        clip_features_dino = self.linear(self.image_encoder(clip_images).last_hidden_state[:, 4:, :])
        # clip_features_dino = self.linear(self.image_encoder(clip_images_resized, output_hidden_states=True).hidden_states[11][:, 4:, :])
        # clip_features_dino_90 = self.linear1(self.image_encoder(clip_images_resized_90).last_hidden_state[:, 4:, :])
        # clip_features_dino_180 = self.linear2(self.image_encoder(clip_images_resized_180).last_hidden_state[:, 4:, :])
        # clip_features_dino_270 = self.linear3(self.image_encoder(clip_images_resized_270).last_hidden_state[:, 4:, :])
        print("clip_features_dino shape:", clip_features_dino.shape)
        print("clip_features shape before adding dino:", clip_features.shape)
        # clip_features += clip_features_dino
        # clip_features =  self.linearcat1(torch.cat([clip_features, clip_features_dino], dim=-1))
        # print(self.alphaparams[0])
        # print(self.betaparams[0])
        # clip_features = clip_features + clip_features_dino
        token_dim = clip_features_dino[:, :1, :]
        clip_features_dino = clip_features_dino[:, 1:, :]
        clip_features_dino_tmp = clip_features_dino
        clip_features_dino = rearrange(clip_features_dino, "B (H W) C -> B C H W", H=24)
        # clip_features_dino_90 = torch.rot90(clip_features_dino, k=1, dims=(2, 3))
        # clip_features_dino_180 = torch.rot90(clip_features_dino, k=2, dims=(2, 3))
        # clip_features_dino_270 = torch.rot90(clip_features_dino, k=3, dims=(2, 3))
        # clip_features_dino_90 = torch.cat((rearrange(clip_features_dino_90, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features_dino_180 = torch.cat((rearrange(clip_features_dino_180, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features_dino_270 = torch.cat((rearrange(clip_features_dino_270, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features = self.weights(clip_features, clip_features_dino)
        # clip_features = clip_features + self.linear1(clip_features_dino)
        # clip_features1 = clip_features1 + self.linear1(clip_features_dino_90)
        # clip_features2 = clip_features2 + self.linear2(clip_features_dino_180)
        # clip_features3 = clip_features3 + self.linear3(clip_features_dino_270)
        
        # clip_features1 = self.linearcat2(torch.cat([clip_features1, clip_features_dino_90], dim=-1))
        # clip_features2 = self.linearcat3(torch.cat([clip_features2, clip_features_dino_180], dim=-1))
        # clip_features3 = self.linearcat4(torch.cat([clip_features3, clip_features_dino_270], dim=-1))

        # clip_features1 = self.image_encoder(clip_images_resized_90).last_hidden_state[:, 4:, :]
        # clip_features2 = self.image_encoder(clip_images_resized_180).last_hidden_state[:, 4:, :]
        # clip_features3 = self.image_encoder(clip_images_resized_270).last_hidden_state[:, 4:, :]
        # clip_features = self.image_encoder(clip_images_resized)[11]["patch_tokens_norm"]
        # clip_features1 = self.image_encoder(clip_images_resized_90)[11]["patch_tokens_norm"]
        # clip_features2 = self.image_encoder(clip_images_resized_180)[11]["patch_tokens_norm"]
        # clip_features3 = self.image_encoder(clip_images_resized_270)[11]["patch_tokens_norm"]
        # B = clip_features1.shape[0]
        # shared = self.shared_token.expand(B, -1, -1) 
        # clip_features = torch.cat([clip_features, shared], dim=1)
        # clip_features1 = torch.cat([clip_features1, shared], dim=1)
        # clip_features2 = torch.cat([clip_features2, shared], dim=1)
        # clip_features3 = torch.cat([clip_features3, shared], dim=1)
        print("clip_features shape:", clip_features.shape)
        image_features = self.linear8(torch.cat([clip_features[:, 1:, :], clip_features_dino_tmp], dim=-1))
        # image_features = clip_features[:, 1:, :] + clip_features_dino_tmp
        print("image_features shape:", image_features.shape)
        # CLIP ViT features for guidance
        
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        # res4_hidden = self.linear4(hidden_states[self.layer_indexes[0] ][:, 5:, :])
        # res5_hidden = self.linear5(hidden_states[self.layer_indexes[1] ][:, 5:, :])
        res4_hidden = self.linear4(self.image_encoder(clip_images, output_hidden_states=True).hidden_states[self.layer_indexes[0] ][:, 5:, :])
        res5_hidden = self.linear5(self.image_encoder(clip_images, output_hidden_states=True).hidden_states[self.layer_indexes[1] ][:, 5:, :])
        res4_hidden = rearrange(res4_hidden, "B (H W) C -> B C H W", H=24)
        res5_hidden = rearrange(res5_hidden, "B (H W) C -> B C H W", H=24)
        print("res4 shape:", res4_hidden.shape)
        # res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24) + res4_hidden
        # res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24) + res5_hidden
        res4 = self.linear6(torch.cat([rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24), res4_hidden], dim=1))
        res5 = self.linear7(torch.cat([rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24), res5_hidden], dim=1))

        # outs = self.annother_backbone(clip_images_resized)
        # res5, res4 = self.upsample5(outs[1]), self.upsample6(outs[2])
        # res3 = self.upsample3(outs[3])
        # res7, res6 = self.upsample6(outs[0]), self.upsample5(outs[1])
        # res4 = self.upsample1(res4)
        # res5 = self.upsample2(res5)
        # 假设 self.image_encoder 已是 Dinov3 模型实例，self.layer_indexes = [3, 7]
        # 在 forward 中，得到 intermediate layers（patch tokens 已 reshape 为 B,C,H,W）
        # outs = self.image_encoder.get_intermediate_layers(clip_images_resized, n=self.layer_indexes, reshape=True)
        # --- 替换旧的 self.image_encoder.get_intermediate_layers ---
        # outs = self.get_hf_intermediate_layers(
        #     self.image_encoder, 
        #     clip_images_resized, 
        #     self.layer_indexes
        # )
        # outs = self.image_encoder(clip_images_resized)[self.layer_indexes[0]]["patch_tokens_norm"], self.image_encoder(clip_images_resized)[self.layer_indexes[1]]["patch_tokens_norm"]
        # outs 是 tuple，顺序对应传入的 layer_indexes
        # res4 = rearrange(outs[0], "B (H W) C -> B C H W", H=24) # (B, C, H_patch, W_patch)
        # print("res4 shape before upsample:", res4.shape)
        # res5 = rearrange(outs[1], "B (H W) C -> B C H W", H=24) # (B, C, H_patch, W_patch)
        # 接下来与原代码一致：
        print("res5 shape before upsample:", res5.shape)
        print("res4 shape before upsample:", res4.shape)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        print("res4 after shape:", res4.shape)
        print("res5 after shape:", res5.shape)
        features = {'res5': res5, 'res4': res4, 'res3': res3}
        # res3 = rearrange(clip_features, "B (H W) C -> B C H W", H=24)
        # res4 = self.upsample1(rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24))
        # res5 = self.upsample2(rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24))

        # features = {'res5': res5, 'res4': res4, 'res3': res3,}


        outputs = self.sem_seg_head([clip_features, clip_features1, clip_features2,clip_features3], features)
        

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
