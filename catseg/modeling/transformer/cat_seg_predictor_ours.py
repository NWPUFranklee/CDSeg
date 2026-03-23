# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py
import fvcore.nn.weight_init as weight_init
import torch

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .model_ours import Aggregator
from cat_seg.third_party import clip
from cat_seg.third_party import imagenet_templates
from peft import LoraConfig, get_peft_model, TaskType

import numpy as np
import open_clip
class CATSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
    ):
        """
        Args:
            
        """
        super().__init__()
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
  
        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=device, 
                force_image_size=336,)
        
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            # clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False)
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
            # clip_model1, clip_preprocess1 = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
            # clip_model2, clip_preprocess2 = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
            # clip_model3, clip_preprocess3 = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
        # config = LoraConfig(
        #     r=8,
        #     lora_alpha=32,
        #     target_modules=["visual.transformer.resblocks.11.attn.q_proj_weight"], # 举例：只针对视觉最后一层
        # )
        # clip_model.visual = get_peft_model(clip_model.visual, config)
        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates

        self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        
        self.clip_model = clip_model.float()
        # self.clip_model.visual.set_params('reduced', 'naclip', 5)

        # self.clip_model1 = clip_model1.float()
        # self.clip_model2 = clip_model2.float()
        # self.clip_model3 = clip_model3.float()
        self.clip_preprocess = clip_preprocess
        self.linear = nn.Linear(512, 1024)
        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            )
        self.transformer = transformer
        
        self.tokens = None
        self.cache = None

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        return ret

    def forward(self, x, vis_guidance, prompt=None, gt_cls=None):
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        text = self.class_texts if self.training else self.test_class_texts
        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)

        text = text.repeat(x[0].shape[0], 1, 1, 1)
        print("text shape:", text.shape)
        print("x[0] shape:", x[0].shape)
        print("x[1] shape:", x[1].shape)
        print("x[2] shape:", x[2].shape)
        print("x[3] shape:", x[3].shape)
        print("vis shape:", vis[0].shape)
        print("vis shape:", vis[1].shape)
        print("vis shape:", vis[2].shape)

        out = self.transformer(x, text, vis)
        return out

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else: 
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        if self.cache is not None and not self.training:
            return self.cache
        
        if self.tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(texts).cuda()
                else: 
                    texts = clip.tokenize(texts).cuda()
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        # class_embeddings = clip_model.encode_text(tokens)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        
        class_embeddings = class_embeddings.unsqueeze(1)
        
        if not self.training:
            self.cache = class_embeddings
            
        return class_embeddings