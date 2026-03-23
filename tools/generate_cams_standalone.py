#!/usr/bin/env python3
"""
Standalone Grad-CAM generator for this repository.

Design goals:
- Do not modify any existing code in the repo.
- Use the repo's Trainer.build_model(cfg) to construct the exact model used by training.
- Register forward hooks to capture activations and the semantic head's raw logits,
  then backprop from the raw logits to obtain gradients for Grad-CAM.
- Minimal, best-effort preprocessing that matches the training pipeline (ResizeShortestEdge + pad).

Usage example:
  python3 tools/generate_cams_standalone.py \
    --config configs/vitb_384.yaml \
    --weights output/model_0059999.pth \
    --images /path/a.jpg,/path/b.jpg \
    --modules sem_seg_head.predictor.clip_model.visual.transformer,sem_seg_head \
    --classes 0,5 \
    --out output/cams

This script is intentionally conservative: it tries to locate modules by dotted path
and registers hooks on the first modules it finds. It does not change any source files.
"""

import argparse
import os
import sys
import math
import random
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog
from detectron2.data import transforms as T

# ensure repo root is importable
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_net import Trainer
from cat_seg import add_cat_seg_config


def find_module(root: torch.nn.Module, path: str):
    """Resolve dotted path like 'sem_seg_head.predictor.clip_model.visual'.
    Supports simple index like blocks[3]. Returns None if not found.
    """
    cur = root
    for part in path.split('.'):
        if '[' in part and part.endswith(']'):
            name, idx = part[:-1].split('[')
            if not hasattr(cur, name):
                return None
            cur = getattr(cur, name)
            cur = cur[int(idx)]
        else:
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
    return cur


def preprocess_image(img_path: str, cfg) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = img_rgb.shape[:2]

    # ResizeShortestEdge using cfg (best-effort)
    min_size = getattr(cfg.INPUT, 'MIN_SIZE_TEST', None)
    max_size = getattr(cfg.INPUT, 'MAX_SIZE_TEST', None)
    if min_size is None:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
    if isinstance(min_size, (list, tuple)):
        min_size = min_size[0]
    if max_size is None:
        max_size = cfg.INPUT.MAX_SIZE_TRAIN

    resize = T.ResizeShortestEdge(min_size, max_size)
    aug_input = T.AugInput(img_rgb)
    aug_input, transforms = T.apply_transform_gens([resize], aug_input)
    img_trans = aug_input.image

    # convert to CHW tensor, keep values 0..255
    img_tensor = torch.as_tensor(np.ascontiguousarray(img_trans.transpose(2, 0, 1)))

    # pad to size_divisibility if set
    size_div = int(getattr(cfg.INPUT, 'SIZE_DIVISIBILITY', 0))
    if size_div > 0:
        h, w = img_tensor.shape[-2:]
        pad_w = (size_div - w) if (w % size_div) != 0 else 0
        pad_h = (size_div - h) if (h % size_div) != 0 else 0
        if pad_h or pad_w:
            padding = [0, pad_w, 0, pad_h]
            img_tensor = F.pad(img_tensor, padding, value=128).contiguous()

    return img_tensor, (ori_h, ori_w), img_rgb


def make_overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    cam_uint8 = (np.clip(cam, 0, 1) * 255).astype('uint8')
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap.astype('float32') * alpha + img_rgb.astype('float32') * (1 - alpha)).astype('uint8')
    return overlay


def compute_gradcam_from_activation(act: torch.Tensor, grad: torch.Tensor, target_class: int = None) -> np.ndarray:
    # act: [B,C,H,W] expected; grad same shape
    if act.dim() == 4:
        act_t = act[0]
    elif act.dim() == 3:
        # could be [B, N, C] or [C,H,W]
        if act.shape[0] == 1:
            act_t = act.squeeze(0)
        else:
            # ambiguous: treat as [C,H,W]
            act_t = act
    else:
        act_t = act

    if grad.dim() == 4:
        grad_t = grad[0]
    elif grad.dim() == 3:
        grad_t = grad
    else:
        grad_t = grad

    # Special case: activations that include a class/token dimension, e.g. [B, C, T, H, W]
    if act.dim() == 5:
        # Expect [B, C, T, H, W] or [B, T, H, W, C] variants; try to find T axis
        # Common format in this repo: B C T H W -> select by target_class
        if target_class is None:
            # can't select; collapse over T
            act_sel = act[0].mean(dim=1) if act.shape[1] != 1 else act[0, 0]
            grad_sel = grad[0].mean(dim=1) if grad is not None and grad.dim() == 5 else (grad[0] if grad is not None else None)
        else:
            # select channel maps for this class
            try:
                act_sel = act[0, :, target_class, :, :]
                grad_sel = grad[0, :, target_class, :, :] if (grad is not None and grad.dim() == 5) else (grad[0] if grad is not None else None)
            except Exception:
                # fallback: average over token dim
                act_sel = act[0].mean(dim=2)
                grad_sel = grad[0].mean(dim=2) if (grad is not None and grad.dim() == 5) else (grad[0] if grad is not None else None)
        act_t = act_sel
        grad_t = grad_sel

    # Now expect act_t shape [C,H,W]
    if act_t.dim() == 1:
        # token vector; fallback to zeros
        return np.zeros((1,1), dtype='float32')

    if act_t.dim() == 3:
        weights = grad_t.view(grad_t.shape[0], -1).mean(dim=1)
        cam = (weights.view(-1, 1, 1) * act_t).sum(dim=0).cpu().detach().numpy()
        cam = np.maximum(cam, 0.0)
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
        else:
            cam = np.zeros_like(cam)
        return cam

    # fallback
    return np.zeros((1,1), dtype='float32')


def try_reshape_token_activation(act: torch.Tensor):
    # If activation is [B, N, C] or [N, C], try to reshape N into sqrt(N)x sqrt(N)
    t = act
    if t.dim() == 3:
        B, N, C = t.shape
        if B == 1:
            n = N
            s = int(math.sqrt(n))
            if s * s == n:
                # assume no class token
                resh = t[0].transpose(0, 1).reshape(C, s, s)
                return resh
    if t.dim() == 2:
        N, C = t.shape
        s = int(math.sqrt(N))
        if s * s == N:
            resh = t.transpose(0, 1).reshape(C, s, s)
            return resh
    return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--images', default=None, help='comma-separated image files')
    parser.add_argument('--samples', type=int, default=0, help='random samples from cfg.DATASETS.TEST')
    parser.add_argument('--modules', default=None, help='comma-separated module dotted paths to hook')
    parser.add_argument('--classes', default='0', help='comma-separated class ids')
    parser.add_argument('--out', default='output/cams', help='output directory')
    parser.add_argument('--verbose', action='store_true', default=True, help='print debug info')
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.freeze()

    model = Trainer.build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.resume_or_load(args.weights, resume=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # choose images
    image_list: List[str] = []
    if args.images:
        image_list = [p for p in args.images.split(',') if p]
    if args.samples > 0:
        ds = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) > 0 else None
        if ds is None:
            raise RuntimeError('cfg.DATASETS.TEST empty; provide --images or set cfg')
        dataset = DatasetCatalog.get(ds)
        picks = random.sample(dataset, min(args.samples, len(dataset)))
        image_list += [p['file_name'] for p in picks]
    if not image_list:
        raise RuntimeError('No images provided; use --images or --samples')

    # prepare modules to hook
    if args.modules:
        cand = [m for m in args.modules.split(',') if m]
    else:
        cand = [
            'sem_seg_head.predictor.clip_model.visual.transformer',
            'sem_seg_head.predictor.clip_model.visual',
            'sem_seg_head.predictor.transformer',
            'sem_seg_head',
        ]

    target_modules = []
    for path in cand:
        mod = find_module(model, path)
        if mod is not None:
            target_modules.append((path, mod))

    # If transformer aggregator exists, also add internal submodules that are likely
    # responsible for producing logits or useful visual features. This helps when
    # top-level hooks don't receive gradients.
    try:
        agg = find_module(model, 'sem_seg_head.predictor.transformer')
        if agg is not None:
            extra = ['conv1', 'corr_embed', 'conv_decoder', 'decoder1', 'decoder2', 'head', 'layers']
            for name in extra:
                if hasattr(agg, name):
                    sub = getattr(agg, name)
                    # only append nn.Module instances (skip methods / functions)
                    if isinstance(sub, torch.nn.Module):
                        target_modules.append((f'sem_seg_head.predictor.transformer.{name}', sub))
            # also try to hook individual layers inside layers ModuleList
            if hasattr(agg, 'layers'):
                for i, layer in enumerate(agg.layers):
                    if isinstance(layer, torch.nn.Module):
                        target_modules.append((f'sem_seg_head.predictor.transformer.layers[{i}]', layer))
                # Monkey-patch instance methods to capture intermediate 'corr' tensors at runtime
                try:
                    # wrap correlation to save the raw corr tensor on the instance for later use
                    if hasattr(agg, 'correlation'):
                        orig_corr = agg.correlation
                        def corr_wrapper(*args, **kwargs):
                            corr = orig_corr(*args, **kwargs)
                            try:
                                agg._last_corr = corr.detach().cpu()
                            except Exception:
                                agg._last_corr = corr
                            return corr
                        agg.correlation = corr_wrapper

                    # wrap corr_embed to save its input (pre-conv) as well
                    if hasattr(agg, 'corr_embed'):
                        orig_corr_embed = agg.corr_embed
                        def corr_embed_wrapper(x, *a, **kw):
                            try:
                                agg._last_corr_input = x.detach().cpu()
                            except Exception:
                                agg._last_corr_input = x
                            return orig_corr_embed(x, *a, **kw)
                        agg.corr_embed = corr_embed_wrapper
                except Exception:
                    # non-fatal: if monkeypatch fails, continue without saved corr
                    pass
    except Exception:
        pass

    if not target_modules:
        print('No target modules found from candidates:', cand)
        raise RuntimeError('No hookable module found; pass --modules')

    if args.verbose:
        print('Hooking modules:')
        for name, mod in target_modules:
            # Some entries can be callables or other non-Module objects (e.g., functions).
            # Guard the call to named_children() so we don't crash on unexpected types.
            try:
                children = [n for n, _ in mod.named_children()]
            except Exception:
                if callable(mod):
                    children = ['<callable>']
                else:
                    # fallback: show a short dir listing (trimmed)
                    try:
                        children = list(dir(mod))[:10]
                    except Exception:
                        children = []
            print(' -', name, '->', type(mod), 'children:', children)

    classes = [int(x) for x in args.classes.split(',') if x != '']

    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}
    semseg_raw = {'out': None}

    def make_act_hook(name):
        def hook(module, input, output):
            out = output
            if isinstance(out, (list, tuple)):
                out = out[0]
            activations[name] = out

            def save_grad(g):
                gradients[name] = g

            try:
                out.register_hook(save_grad)
            except Exception:
                pass

        return hook

    def semseg_hook(module, input, output):
        out = output
        if isinstance(out, (list, tuple)):
            out = out[0]
        semseg_raw['out'] = out

    hooks = []
    # hook activations (only on Module instances)
    for name, mod in target_modules:
        if hasattr(mod, 'register_forward_hook'):
            try:
                hooks.append(mod.register_forward_hook(make_act_hook(name)))
            except Exception as e:
                print(f'Warning: failed to register forward hook on {name}: {e}')
        else:
            print(f'Warning: skipping hook for {name} because it is not an nn.Module')
    # hook sem_seg_head to capture raw logits
    if hasattr(model, 'sem_seg_head'):
        hooks.append(model.sem_seg_head.register_forward_hook(semseg_hook))
    else:
        print('Warning: model has no sem_seg_head attribute; cannot capture logits directly')

    for img_path in image_list:
        print('Processing', img_path)
        img_tensor, (ori_h, ori_w), img_rgb = preprocess_image(img_path, cfg)
        batched = [{ 'image': img_tensor, 'file_name': img_path, 'height': ori_h, 'width': ori_w }]

        with torch.enable_grad():
            model.zero_grad()
            for x in batched:
                x['image'] = x['image'].to(device=device, dtype=torch.float32)
                # ensure input requires grad so autograd builds graph for backward
                x['image'].requires_grad_(True)

            outputs = model(batched)

            raw = semseg_raw['out']
            if raw is None:
                # try to extract from outputs if it's raw (rare)
                raise RuntimeError('Could not capture sem_seg raw logits; ensure sem_seg_head forward runs and is hooked')

            raw = raw.to(device)
            if args.verbose:
                try:
                    r = raw.detach().cpu().numpy()
                    print(f'raw logits shape: {raw.shape}, min {r.min():.6f}, max {r.max():.6f}, mean {r.mean():.6f}')
                except Exception:
                    print('raw logits shape:', raw.shape)

            for cls in classes:
                # primary backward target: mean over spatial dims
                score = raw[0, cls].mean()
                if args.verbose:
                    print(f'Computing CAM for class {cls}: score shape {raw[0, cls].shape}, scalar {score.item():.6f}')
                    # print activations shapes
                    for aname, at in activations.items():
                        try:
                            print(f' activation {aname} shape {tuple(at.shape)} requires_grad={getattr(at, "requires_grad", None)}')
                        except Exception:
                            print(f' activation {aname} (could not read shape)')
                model.zero_grad()
                score.backward(retain_graph=True)

                # diagnostic: check gradients; if all zeros, try sum as fallback
                any_grad_nonzero = False
                for _n, _ in target_modules:
                    g = gradients.get(_n, None)
                    if g is not None:
                        try:
                            mag = g.abs().max().item()
                            if args.verbose:
                                print(f'grad max for module {_n}:', mag)
                        except Exception:
                            mag = 0.0
                        if mag > 1e-7:
                            any_grad_nonzero = True
                if not any_grad_nonzero:
                    if args.verbose:
                        print('No non-zero gradients after mean-backward; trying sum-backward fallback')
                    model.zero_grad()
                    score2 = raw[0, cls].sum()
                    score2.backward(retain_graph=True)
                    # re-check
                    any_grad_nonzero = False
                    for _n, _ in target_modules:
                        g = gradients.get(_n, None)
                        if g is not None:
                            try:
                                mag = g.abs().max().item()
                            except Exception:
                                mag = 0.0
                            if mag > 1e-7:
                                any_grad_nonzero = True

                # If still no gradients, try per-pixel backward at predicted max location
                if not any_grad_nonzero:
                    if args.verbose:
                        print('Still no gradients after sum-backward; trying per-pixel backward at argmax')
                    try:
                        probs = torch.sigmoid(raw)
                        pm = probs[0, cls]
                        # find spatial argmax
                        if pm.dim() == 2:
                            idx = torch.argmax(pm)
                            y = int(idx // pm.shape[1])
                            x = int(idx % pm.shape[1])
                            model.zero_grad()
                            raw[0, cls, y, x].backward(retain_graph=True)
                            if args.verbose:
                                print(f'Per-pixel backward at ({y},{x})')
                        else:
                            if args.verbose:
                                print('raw logits spatial dims not found for per-pixel fallback')
                    except Exception as e:
                        if args.verbose:
                            print('Per-pixel fallback failed:', e)

                # If still no gradients, compute corr-based CAM (similarity between dense visual features and text embedding)
                if not any_grad_nonzero:
                    if args.verbose:
                        print('No non-zero gradients found for any hooked module — computing corr-based heatmap fallback using CLIP visual/text features')
                    try:
                        predictor = None
                        if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor'):
                            predictor = model.sem_seg_head.predictor
                        if predictor is None or not hasattr(predictor, 'clip_model'):
                            if args.verbose:
                                print('No predictor.clip_model found; cannot compute corr-based fallback')
                        else:
                            # prepare PIL image and preprocessing using predictor.clip_preprocess if available
                            from PIL import Image
                            pil = Image.fromarray(img_rgb)
                            try:
                                clip_pre = getattr(predictor, 'clip_preprocess', None)
                                if clip_pre is None:
                                    # try predictor.clip_model preprocess attribute
                                    clip_pre = getattr(predictor.clip_model, 'preprocess', None)
                                if clip_pre is None:
                                    # fallback: convert to tensor and normalize roughly (may be imperfect)
                                    clip_img_t = torch.as_tensor(np.array(pil).transpose(2,0,1)).float().unsqueeze(0).to(device)
                                else:
                                    # clip_pre may be a callable transform
                                    clip_img_t = clip_pre(pil)
                                    if isinstance(clip_img_t, tuple):
                                        clip_img_t = clip_img_t[0]
                                    clip_img_t = clip_img_t.unsqueeze(0).to(device)
                            except Exception as e:
                                if args.verbose:
                                    print('Error running clip_preprocess:', e)
                                clip_img_t = torch.as_tensor(np.array(pil).transpose(2,0,1)).float().unsqueeze(0).to(device)

                            # First, if we monkey-patched Aggregator to save corr, prefer that exact tensor
                            try:
                                agg = find_module(model, 'sem_seg_head.predictor.transformer')
                            except Exception:
                                agg = None

                            if agg is not None and hasattr(agg, '_last_corr'):
                                try:
                                    corr_cpu = agg._last_corr
                                    if args.verbose:
                                        print('Using saved aggregator._last_corr for corr-fallback, shape:', tuple(corr_cpu.shape))
                                    # corr_cpu expected shape: B x .. x H x W (class/prompt dims in the middle)
                                    num_classes = None
                                    try:
                                        num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
                                    except Exception:
                                        pass
                                    c = corr_cpu.numpy()
                                    # find axis corresponding to classes (match dimension == num_classes)
                                    class_axis = None
                                    if num_classes is not None:
                                        for ax in range(1, c.ndim - 2):
                                            if c.shape[ax] == num_classes:
                                                class_axis = ax
                                                break
                                    # If we found class axis, index it; else try plausible defaults
                                    if class_axis is not None:
                                        # select class map
                                        selector = [0] * c.ndim
                                        selector[0] = 0
                                        selector[class_axis] = cls
                                        # build slice
                                        sl = [slice(None)] * c.ndim
                                        sl[0] = 0
                                        sl[class_axis] = cls
                                        sel = c[tuple(sl)]
                                        # now sel shape should be something like (other_dims..., H, W)
                                        # sum over all non-spatial dims
                                        spatial = sel
                                        if spatial.ndim > 2:
                                            # collapse leading dims
                                            spatial = spatial.reshape(-1, spatial.shape[-2], spatial.shape[-1]).sum(axis=0)
                                        cam_map = spatial
                                    else:
                                        # fallback: try common layout B, (N*P), T, H, W -> T is class axis at -3
                                        if c.ndim >= 5 and c.shape[-3] <= 10000:
                                            # assume class axis = -3
                                            cam = c[0, :, cls, :, :] if c.shape[-3] > 1 else c[0, :, 0, :, :]
                                            # collapse first dim
                                            if cam.ndim == 3:
                                                cam_map = cam.sum(axis=0)
                                            else:
                                                cam_map = cam
                                        else:
                                            raise RuntimeError('Could not infer class axis in saved corr tensor')

                                    # normalize and save
                                    cam_map = cam_map - cam_map.min()
                                    if cam_map.max() > 0:
                                        cam_map = cam_map / (cam_map.max() + 1e-8)
                                    cam_resized = cv2.resize(cam_map.astype('float32'), (img_rgb.shape[1], img_rgb.shape[0]))
                                    overlay = make_overlay(img_rgb, cam_resized)
                                    base = os.path.splitext(os.path.basename(img_path))[0]
                                    out_path = os.path.join(args.out, f'{base}_cls{cls}_corr_saved.png')
                                    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                                    print('Saved corr-saved fallback CAM to', out_path)
                                    # done
                                    continue
                                except Exception as e:
                                    if args.verbose:
                                        print('Using saved corr failed:', e)

                            # get dense visual features
                            try:
                                with torch.no_grad():
                                    clip_feats = predictor.clip_model.encode_image(clip_img_t, dense=True)
                            except TypeError:
                                # some clip implementations have signature without dense; try without
                                with torch.no_grad():
                                    clip_feats = predictor.clip_model.encode_image(clip_img_t)

                            # clip_feats expected shape [B, 1+N, C] where first token may be global
                            cf = clip_feats.detach().cpu()
                            if cf.dim() == 3 and cf.shape[1] > 1:
                                tokens = cf[0, 1:, :]
                                N = tokens.shape[0]
                                s = int(math.sqrt(N))
                                if s * s == N:
                                    # reshape to HxW
                                    vis = tokens.reshape(s, s, -1)  # H W C
                                    vis = vis.reshape(-1, vis.shape[-1])  # (H*W) C
                                else:
                                    # fallback: average tokens to single spatial map
                                    vis = tokens.mean(dim=0, keepdim=True)
                                    vis = vis.reshape(-1, vis.shape[-1])
                            else:
                                if args.verbose:
                                    print('Unexpected clip_feats shape for dense features:', cf.shape)
                                vis = cf.reshape(-1, cf.shape[-1])

                            # get text embeddings for classes
                            try:
                                # prefer get_text_embeds if available
                                text_emb = None
                                if hasattr(predictor, 'get_text_embeds'):
                                    text_emb = predictor.get_text_embeds(predictor.test_class_texts if hasattr(predictor, 'test_class_texts') else predictor.class_texts, predictor.prompt_templates, predictor.clip_model)
                                elif hasattr(predictor, 'text_features_test'):
                                    text_emb = predictor.text_features_test
                                elif hasattr(predictor, 'text_features'):
                                    text_emb = predictor.text_features
                                else:
                                    text_emb = None

                                if text_emb is None:
                                    raise RuntimeError('No text embeddings available')

                                # bring to cpu and shape to (num_classes, C)
                                te = text_emb.detach().cpu()
                                # te might be (P, num_classes, C) or (num_classes, P, C) or (num_classes,1,C)
                                if te.dim() == 3 and te.shape[1] != 1:
                                    # try to reduce prompt dimension
                                    if te.shape[0] == 1:
                                        te = te.squeeze(0)
                                # Now ensure shape (num_classes, C)
                                if te.dim() == 3:
                                    # (P, num_classes, C) -> mean over prompts
                                    te = te.mean(dim=0)
                                if te.dim() == 2:
                                    text_mat = te  # num_classes x C
                                else:
                                    text_mat = te.reshape(te.shape[0], -1)

                                # normalize
                                text_norm = text_mat / (text_mat.norm(dim=-1, keepdim=True) + 1e-8)
                                vis_t = torch.tensor(vis)
                                vis_norm = vis_t / (vis_t.norm(dim=-1, keepdim=True) + 1e-8)

                                # similarity: (H*W, C) x (num_classes, C).T -> (H*W, num_classes)
                                sim = (vis_norm @ text_norm.t()).numpy()
                                cls_map = sim[:, cls] if sim.ndim == 2 else sim[:, 0]
                                # reshape back
                                if 's' in locals() and s * s == N:
                                    cam_map = cls_map.reshape(s, s)
                                else:
                                    cam_map = cls_map.reshape(1, -1)

                                cam_map = cam_map - cam_map.min()
                                if cam_map.max() > 0:
                                    cam_map = cam_map / (cam_map.max() + 1e-8)

                                # resize and save
                                cam_resized = cv2.resize(cam_map.astype('float32'), (img_rgb.shape[1], img_rgb.shape[0]))
                                overlay = make_overlay(img_rgb, cam_resized)
                                base = os.path.splitext(os.path.basename(img_path))[0]
                                out_path = os.path.join(args.out, f'{base}_cls{cls}_corr_fallback.png')
                                cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                                print('Saved corr-fallback CAM to', out_path)
                            except Exception as e:
                                if args.verbose:
                                    print('corr-fallback failed:', e)
                    except Exception as e:
                        if args.verbose:
                            print('corr-fallback high-level error:', e)

                # Save input-gradient saliency as diagnostic
                try:
                    img_var = batched[0]['image']
                    if hasattr(img_var, 'grad') and img_var.grad is not None:
                        g = img_var.grad.detach().cpu()
                        # sum over channels
                        sal = g.abs().sum(dim=0).numpy()
                        sal = sal - sal.min()
                        if sal.max() > 0:
                            sal = sal / (sal.max() + 1e-8)
                        sal_img = (sal * 255).astype('uint8')
                        sal_color = cv2.applyColorMap(sal_img, cv2.COLORMAP_JET)
                        sal_color = cv2.cvtColor(sal_color, cv2.COLOR_BGR2RGB)
                        base = os.path.splitext(os.path.basename(img_path))[0]
                        out_sal = os.path.join(args.out, f'{base}_cls{cls}_input_grad.png')
                        cv2.imwrite(out_sal, cv2.cvtColor(sal_color, cv2.COLOR_RGB2BGR))
                        if args.verbose:
                            print('Saved input-gradient saliency to', out_sal)
                except Exception:
                    pass

                for name, _ in target_modules:
                    act = activations.get(name, None)
                    grad = gradients.get(name, None)
                    if act is None:
                        print(f'No activation for {name}; skipping')
                        continue
                    if grad is None:
                        print(f'No gradient captured for {name}; skipping')
                        continue

                    # If activation is token-like, try to reshape to C,H,W
                    act_to_use = act.detach()
                    grad_to_use = grad
                    # attempt token reshape
                    if act_to_use.dim() in (2, 3) and not (act_to_use.dim() == 4):
                        maybe = try_reshape_token_activation(act_to_use)
                        if maybe is not None:
                            act_to_use = maybe.unsqueeze(0) if maybe.dim()==3 and maybe.shape[0]!=1 else maybe
                            # gradients may not be reshaped; try to mirror
                            gmaybe = try_reshape_token_activation(grad_to_use)
                            if gmaybe is not None:
                                grad_to_use = gmaybe.unsqueeze(0) if gmaybe.dim()==3 and gmaybe.shape[0]!=1 else gmaybe

                    cam = compute_gradcam_from_activation(act_to_use, grad_to_use, target_class=cls)
                    # if cam is 2D HxW, resize to original image
                    if cam.ndim == 2:
                        cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
                    else:
                        cam_resized = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype='float32')

                    overlay = make_overlay(img_rgb, cam_resized)
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    safe_name = name.replace('.', '_')
                    out_path = os.path.join(args.out, f'{base}_cls{cls}_{safe_name}.png')
                    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    print('Saved', out_path)

    # remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass


if __name__ == '__main__':
    main()
