from pathlib import Path
import numpy as np
import imageio.v2 as imageio


def _to_np(x):
    import torch
    if isinstance(x, np.ndarray): return x
    if isinstance(x, list): return np.asarray(x)
    if isinstance(x, tuple): return np.asarray(x)
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)

def _to_T_H_W_C(x):
    x = _to_np(x)
    if x.ndim == 5:             # (B,T,C,H,W) or (B,T,H,W,C) → take first item
        x = x[0]
    if x.ndim == 4:
        T, A, B, C = x.shape
        if A in (1,3):          # (T,C,H,W) → (T,H,W,C)
            x = np.transpose(x, (0,2,3,1))
    elif x.ndim == 3:           # (T,H,W) → (T,H,W,1)
        x = x[..., None]
    else:
        raise ValueError(f"Unexpected shape {x.shape}")
    return x

def _to_uint8(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mn >= -1.01 and mx <= 1.01: x = (x + 1.0) / 2.0
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)

def _rgb(x):
    return np.repeat(x, 3, axis=-1) if x.shape[-1] == 1 else x

def save_gif(frames, path, fps=10):
    v = _to_uint8(_to_T_H_W_C(frames))
    frames_list = [f[...,0] if f.shape[-1]==1 else f for f in v]
    imageio.mimsave(str(path), frames_list, duration=1.0/fps, loop=0)

def save_mp4(frames, path, fps=10, quality=7):
    v = _rgb(_to_uint8(_to_T_H_W_C(frames)))
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=None, quality=quality)
    try:
        for f in v: writer.append_data(f)
    except Exception:
        writer.close()
        writer = imageio.get_writer(str(path), fps=fps, codec="mpeg4", macro_block_size=None, quality=quality)
        for f in v: writer.append_data(f)
    finally:
        writer.close()

def hstack(*videos):
    vs = [_rgb(_to_uint8(_to_T_H_W_C(v))) for v in videos]
    T = min(v.shape[0] for v in vs)
    vs = [v[:T] for v in vs]
    return np.stack([np.hstack([v[t] for v in vs]) for t in range(T)], axis=0)

def save_rollout(out_dir, *, context, pred, truth=None, fps=10):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    save_gif(context, out/"context.gif", fps=fps);  save_mp4(context, out/"context.mp4", fps=fps)
    save_gif(pred,    out/"pred.gif",    fps=fps);  save_mp4(pred,    out/"pred.mp4",    fps=fps)
    comp = hstack(context, pred, truth) if truth is not None else hstack(context, pred)
    save_gif(comp, out/"composite.gif", fps=fps);  save_mp4(comp, out/"composite.mp4", fps=fps)
