# utils/viz.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image
import imageio.v2 as imageio

def _to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _to_T_H_W_C(x):
    x = _to_np(x)
    if x.ndim == 5:  # (B,T,C,H,W) or (B,T,H,W,C)
        x = x[0]
    if x.ndim == 4:
        T, A, B, C = x.shape
        if A in (1, 3):        # (T,C,H,W) -> (T,H,W,C)
            x = x.transpose(0, 2, 3, 1)
    elif x.ndim == 3:          # (T,H,W) -> (T,H,W,1)
        x = x[..., None]
    else:
        raise ValueError(f"Unexpected shape {x.shape}")
    return x

def _to_uint8(v):
    v = v.astype(np.float32)
    vmin, vmax = float(v.min()), float(v.max())
    if vmin >= -1.01 and vmax <= 1.01:
        v = (v + 1.0) / 2.0
    v = np.clip(v, 0.0, 1.0)
    return (v * 255.0).round().astype(np.uint8)

def _rgb(v):
    return np.repeat(v, 3, axis=-1) if v.shape[-1] == 1 else v

def save_gif(frames, path, fps=10):
    v = _to_uint8(_to_T_H_W_C(frames))
    frames_list = [f[..., 0] if f.shape[-1] == 1 else f for f in v]
    imageio.mimsave(str(path), frames_list, duration=1.0/fps, loop=0)

def save_mp4(frames, path, fps=10, quality=7):
    import tempfile, shutil, subprocess, os
    v = _rgb(_to_uint8(_to_T_H_W_C(frames)))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer using system ffmpeg for widest compatibility
    try:
        tmpdir = tempfile.mkdtemp(prefix="viz_frames_")
        try:
            # write frames as PNGs
            for i, f in enumerate(v):
                fname = os.path.join(tmpdir, f"frame_{i:06d}.png")
                imageio.imwrite(fname, f)

            # build ffmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", str(max(18, 30 - int(quality))),
                "-movflags", "+faststart",
                str(out_path)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            shutil.rmtree(tmpdir)
    except FileNotFoundError:
        # ffmpeg not installed; fall back to imageio which may use imageio-ffmpeg plugin
        try:
            with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=None, quality=quality) as w:
                for f in v:
                    w.append_data(f)
        except Exception:
            # fallback to generic mpeg4 codec
            with imageio.get_writer(str(path), fps=fps, codec="mpeg4", macro_block_size=None, quality=quality) as w:
                for f in v:
                    w.append_data(f)

def hstack(*videos):
    vs = [_rgb(_to_uint8(_to_T_H_W_C(v))) for v in videos]
    T = min(v.shape[0] for v in vs)
    vs = [v[:T] for v in vs]
    return np.stack([np.hstack([v[t] for v in vs]) for t in range(T)], axis=0)

def save_rollout(out_dir, *, context, pred, truth=None, compute_metrics=False):
    """Save prediction rollout as individual PNG frames."""
    import numpy as np
    from PIL import Image
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    def save_sequence(frames, subdir):
        save_dir = out / subdir
        save_dir.mkdir(exist_ok=True)
        frames = _to_uint8(_to_T_H_W_C(frames))  # Convert to uint8 numpy array
        for t, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(save_dir / f"frame_{t:03d}.png")
    
    # Save each sequence
    save_sequence(context, "context")
    save_sequence(pred, "predicted")
    if truth is not None:
        save_sequence(truth, "ground_truth")
    
    # Create and save composite frames
    if truth is not None:
        comp = hstack(context, pred, truth)
    else:
        comp = hstack(context, pred)
    comp_frames = _to_uint8(_to_T_H_W_C(comp))
    for t, frame in enumerate(comp_frames):
        img = Image.fromarray(frame)
        img.save(out / f"composite_frame_{t:03d}.png")

    if compute_metrics and truth is not None:
        from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
        def _prep_gray(x):
            v = _to_uint8(_to_T_H_W_C(x))
            return v[..., 0] if v.shape[-1] > 1 else v[..., 0]
        P, G = _prep_gray(pred), _prep_gray(truth)
        T = min(P.shape[0], G.shape[0])
        metrics = {"ssim": [], "psnr": []}
        for t in range(T):
            metrics["ssim"].append(float(ssim(G[t], P[t], data_range=255)))
            metrics["psnr"].append(float(psnr(G[t], P[t], data_range=255)))
        (out/"metrics.json").write_text(json.dumps(metrics, indent=2))
