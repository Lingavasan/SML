import os
import numpy as np
import imageio

OUT = 'outputs_test2'
os.makedirs(OUT, exist_ok=True)

# create fake GT and recon frames (10 frames, 64x64 RGB)
frames_gt = [(np.random.rand(64,64,3)*255).astype(np.uint8) for _ in range(10)]
frames_recon = [(np.random.rand(64,64,3)*255).astype(np.uint8) for _ in range(10)]

# side-by-side concat
out_frames = [np.concatenate([frames_gt[i], frames_recon[i]], axis=1) for i in range(10)]

# write gif
gif_path = os.path.join(OUT, 'test.gif')
imageio.mimsave(gif_path, out_frames, fps=5, loop=0)
print('Wrote', gif_path)

# write mp4
mp4_path = os.path.join(OUT, 'test.mp4')
writer = imageio.get_writer(mp4_path, fps=5, codec='libx264')
for f in out_frames:
    writer.append_data(f)
writer.close()
print('Wrote', mp4_path)

# save frames
frames_dir = os.path.join(OUT, 'frames')
os.makedirs(frames_dir, exist_ok=True)
for i, f in enumerate(out_frames):
    imageio.imwrite(os.path.join(frames_dir, f'frame_{i:04d}.png'), f)
print('Wrote frames to', frames_dir)
