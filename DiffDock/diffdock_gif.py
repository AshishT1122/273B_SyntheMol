from pymol import cmd
from PIL import Image
import os

# Load your protein and compounds here
cmd.load('GLP1.pdb')
# cmd.load('compound1.sdf')  # Load compounds as needed

# Enable rock mode
cmd.rock()

# Set the background to transparent
cmd.bg_color('white')
cmd.set('ray_opaque_background', 0)

# Increase resolution
cmd.viewport(800, 800)  # Adjust as needed for higher quality

# Directory to save frames
frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

frames = []

# Capture frames
for frame in range(120):  # increased number of frames for smoother motion
    cmd.turn('y', 3)  # reduced rotation angle for smoother motion
    frame_path = os.path.join(frames_dir, f'frame_{frame:03d}.png')
    cmd.png(frame_path)
    frames.append(Image.open(frame_path))

# Convert frames to GIF
gif_path = 'output.gif'
frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0, transparency=255, disposal=2)

# Cleanup
for frame in frames:
    os.remove(frame.filename)

print(f"GIF saved as {gif_path}")
