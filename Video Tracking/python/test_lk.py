import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from LucasKanade import LucasKanade
from file_utils import mkdir_if_missing

# NOTE: Either change data_name through command line or directly in code
import sys
data_name = sys.argv[1] if len(sys.argv) > 1 else 'landing'
# data_name = 'landing'      # could choose from (car1, car2, landing)

do_display = (not int(sys.argv[2])) if len(sys.argv) > 2 else 1

# load data name
data = np.load('../data/%s.npy' % data_name)

# obtain the initial rect with format (x1, y1, x2, y2)
if data_name == 'car1':
    initial = np.array([170, 130, 290, 250])
    fps = 50  # Set frame rate
elif data_name == 'car2':
    initial = np.array([59,116,145,151])
    fps = 50  # Set frame rate
elif data_name == 'landing':
    initial = np.array([440, 80, 560, 140])
    fps = 10  # Set frame rate
else:
    assert False, 'the data name must be one of (car1, car2, landing)'

numFrames = data.shape[2]
frame_height, frame_width = data.shape[:2]
w = initial[2] - initial[0]
h = initial[3] - initial[1]

# Desired output video dimensions
desired_width = 1080
desired_height = 720
video_aspect_ratio = desired_width / desired_height

# Initialize VideoWriter
output_video_path = f"../results/lk/{data_name}/output_video.mp4"
mkdir_if_missing(output_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = 10  # Set frame rate
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (desired_width, desired_height))

# Function to resize and pad the frame
def resize_and_pad(img, target_width, target_height):
    img_height, img_width = img.shape[:2]
    img_aspect_ratio = img_width / img_height

    if img_aspect_ratio > video_aspect_ratio:
        # Wider image: fit width and pad height
        scale = target_width / img_width
        new_width = target_width
        new_height = int(img_height * scale)
        resized_img = cv2.resize(img, (new_width, new_height))
        padding = (target_height - new_height) // 2
        padded_img = cv2.copyMakeBorder(resized_img, padding, target_height - new_height - padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        # Taller image: fit height and pad width
        scale = target_height / img_height
        new_height = target_height
        new_width = int(img_width * scale)
        resized_img = cv2.resize(img, (new_width, new_height))
        padding = (target_width - new_width) // 2
        padded_img = cv2.copyMakeBorder(resized_img, 0, 0, padding, target_width - new_width - padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return padded_img

# Loop over frames
rects = []
rects.append(initial)
fig = plt.figure(1) #1
ax = fig.add_subplot(111)

for i in range(numFrames-1):
    print("frame****************", i)
    It = data[:,:,i]
    It1 = data[:,:,i+1]
    rect = rects[i]

    # Run algorithm
    dx, dy = LucasKanade(It, It1, rect)
    print("dx, dy: ", (dx,dy))

    # Transform the old rect to new one
    newRect = np.array([rect[0] + dx, rect[1] + dy, rect[0] + dx + w, rect[1] + dy + h])
    rects.append(newRect)

    # Show image
    print("Plotting: ", rect)
    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
    plt.imshow(It1, cmap='gray')
    save_path = "../results/lk/%s/frame%06d.jpg" % (data_name, i+1)
    mkdir_if_missing(save_path)
    plt.savefig(save_path)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    # img = cv2.resize(img, (frame_width, frame_height))
    img = resize_and_pad(img, desired_width, desired_height)
    video_writer.write(img)

    if do_display:
        plt.pause(0.00001)
    ax.clear()

# Release the video writer
video_writer.release()
print(f"Video saved to {output_video_path}")