import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


COLORS = [{'name': k.split(':')[1], 'value': tuple(int(v.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))} for k, v in mcolors.TABLEAU_COLORS.items()]

def blur_area(frame, center, radius, kernel_size=(50, 50)):
    c_mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.circle(c_mask, center, radius, 1, thickness=-1)
    mask = cv2.bitwise_and(frame, frame, mask=c_mask)
    img_mask = frame - mask
    blur = cv2.blur(frame, kernel_size)
    mask2 = cv2.bitwise_and(blur, blur, mask=c_mask)  # mask
    final_img = img_mask + mask2
    return final_img

def fig2np(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    raw_data = np.array(fig.canvas.buffer_rgba())
    frame = raw_data[:, :, :3]
    # frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
