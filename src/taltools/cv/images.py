import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image


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


def rotate_3d_landmarks(kp, pitch=0, roll=0, yaw=0):
    """
    Rotates a set of 3D landmarks using pitch, roll, and yaw angles.
    :param kp: numpy array of shape (J, 3), where J is the number of landmarks.
    :param pitch: Rotation around the X-axis in degrees.
    :param roll: Rotation around the Y-axis in degrees.
    :param yaw: Rotation around the Z-axis in degrees.
    :return: Rotated numpy array of shape (J, 3).
    """
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])

    Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                   [0, 1, 0],
                   [-np.sin(roll), 0, np.cos(roll)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return (kp @ R.T)


def plot_3d_landmarks(kp, pairs=None, joint_indices=False, elev=-120, azim=90, roll=180):
    """
    Plots a set of 3D landmarks with object rotation and returns the image as a numpy array.
    :param kp: numpy array of shape (J, 3), where J is the number of landmarks.
    :param elev: Elevation angle for the view.
    :param azim: Azimuth angle for the view.
    :param roll: Roll angle for the view.
    :return: numpy array of the image.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of landmarks
    ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], c='b', marker='o')

    if joint_indices:
        for i, (x, y, z) in enumerate(kp):
            ax.text(x, y, z, str(i), fontsize=8, color='black', ha='center', va='center')

    if pairs:
        for (i, j) in pairs:
            x_values = [kp[i, 0], kp[j, 0]]
            y_values = [kp[i, 1], kp[j, 1]]
            z_values = [kp[i, 2], kp[j, 2]]
            ax.plot(x_values, y_values, z_values, c='r', linewidth=2)  # Red edges

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio
    max_range = np.ptp(kp, axis=0).max() / 2.0
    mid_x = np.mean(kp[:, 0])
    mid_y = np.mean(kp[:, 1])
    mid_z = np.mean(kp[:, 2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set view angle
    ax.view_init(elev=elev, azim=azim, roll=roll)

    plt.title("3D Landmarks")

    # Convert plot to numpy image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = np.array(Image.open(buf))

    return image