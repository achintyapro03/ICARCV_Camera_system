import time
import serial
import numpy as np
import cv2
import threading
import mediapipe as mp
import os
from IPython.display import clear_output
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from cv2 import aruco
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import networkx as nx

ser = serial.Serial('COM4', 115200)

camera_width = 1280.0
camera_height = 960.0
fx = 1430.0
fy = 1430.0
cx = 480.0
cy = 620.0

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

points_left = []
points_right = []
caliberated = [False, False]
theta_offsets = [0, 5.1, 76.44]
coordinates = [[0, 0, 0] for i in range(4)]

def calc_dist(a, b, x_world, t):
    global d
    # a_rad, b_rad = (180 - a) * np.pi/180, b * np.pi/180

    x1 = d * (0.5 - (np.sin(a) * np.cos(b)) / np.sin(a + b))
    y1 = d * ((np.sin(a) * np.sin(b)) / np.sin(a + b))

    val = t[0] + t[1]
    X1 = val[0]
    Y1 = val[1]
    Z1 = val[2]

    X2 = x_world[0]
    Y2 = x_world[1]
    Z2 = x_world[2]

    depth3 = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2)
    val = t[0][1] - 0.055

    depth2 = np.sqrt(x1*x1 + y1*y1)
    try:
        new_depth = np.sqrt(depth2 * depth2 - val*val)
    except:
        new_depth = 0
    return x1, y1, np.sqrt(x1*x1 + y1*y1), depth3, new_depth

def rotation_matrix(yaw, pitch, roll):

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    # Combined rotation matrix
    R = np.dot(Rz, np.dot(Ry, Rx))

    return R

def translation_vector(x_pos, roll):
    global d, camera_stalk_height, platform_height, foot_to_camera_base
    t = np.array([x_pos * d/2, platform_height + foot_to_camera_base + camera_stalk_height * np.cos(roll),  camera_stalk_height * np.sin(roll)])

    return t

d = 0.1625
platform_height = 0
foot_to_camera_base = 0.16
camera_stalk_height = 0.04

thetas = np.array([0, 0, 0], dtype=float)
radians = np.array([0, 0, 0], dtype=float)

frame_shape_x = 0
frame_shape_y = 0

R = [rotation_matrix(0, radians[1], radians[0]), rotation_matrix(0, radians[2], radians[0])]
t = [translation_vector(-1, radians[0]), translation_vector(1, radians[0])]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-55, azim=75) 

def plot():
    global R, t, coordinates
    ax.clear()
    
    R1 = rotation_matrix(0, radians[1], radians[0])
    R2 = rotation_matrix(0, radians[2], radians[0])

    t1 = translation_vector(1, radians[0])
    t2 = translation_vector(-1, radians[0])
    
    vectors = [(R1, t1), (R2, t2)]
    colors = ['b', 'g', 'r', 'c', 'y', 'k', 'orange', 'orange', 'purple', 'brown']
    count = 0
    for i in coordinates:
        # if(count == 0):
        #     ax.scatter([i[0]], [i[1]], [i[2]], color=colors[count], s=50)  # Plotting coordinates
        # else:
        #     ax.scatter([i[0]], [i[1]], [i[2]], color=colors[count], s=10)  # Plotting coordinates
        ax.scatter([i[0]], [i[1]], [i[2]], color=colors[count], s=10)  # Plotting coordinates
        count += 1

    edges = [['0', '1'], ['0', '2'], ['1', '3'], ['3', '5'], ['2', '4'], ['4', '6']]

    ax.scatter(0, 0, 4, color='green', marker='^', label='Camera')
    for R, t in vectors:
        # Plot local XYZ axes
        axes = 0.2 * np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
        local_axes = np.dot(R, axes.T).T

        ax.quiver(*t, *local_axes[0], color='r', linestyle='dashed', arrow_length_ratio=0.1)
        ax.quiver(*t, *local_axes[1], color='g', linestyle='dashed', arrow_length_ratio=0.1)
        ax.quiver(*t, *local_axes[2], color='b', linestyle='dashed', arrow_length_ratio=0.1)

    # Plot edges between coordinates
    # for edge in edges:
    #     start, end = int(edge[0]), int(edge[1])
    #     ax.plot([coordinates[start][0], coordinates[end][0]],
    #             [coordinates[start][1], coordinates[end][1]],
    #             [coordinates[start][2], coordinates[end][2]], color='black')

    # Set plot labels and limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rotation Vectors and Local Axes')

    ax.plot([-5, 5], [0, 0], [0, 0], color='darkgrey', linewidth=1)  # X-axis
    ax.plot([0, 0], [-5, 5], [0, 0], color='darkgrey', linewidth=1)  # Y-axis
    ax.plot([0, 0], [0, 0], [-7, 7], color='darkgrey', linewidth=1)  # Z-axis

    ax.grid(False)  # Hide the grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.line.set_color((0.5, 0.5, 0.5, 0.1))
    ax.yaxis.line.set_color((0.5, 0.5, 0.5, 0.1))
    ax.zaxis.line.set_color((0.5, 0.5, 0.5, 0.1))
    ax.grid(True)

    ax.xaxis._axinfo['grid'].update(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.yaxis._axinfo['grid'].update(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.zaxis._axinfo['grid'].update(color='lightgrey', linestyle='-', linewidth=0.3)



def update(frame):
    plot()

class CameraThread(threading.Thread):
    def __init__(self, cam_id, cam_name, points_list, serial_thread, caliberated, starting_mode):
        super().__init__()
        self.cam_id = cam_id
        self.cam_name = cam_name
        self.points_list = points_list
        self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        self.running = True
        self.serial_thread = serial_thread
        self.calibrated = caliberated
        self.starting_mode = starting_mode

    def find_aruco(self):
        global frame_shape_x, frame_shape_y
        self.serial_thread.set_mode(1)
        while (self.calibrated[self.cam_id - 1] == False):
            ret, frame = self.cap.read()
            if(frame_shape_x == 0):
                frame_shape_x = frame.shape[1]
                frame_shape_y = frame.shape[0]
            if not ret:
                print("Failed to grab frame ")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters = aruco.DetectorParameters_create()
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
            flag = False
            if ids is not None:
                self.points_list.clear()
                for i in range(len(ids)):

                    if(ids[i] == 3):
                        flag = True
                        c = corners[i][0]
                        center_x = int(c[:, 0].mean())
                        center_y = int(c[:, 1].mean())
                        cv2.putText(frame_markers, "id={0}".format(ids[i][0]), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        x_diff = int(center_x - frame.shape[1] / 2.0)
                        # y_diff = int(center_y - frame.shape[0] / 2.0)
                        if abs(x_diff) < 5:
                            self.calibrated[self.cam_id - 1] = True 
                        else:
                            self.calibrated[self.cam_id - 1] = False
                        self.points_list.append([x_diff, 0])
                        break
                if(not flag):
                    self.points_list.clear()
                    self.points_list.append([6969, 6969])
            else:
                self.points_list.clear()
                self.points_list.append([6969, 6969])
                
            # Display the frame with markers
            cv2.imshow(f'Frame {self.cam_id}', frame_markers)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                self.running = False
                break
        cv2.destroyWindow(f'Frame {self.cam_id}')
        # self.calib_event.wait()

    def track_person(self):
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1, smooth_landmarks=True) as pose:
            # print("hi")

            while self.cap.isOpened() and self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    self.points_list.clear()
                    landmarks = results.pose_landmarks.landmark
                    # print(landmarks)
                    for landmark in landmarks:
                        # self.points_list.append([int((landmark.x - 0.5) * frame.shape[1]), int((landmark.y - 0.2) * frame.shape[0])])
                        self.points_list.append([int((landmark.x - 0.5) * frame.shape[1]), int((landmark.y - 0.2) * frame.shape[0])])


                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )

                cv2.imshow(f'Mediapipe Feed - {self.cam_name}', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    self.running = False
                    break
            cv2.destroyWindow(f'Mediapipe Feed - {self.cam_name}')

    def track_multi_aruco(self, tracking_id):
        self.serial_thread.set_mode(2)
        while (True):
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame ")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters = aruco.DetectorParameters_create()
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

            if ids is not None:
                self.points_list.clear()
                for i in range(len(ids)):
                    c = corners[i][0]
                    center_x = int(c[:, 0].mean())
                    center_y = int(c[:, 1].mean())
                    if(ids[i] == tracking_id):
                        cv2.putText(frame_markers, "id={0}".format(ids[i][0]), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)
                        x_diff = int(center_x - frame.shape[1] / 2.0)
                        y_diff = int(center_y - frame.shape[0] / 2.0)
                        self.points_list.append([x_diff, y_diff])
                    else:
                        cv2.putText(frame_markers, "id={0}".format(ids[i][0]), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            else:
                self.points_list.clear()
                self.points_list.append([6969, 6969])
                
            # Display the frame with markers
            cv2.imshow(f'multiaruco {self.cam_id}', frame_markers)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                self.running = False
                break
        cv2.destroyWindow(f'multiaruco {self.cam_id}')
            
    def run(self):
        if(self.starting_mode == 1):
            self.find_aruco()
        if(self.starting_mode == 2):
            # self.find_aruco()
            # self.track_person()
            self.find_aruco()
            tracking_id = int(input("Enter id of aruco to be tracked : "))
            self.serial_thread.set_mode(2)
            self.track_multi_aruco(tracking_id)

        elif(self.starting_mode == 3):
            self.find_aruco()
            tracking_id = int(input("Enter id of aruco to be tracked : "))
            self.serial_thread.set_mode(2)
            self.track_multi_aruco(tracking_id)
        self.cap.release()
        


def triangulate_point_direct(u_l, v_l, u_r, v_r, K, R, t):
    P_l = np.linalg.inv(K) @ np.array([u_l, v_l, 1])
    P_r = np.linalg.inv(K) @ np.array([u_r, v_r, 1])

    P_l_world = R[0].T @ P_l
    P_r_world = R[1].T @ P_r

    A = np.array([
        [P_l_world[0], -P_r_world[0]],
        [P_l_world[1], -P_r_world[1]],
        [P_l_world[2], -P_r_world[2]]
    ])

    B = t[1] - t[0]

    X = np.linalg.lstsq(A, B, rcond=None)[0]

    X_l_world = P_l_world * X[0] + t[0]
    X_r_world = P_r_world * X[1] + t[1]

    X_world = (X_l_world + X_r_world) / 2

    return X_world


def triangulate_point_dlt(u_l, v_l, u_r, v_r, K, R, t):
    R1 = np.array(R[0]).reshape(3, 3)
    R2 = np.array(R[1]).reshape(3, 3)
    t1 = np.array(t[0]).reshape(3, 1)
    t2 = np.array(t[1]).reshape(3, 1)

    RT1 = np.hstack([R1, t1])
    RT2 = np.hstack([R2, t2])

    P1 = K @ RT1 
    P2 = K @ RT2 

    A = np.array([
        (u_l * P1[2, :] - P1[0, :]),
        (v_l * P1[2, :] - P1[1, :]),
        (u_r * P2[2, :] - P2[0, :]),
        (v_r * P2[2, :] - P2[1, :])
    ])

    A = A.reshape((4, 4))
    B = A.T @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    point_3d_homogeneous = Vh[-1, 0:3] / Vh[-1, 3]

    return point_3d_homogeneous

class SerialCommunicationThread(threading.Thread):
    def __init__(self, ser, points_left, points_right, thetas, radians, R, t, caliberated, theta_offsets, starting_mode, coordinates):
        threading.Thread.__init__(self)
        self.ser = ser
        self.points_left = points_left
        self.points_right = points_right
        self.send_interval = 0.05
        self.last_send_time = time.time()
        self.thetas = thetas
        self.radians = radians
        self.R = R
        self.t = t
        self.caliberated = caliberated
        self.mode = starting_mode  # Start with mode 1
        self.theta_offsets = theta_offsets
        self.times = 0
        self.coordinates = coordinates


    def run(self):
        global frame_shape_x, frame_shape_y, d
        # self.handshake_with_arduino()

        while True:
            # print(f"{self.thetas}\t\t{self.theta_offsets}")
            current_time = time.time()

            if current_time - self.last_send_time >= self.send_interval:
                try:
                    if self.points_left and self.points_right:
                        # pass
                        # print("writing")
                        self.write_to_stream(self.mode, self.points_left[0][0], self.points_right[0][0], self.points_left[0][1], self.points_right[0][1])
                    self.last_send_time = current_time
                except Exception as e:
                    print("write exception : " + str(e))

            received_data = self.read_from_stream()

            try:
                if received_data:
                    self.thetas[0] = (received_data[0] - self.theta_offsets[0]) % 360
                    self.thetas[1] = (received_data[1] - self.theta_offsets[1]) % 360
                    self.thetas[2] = (received_data[2] - self.theta_offsets[2]) % 360

                    self.radians[0] = np.pi / 180 * self.thetas[0]
                    self.radians[1] = np.pi / 180 * self.thetas[1]
                    self.radians[2] = np.pi / 180 * self.thetas[2]

                    # self.write_radians_to_file()

                    self.R = [rotation_matrix(0, self.radians[1], self.radians[0]), rotation_matrix(0, self.radians[2], self.radians[0])]
                    self.t = [translation_vector(1, self.radians[0]), translation_vector(-1, self.radians[0])]

                    # R1 = rotation_matrix(0, radians[1], radians[0])
                    # R2 = rotation_matrix(0, radians[2], radians[0])

                    # t1 = translation_vector(1, radians[0])
                    # t2 = translation_vector(-1, radians[0])

                    # print(self.R, self.t)

                    if(self.mode == 2):
                        if (self.points_left and self.points_right):
                            count = 0
                            for i, (point_left, point_right) in enumerate(zip(self.points_left, self.points_right)):
                                # if i in [0]:
                                # self.coordinates.clear()
                               
                                # if i in [0, 11, 12, 13, 14, 15, 16]:
                                if(1):
                                    angle_rad = [self.thetas[0] * np.pi/180, (180 - self.thetas[1]) * np.pi/180, (self.thetas[2]) * np.pi/180]
                                    # X_world = triangulate_point((point_left[0] + 0.5 * frame_shape_x), (point_left[1] + 0.2 * frame_shape_y), (point_right[0] + 0.5 * frame_shape_x), (point_right[1] + + 0.2 * frame_shape_y), K, self.R, self.t)
                                    # X_world = triangulate_point((point_left[0] + 0.5 * frame_shape_x), (point_left[1] + 0.5 * frame_shape_y), (point_right[0] + 0.5 * frame_shape_x), (point_right[1] + 0.5 * frame_shape_y), K, self.R, self.t)
                                    # X_world1 = triangulate_point_direct((point_left[0] + 0.5 * frame_shape_x), (point_left[1] + 0.5 * frame_shape_y), (point_right[0] + 0.5 * frame_shape_x), (point_right[1] + 0.5 * frame_shape_y), K, self.R, self.t)
                                    X_world2 = triangulate_point_dlt((point_left[0] + 0.5 * frame_shape_x), (point_left[1] + 0.5 * frame_shape_y), (point_right[0] + 0.5 * frame_shape_x), (point_right[1] + 0.5 * frame_shape_y), K, self.R, self.t)

                                    # x, y, depth2, depth1 = calc_dist(angle_rad[1], angle_rad[2], X_world1, self.t)
                                    x, y, depth2, depth3, new_depth = calc_dist(angle_rad[1], angle_rad[2], X_world2, self.t)
                                    
                                    # self.coordinates1[0] = X_world1[0]
                                    # self.coordinates1[1] = X_world1[1]
                                    # self.coordinates1[2] = X_world1[2]
                                    # print(i, point_left, point_right, count)
                                    
                                    self.coordinates[count][0] = X_world2[0]
                                    self.coordinates[count][1] = X_world2[1]
                                    self.coordinates[count][2] = X_world2[2]
                                    
                                    count += 1


                                    # print(f"{i+1} (X, Y, Z): {X_world2}\t depth2 = {depth2}\tdepth3 = {depth3}\t new depth = {new_depth}\t{self.thetas}\t{x} {y}")
                            # print(self.coordinates)
                            # self.write_coordiantes_to_file()
                            self.points_left.clear()
                            self.points_right.clear()

            except Exception as e:
                print("read exception : " + str(e))
            
            if self.mode == 1 and self.caliberated[0] and self.caliberated[1] and self.times != 1:
                print("gay")
                # self.theta_offsets[0] = 0
                # self.theta_offsets[1] = self.thetas[1] - 180 + 2
                # self.theta_offsets[2] = self.thetas[2] - 0.5

                # self.theta_offsets[1] = self.thetas[1] - 180
                # self.theta_offsets[2] = self.thetas[2]
                
                self.times = 1
                while(True):
                    self.write_to_stream(4, 0, 0, 0, 0)
                    if(self.ser.in_waiting > 0):
                        # print("ola")
                        data = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        # print(data)
                        try:
                            if(int(data) == 6969):
                                print("celebration")
                                self.set_mode(2)
                                self.write_to_stream(self.mode, 0, 0, 0, 0)
                                break
                        except Exception as e:
                            print("calib exception : " + str(e))
                    time.sleep(0.05)


    def set_mode(self, mode):
        self.mode = mode

    def handshake_with_arduino(self):
        while True:
            if self.ser.in_waiting > 0:
                handshake = self.ser.readline().decode('utf-8').strip()
                if handshake == "READY":
                    self.ser.write("START\n".encode())
                    break

    def read_from_stream(self):
        if self.ser.in_waiting > 0:
            data = self.ser.readline().decode('utf-8', errors='ignore').strip()
            # print(data)
            try:
                data_list = list(map(float, data.split(',')))
                return data_list
            except ValueError as e:
                pass
        return None
    
    def write_to_stream(self, mode, x_left, x_right, y_left, y_right):
        data = f"{(y_right + y_left)//2},{x_left},{x_right},{mode}\n"
        # print(f"Sent to Arduino : {data}")
        ser.write(data.encode())

    def write_coordiantes_to_file(self):
        li = [0, 11, 12, 13, 14, 15, 16]
        count = 0
        with open("coordinates.txt", "w") as file:
            for i in self.coordinates:
                file.write(f"{li[count]},{i[0]},{i[1]},{i[2]}\n")
                count += 1



def main():
    starting_mode = int(input("Enter Mode : \n1: Calibration\n2: Person Tracking\n3: Aruco Stepper track\n"))


    serial_thread = SerialCommunicationThread(ser, points_left, points_right, thetas, radians, R, t, caliberated, theta_offsets, starting_mode, coordinates)
    left_cam_thread = CameraThread(1, "Camera 1", points_left, serial_thread, caliberated, starting_mode)
    right_cam_thread = CameraThread(2, "Camera 2", points_right, serial_thread, caliberated, starting_mode)
         
                            
    serial_thread.start()
    left_cam_thread.start()
    right_cam_thread.start()

    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=1, repeat=True)
    plt.show()

    left_cam_thread.join()
    right_cam_thread.join()
    serial_thread.join()

if __name__ == "__main__":
    main()
