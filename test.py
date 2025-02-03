# ref : https://qiita.com/k-keita/items/4546396fa5b7c242c4df

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import warnings
import sysv_ipc
warnings.simplefilter('ignore', FutureWarning)

# Message queue ID
QUE_ID = 1234

send_mode = False

def send_angles_to_queue(angles, queue_id):
    '''Send angles DataFrame to message queue'''
    # Convert DataFrame to comma-separated string
    angles_str = angles.to_csv(header=False, index=False, sep=',')
    
    # Create message queue
    mq = sysv_ipc.MessageQueue(queue_id, sysv_ipc.IPC_CREAT)
    
    # Send message
    mq.send(angles_str)

def main():

    input_mode = input("Send angles to message queue? (y/n): ")
    if input_mode == 'y':
        send_mode = True
    else:
        send_mode = False
    
    # MediaPipeのポーズ推定モジュールをインスタンス化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Webカメラを開く
    cap = cv2.VideoCapture(0)

    # カメラが開けない場合のエラーハンドリング
    if not cap.isOpened():
        print("Error: Camera not found.")
        exit()

    # # 3Dプロットのセットアップ
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.ion()

    # 画像処理ループ
    while cap.isOpened():
        # DataFrameの初期化
        landmarks_df = pd.DataFrame(columns=['id', 'x', 'y', 'z', 'visibility'])
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # BGRからRGBに変換（MediaPipeはRGB形式を期待するため）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipeで骨格推定を実行
        results = pose.process(rgb_frame)

        # 結果を描画
        if results.pose_landmarks:
            # 骨格のランドマークを描画
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 各ランドマークの座標と可視性を取得して表示
            x_data = []
            y_data = []
            z_data = []
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                x_data.append(landmark.x)
                z_data.append(landmark.y)  # y軸とz軸を入れ替え
                y_data.append(landmark.z)  # y軸とz軸を入れ替え
                #print(f"Landmark {id}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z}, visibility: {landmark.visibility})")
                new_row = pd.DataFrame([[id, landmark.x, landmark.y, landmark.z, landmark.visibility]], 
                                    columns=['id', 'x', 'y', 'z', 'visibility'])
                landmarks_df = pd.concat([landmarks_df, new_row], ignore_index=True)
            
            # # 3Dプロットを更新
            # ax.clear()
            # ax.scatter(x_data, y_data, z_data, c='r', marker='o')
            # plt.draw()
            # plt.pause(0.001)
            
            angles = angle_calq(landmarks_df)
            #elbow_rot = rot_calq(landmarks_df)
            
            print(landmarks_df)
            print(angles)
            #print(elbow_rot)
            
            if send_mode:
                # Send angles to message queue
                send_angles_to_queue(angles, QUE_ID)
                
            #servo_angles =
            #khr_send(servo_angles)

        # 画像を表示
        cv2.imshow("Pose Estimation", frame)

        # 'q'を押すと終了
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
    
    
def angle_calq(landmarks_df):
    '''2つのベクトル間の単純角度を計算'''
    # 計算する関節のペアを指定
    joint_pairs = {
        2 : [12, 11, 24, 23], # BODY_Y
        9 : [13, 11, 13, 15], # L_ELBOW_P
        10: [14, 12, 14, 16], # R_ELBOW_P
        15: [23, 25, 23, 11], # L_HIP_P
        16: [24, 26, 24, 12], # R_HIP_P
        19: [25, 23, 25, 27], # L_KNEE_P
        20: [26, 24, 26, 28], # R_KNEE_P

    }
    
    angle_df = pd.DataFrame(columns=['servo_id', 'angle'])
    for i, (key, joint_pair) in enumerate(joint_pairs.items()):
        # 3点の座標を取得
        p_a1 = landmarks_df[landmarks_df['id'] == joint_pair[0]][['x', 'y', 'z']].values
        p_a2 = landmarks_df[landmarks_df['id'] == joint_pair[1]][['x', 'y', 'z']].values
        p_b1 = landmarks_df[landmarks_df['id'] == joint_pair[2]][['x', 'y', 'z']].values
        p_b2 = landmarks_df[landmarks_df['id'] == joint_pair[3]][['x', 'y', 'z']].values

        # ベクトルを計算
        v1 = p_a2 - p_a1
        v2 = p_b2 - p_b1

        v1 = v1.squeeze()
        v2 = v2.squeeze()

        # ベクトルの内積を計算
        dot = np.dot(v1, v2)

        # ベクトルのノルムを計算
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # 角度を計算
        cos_theta = dot / (norm1 * norm2)
        theta = np.arccos(cos_theta)
        degree = np.degrees(theta)

        cross_product = np.cross(v1, v2)

        if cross_product[2] < 0:
            degree = -degree

        # DataFrameに追加
        new_row = pd.DataFrame([[key, degree]], columns=['servo_id', 'angle'])
        angle_df = pd.concat([angle_df, new_row], ignore_index=True)
    return angle_df

def rot_calq(landmarks_df):
    '''肘のローテーション角度を取得'''
    # 計算する関節のペアを指定
    joint_pairs = [[12,11,13,15],[14,12,11,16]]
    
    angle_df = pd.DataFrame(columns=['id', 'angle'])
    for i, joint_pair in enumerate(joint_pairs):
        # 4点の座標を取得
        p_t1 = landmarks_df[landmarks_df['id'] == joint_pair[0]][['x', 'y', 'z']].values
        p_t2 = landmarks_df[landmarks_df['id'] == joint_pair[1]][['x', 'y', 'z']].values
        p_t3 = landmarks_df[landmarks_df['id'] == joint_pair[2]][['x', 'y', 'z']].values
        p_z = landmarks_df[landmarks_df['id'] == joint_pair[3]][['x', 'y', 'z']].values

        # ベクトルを計算
        v1 = p_t1 - p_t2
        v2 = p_t3 - p_t2
        v3 = p_z - p_t2

        # v1,v2平面の法線ベクトルを計算
        n1 = np.cross(v1, v2)

        # v2を法線ベクトルとした平面上で、n1とv3のなす角度を計算
        cos_theta = np.dot(n1, v3) / (np.linalg.norm(n1) * np.linalg.norm(v3))
        theta = np.arccos(cos_theta) * 180 / np.pi

        # DataFrameに追加
        new_row = pd.DataFrame([[i, theta]], columns=['id', 'rotation'])
        rotation_df = pd.concat([angle_df, new_row], ignore_index=True)
        
    return rotation_df


if __name__ == '__main__':
    main()