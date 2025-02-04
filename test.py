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

def clear_message_queue(queue_id):
    '''メッセージキューのキャッシュをクリアする'''
    try:
        # メッセージキューに接続
        mq = sysv_ipc.MessageQueue(queue_id)
        
        # メッセージキュー内のすべてのメッセージを削除
        while True:
            try:
                mq.receive(block=False)
            except sysv_ipc.BusyError:
                break
    except sysv_ipc.ExistentialError:
        print(f"Message queue with ID {queue_id} does not exist.")

def main():

    clear_message_queue(QUE_ID)

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
            
            array = np.zeros((22,2))

            angle_df = pd.DataFrame(array, columns=['servo_id', 'angle'])
            angle_df['servo_id'] = range(1,23)

            angle_df = angle_calq(landmarks_df,angle_df)
            #angle_df = rot_calq(landmarks_df, angle_df)
            value_df = angle_to_value(angle_df)
            
            print(landmarks_df)
            print(angle_df)
            print(value_df)
            
            if send_mode:
                # Send angles to message queue
                send_angles_to_queue(value_df.astype(int), QUE_ID)

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
    
    
def angle_calq(landmarks_df, angle_df):
    '''2つのベクトル間の単純角度を計算'''
    # 計算する関節のペアを指定
    joint_pairs = {
        2 : [28, 27, 12, 11], # BODY_Y
        9 : [13, 11, 13, 15], # L_ELBOW_P
        10: [14, 12, 14, 16], # R_ELBOW_P
        15: [23, 25, 23, 11], # L_HIP_P
        16: [24, 26, 24, 12], # R_HIP_P
        19: [25, 23, 25, 27], # L_KNEE_P
        20: [26, 24, 26, 28], # R_KNEE_P

        }
    
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
            degree = - degree
        
        
        # DataFrameに追加
        #angle_df.loc[key-1, 'servo_id'] = key
        angle_df.loc[key-1, 'angle'] = degree
        
    return angle_df

def rot_calq(landmarks_df, angle_df):
    '''肘のローテーション角度を取得'''
    # 計算する関節のペアを指定
    joint_pairs = {
        3 : [12,11,13,15],
        4 : [14,12,11,16],
        }
    
    for i, (key, joint_pair) in enumerate(joint_pairs.items()):
        # 4点の座標を取得
        p_t1 = landmarks_df[landmarks_df['id'] == joint_pair[0]][['x', 'y', 'z']].values
        p_t2 = landmarks_df[landmarks_df['id'] == joint_pair[1]][['x', 'y', 'z']].values
        p_t3 = landmarks_df[landmarks_df['id'] == joint_pair[2]][['x', 'y', 'z']].values
        p_z  = landmarks_df[landmarks_df['id'] == joint_pair[3]][['x', 'y', 'z']].values

        # ベクトルを計算
        v1 = p_t1 - p_t2
        v2 = p_t3 - p_t2
        v3 = p_z - p_t2

        # v1,v2平面の法線ベクトルを計算
        n1 = np.cross(v1, v2)

        # v2を法線ベクトルとした平面上で、n1とv3のなす角度を計算
        cos_theta = np.dot(n1, v3) / (np.linalg.norm(n1) * np.linalg.norm(v3))
        theta = np.arccos(cos_theta)
        degree = np.degrees(theta)

        # DataFrameに追加
        angle_df[key-1] = [key, degree]
        
    return angle_df

def angle_to_value(angle_df):
    '''角度をサーボの値に変換'''
    initial_angle = {
         1:  0,  2:  0,  3: 90,  4: 90,  5: 90,  6:270,  7:  0,  8:  0,  9:-180, 10:180,
        11:  0, 12: 90, 13:  0, 14:  0, 15:  0, 16:  0, 17:  0, 18:  0, 19:  0, 20:  0,
        21:  0, 22:  0
    }

    limit_value = {
            1:[4000, 11000],
            2:[5000, 10000],
            3:[5500, 12000],
            4:[5000, 10000],
            5:[5000, 10000],
            }

    print(angle_df)
    print(initial_angle.values())

    angle_df['angle'] = angle_df['angle'] - list(initial_angle.values())

    # 角度をサーボの値に変換
    value_df = angle_df.copy()
    value_df['angle'] = angle_df['angle'] / 90 * 2500 + 7500

    

    return value_df


if __name__ == '__main__':
    main()