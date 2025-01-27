# ref : https://qiita.com/k-keita/items/4546396fa5b7c242c4df

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

def main():
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
            
            elbow_angle = angle_calq(landmarks_df)
            
            print(landmarks_df)
            print(elbow_angle)

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
    # 計算する関節のペアを指定
    joint_pairs = [[12, 14, 16], [11, 13, 15]]
    
    angle_df = pd.DataFrame(columns=['id', 'angle'])
    for i, joint_pair in enumerate(joint_pairs):
        # 3点の座標を取得
        p1 = landmarks_df[landmarks_df['id'] == joint_pair[0]][['x', 'y', 'z']].values
        p2 = landmarks_df[landmarks_df['id'] == joint_pair[1]][['x', 'y', 'z']].values
        p3 = landmarks_df[landmarks_df['id'] == joint_pair[2]][['x', 'y', 'z']].values

        # ベクトルを計算
        v1 = p1 - p2
        v2 = p3 - p2

        # ベクトルの内積を計算
        dot = np.dot(v1, v2.T)

        # ベクトルのノルムを計算
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # 角度を計算
        cos_theta = dot / (norm1 * norm2)
        theta = np.arccos(cos_theta) * 180 / np.pi

        # DataFrameに追加
        new_row = pd.DataFrame([[i, theta]], columns=['id', 'angle'])
        angle_df = pd.concat([angle_df, new_row], ignore_index=True)
    return angle_df


if __name__ == '__main__':
    main()