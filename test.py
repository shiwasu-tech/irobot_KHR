# ref : https://qiita.com/k-keita/items/4546396fa5b7c242c4df

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# MediaPipeのポーズ推定モジュールをインスタンス化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Webカメラを開く
cap = cv2.VideoCapture(0)

# カメラが開けない場合のエラーハンドリング
if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

# 3Dプロットのセットアップ
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()

# 画像処理ループ
while cap.isOpened():
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
            print(f"Landmark {id}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z}, visibility: {landmark.visibility})")
        
        # 3Dプロットを更新
        ax.clear()
        ax.scatter(x_data, y_data, z_data, c='r', marker='o')
        plt.draw()
        plt.pause(0.001)

    # 画像を表示
    cv2.imshow("Pose Estimation", frame)

    # 'q'を押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
