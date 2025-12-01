import os
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

# F:\Goodminton\data\processed\
# data\raw_videos

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

POSE_LANDMARKS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]


def extract_landmarks_from_video(video_path, target_fps=30):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(original_fps / target_fps))
    
    all_frames = []
    frame_count = 0
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                
                if results.pose_landmarks:
                    landmarks = [
                        (lm.x, lm.y, lm.z, lm.visibility)
                        for lm in results.pose_landmarks.landmark
                    ]
                    all_frames.append(landmarks)
                else:
                    all_frames.append(None)
            
            frame_count += 1
    
    cap.release()
    return all_frames


def normalize_landmarks(landmarks):
  '''
  set original point relative to the body
  '''
  if landmarks is None:
    return None
  
  LEFT_HIP = 23
  RIGHT_HIP = 24

  left_hip = landmarks[LEFT_HIP]
  right_hip = landmarks[RIGHT_HIP]
  hip_center_x = (left_hip[0] + right_hip[0]) / 2
  hip_center_y = (left_hip[1] + right_hip[1]) / 2
  hip_center_z = (left_hip[2] + right_hip[2]) / 2

  normalized = []
  for lm in landmarks:
    x,y,z,visibility = lm
    
    x_norm = x - hip_center_x
    y_norm = y - hip_center_y
    z_norm = z - hip_center_z

    normalized.append((x_norm, y_norm, z_norm, visibility))

  return normalized


def landmarks_to_dataframe(video_name, all_frames_landmarks):
  rows = []
  for frame_idx, frame_landmarks in enumerate(all_frames_landmarks):
    if frame_landmarks is None:
      continue
    row = {
      'video_name' : video_name,
      'frame_idx': frame_idx
    }

    for landmark_idx, (x, y, z, visibility) in enumerate(frame_landmarks):
      landmark_name = POSE_LANDMARKS[landmark_idx]

      row[f'{landmark_name}_x'] = x
      row[f'{landmark_name}_y'] = y
      row[f'{landmark_name}_z'] = z
      row[f'{landmark_name}_visibility'] = visibility

    rows.append(row)

  data_frame = pd.DataFrame(rows)
  return data_frame


def process_all_videos(input_dirs, output_dir):
  all_dataframes = []
  all_labels = []

  for label, video_dir in input_dirs.items():
    for video_file in Path(video_dir).glob("*.mp4"):
      video_name = video_file.stem

      raw_landmarks = extract_landmarks_from_video(video_file)

      normalized_landmarks = []
      for frame_lm in raw_landmarks:
        norm_lm = normalize_landmarks(frame_lm)
        normalized_landmarks.append(norm_lm)

      df = landmarks_to_dataframe(video_name, normalized_landmarks)
      all_dataframes.append(df)

      all_labels.append({
                'video_name': video_name,
                'label': label
            })

  final_df = pd.concat(all_dataframes, ignore_index=True)
  final_df.to_csv(Path(output_dir) / "features_clear_test.csv", index=False)

  labels_df = pd.DataFrame(all_labels)
  labels_df.to_csv(Path(output_dir) / "labels_clear_test.csv", index=False)
  print(f"   features_clear_test.csv: {len(final_df)} lines")
  print(f"   labels_clear_test.csv: {len(labels_df)} lines")

def main():
    input_dirs = {
        'correct': r"F:\Goodminton\data\raw_videos\clear\test\correct",
        'incorrect': r"F:\Goodminton\data\raw_videos\clear\test\incorrect"
    }
    output_dir = r"F:\Goodminton\data\processed\clear\test"

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_dirs['correct']) and os.path.exists(input_dirs['incorrect']):
        print(f"Processing ALL VIDEOS (correct + incorrect)")
        process_all_videos(input_dirs, output_dir)
    else:
        print(f"ERROR: One or both directories not found:")
        for label, path in input_dirs.items():
            print(f"  {label}: {path} - {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")

if __name__ == "__main__":
  main()
