import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def visualize_video(input_path, output_path):
    """
    Draw MediaPipe pose landmarks on the video
    
    Args:
        input_path: input video path
        output_path: output video path (with skeleton rendering)
    """
    cap = cv2.VideoCapture(input_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            out.write(image)
            frame_count += 1
            
    
    cap.release()
    out.release()
    print(f"Done, saved at: {output_path}")

def main():
    input_dir = r"F:\Goodminton\data\raw_videos\clear\incorrect"
    output_dir = r"F:\Goodminton\outputs\visualized_videos"
    
    os.makedirs(output_dir, exist_ok=True)
    
    test_video = "clear_inc_006.mp4"  
    input_path = os.path.join(input_dir, test_video)
    output_path = os.path.join(output_dir, f"vis_{test_video}")
    
    if os.path.exists(input_path):
        print(f"Processing: {test_video}")
        visualize_video(input_path, output_path)
    else:
        print(f"Not exists: {input_path}")

if __name__ == "__main__":
    main()