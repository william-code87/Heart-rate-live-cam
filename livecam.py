import cv2
import torch
import numpy as np
import mediapipe as mp
import scipy.signal
import time
from modelv1 import EfficientPhys_attention
from scipy.signal import butter, find_peaks

# Definisikan parameter
img_size = 72
window_size = 180
stride = 30
model_checkpoint = r'D:\Code\Project poker\main\rppg\EfficientPhys_model85_rorate10.pt'

def live_cam_bpm_prediction():
    # Load model
    a=0
    device = torch.device("cpu")
    model = EfficientPhys_attention(frame_depth=window_size, img_size=img_size, in_channels=3)
    model.eval()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)

    # Initialize video capture and MediaPipe face detection
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30  # Default fps jika tidak dapat dibaca
    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    [b_pulse, a_pulse] = butter(1, [0.6 / fps * 2, 3 / fps * 2], btype='bandpass')
    
    frames = []
    predict = []
    last_bpm_time = 0
    bpm_value = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat membaca frame dari kamera.")
            break
        a+=1
        # Face detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            for detection in results.detections:
                # Dapatkan bounding box dan buat persegi
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Buat kotak menjadi persegi
                if w > h:
                    y1 -= (w - h) // 2
                    h = w
                else:
                    x1 -= (h - w) // 2
                    w = h
                
                # Pastikan koordinat berada dalam batas frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(iw, x1 + w)
                y2 = min(ih, y1 + h)

                # Extract wajah dalam kotak persegi
                face_crop = frame[y1:y2, x1:x2]
                face_resized = cv2.resize(face_crop, (img_size, img_size))
                face_resized = face_resized / 255.0  # Normalisasi
                frames.append(face_resized)

                # Gambarkan kotak pada wajah
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if len(frames) == window_size:
            # print(frames)
            input_array = np.array(frames)  # [window_size, img_size, img_size, 3]
            input_tensor = torch.tensor(input_array, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # [window_size, 3, img_size, img_size]
            print(f"Input tensor shape: {input_tensor.shape}")  # Debugging line

            if input_tensor.size(0) == window_size:
                # Prediksi sinyal pulse
                with torch.no_grad():
                    try:
                        pulse_pred = model(input_tensor).squeeze().cpu().numpy()
                        predict.append(pulse_pred)
                        print(f"Pulse prediction appended. Total predictions: {len(predict)}")
                    except IndexError as e:
                        print(f"Error during model prediction: {e}")
                        continue

                frames = frames[stride:]  # Hapus frame yang sudah diproses

                if len(predict) > 0:
                    # Filter the pulse signal
                    pulse_pred_combined = np.array(predict).reshape(-1)
                    pulse_filtered = scipy.signal.filtfilt(b_pulse, a_pulse, pulse_pred_combined)

                    # Calculate BPM
                    fft_len = 16384
                    fft = np.abs(np.fft.rfft(pulse_filtered, n=fft_len))
                    bpm_place = np.argmax(fft)
                    bpm = np.fft.rfftfreq(fft_len, 1 / fps)
                    bpm_value = bpm[bpm_place] * 60
                    last_bpm_time = time.time()  # Update last BPM calculation time

                    print(f"Calculated BPM: {bpm_value:.2f}")
                    print(a)

            else:
                print(f"Unexpected input_tensor size: {input_tensor.size()}")

        # Tampilkan BPM pada video feed setiap 4 detik
        if time.time() - last_bpm_time <= 6:
            cv2.putText(frame, f"BPM: {bpm_value:.2f}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Tampilkan feed video
        cv2.imshow('Live BPM', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_cam_bpm_prediction()
