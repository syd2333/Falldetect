from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import threading
from queue import Queue
import time
import subprocess as sp
import json
import requests
import os
from pyngrok import ngrok, conf
from flasgger import Swagger

app = Flask(__name__)
CORS(app)
Swagger(app)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class FallDetectionCamera:
    def __init__(self, camera_id, rtmp_url, push_url):
        self.camera_id = camera_id
        self.rtmp_url = rtmp_url
        self.push_url = push_url
        self.video_processing = False
        self.video_streaming = False
        self.frame_buffer = Queue(maxsize=72)
        self.buffer_lock = threading.Lock()
        self.stream_event = threading.Event()
        self.process_thread = None
        self.stream_thread = None
        self.last_alert_time = 0

    def start_ffmpeg_stream(self, width, height, fps):
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{width}x{height}",
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',
            '-tune', 'ull',
            '-zerolatency', 'true',
            '-b:v', '1000k',
            '-maxrate', '1500k',
            '-bufsize', '2000k',
            '-pix_fmt', 'yuv420p',
            '-g', str(fps * 2),
            '-f', 'flv',
            self.push_url
        ]
        try:
            return sp.Popen(command, stdin=sp.PIPE)
        except Exception as e:
            print(f"Error starting FFmpeg: {e}")
            return None

    @staticmethod
    def is_body_flat(landmarks):
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
        hip_height = (left_hip.y + right_hip.y) / 2
        ankle_height = (left_ankle.y + right_ankle.y) / 2

        max_height_diff = max(abs(shoulder_height - hip_height),
                              abs(shoulder_height - ankle_height),
                              abs(hip_height - ankle_height))

        return max_height_diff < 0.15, max_height_diff

    @staticmethod
    def calculate_head_speed(current_pos, prev_positions, fps):
        if len(prev_positions) < 2:
            return 0

        speeds = []
        for i in range(1, len(prev_positions)):
            distance = np.linalg.norm(np.array(current_pos) - np.array(prev_positions[i]))
            speed = distance * fps / i
            speeds.append(speed)

        return np.mean(speeds)

    def process_frame(self, image, pose, prev_head_positions, fall_detected, fall_frames, fps):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            is_flat, height_diff = self.is_body_flat(landmarks)

            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            head_pos = (nose.x, nose.y)
            prev_head_positions.append(head_pos)
            head_speed = self.calculate_head_speed(head_pos, list(prev_head_positions), fps)

            if is_flat and head_speed > 0.05:
                fall_frames += 1
                if fall_frames > fps / 5:  # 连续0.2秒检测到才算摔倒
                    fall_detected = True
            else:
                fall_frames = max(0, fall_frames - 1)
                if fall_frames == 0:
                    fall_detected = False

            status = "FALL DETECTED!" if fall_detected else "Normal"
            color = (0, 0, 255) if fall_detected else (0, 255, 0)
            cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, f"Flat Frames: {fall_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(image, f"Height Diff: {height_diff:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"Head Speed: {head_speed:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return image, fall_detected, fall_frames, prev_head_positions

    def send_alert(self):
        current_time = time.time()

        if current_time - self.last_alert_time < 5:
            return

        alert_info = {
            "camera_id": self.camera_id,
            "alert_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "alert_module": "falldetect",
            "alert_content": "Fall detected"
        }

        print(json.dumps(alert_info, indent=2))

        target_url = "http://10.60.131.218:8080/api/alert"
        try:
            response = requests.post(target_url, json=alert_info, timeout=5)
            if response.status_code == 200:
                print("Alert sent successfully")
            else:
                print(f"Failed to send alert. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending alert: {e}")

        self.last_alert_time = current_time

    def stream_frames(self):
        width, height = 852, 480
        fps = 24
        ffmpeg_process = None

        frame_time = 1 / fps
        last_frame_time = time.time()

        while not self.stream_event.is_set():
            if self.video_streaming:
                if ffmpeg_process is None:
                    ffmpeg_process = self.start_ffmpeg_stream(width, height, 20)
                    if ffmpeg_process is None:
                        print("Failed to start FFmpeg process")
                        time.sleep(1)
                        continue

                current_time = time.time()
                if current_time - last_frame_time >= frame_time:
                    if not self.frame_buffer.empty():
                        frame = self.frame_buffer.get()
                        try:
                            ffmpeg_process.stdin.write(frame.tobytes())
                        except BrokenPipeError:
                            print("FFmpeg process has terminated unexpectedly")
                            ffmpeg_process = None
                        last_frame_time = current_time
                    else:
                        time.sleep(0.001)
                else:
                    time.sleep(0.001)
            else:
                if ffmpeg_process is not None:
                    ffmpeg_process.stdin.close()
                    ffmpeg_process.wait()
                    ffmpeg_process = None
                time.sleep(0.1)

        if ffmpeg_process is not None:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
        print("Streaming stopped")

    def process_video(self):
        cap = cv2.VideoCapture(self.rtmp_url)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 852)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 24

        fall_detected = False
        fall_frames = 0
        prev_head_positions = deque(maxlen=5)

        self.stream_thread = threading.Thread(target=self.stream_frames)
        self.stream_thread.start()

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.video_processing:
                success, image = cap.read()
                if not success:
                    print("忽略空帧。")
                    continue

                image = cv2.resize(image, (852, 480))

                processed_image, fall_detected, fall_frames, prev_head_positions = self.process_frame(
                    image, pose, prev_head_positions, fall_detected, fall_frames, fps)

                if fall_detected:
                    self.send_alert()

                if self.video_streaming:
                    with self.buffer_lock:
                        if self.frame_buffer.full():
                            temp_buffer = list(self.frame_buffer.queue)
                            self.frame_buffer.queue.clear()
                            for i, frame in enumerate(temp_buffer):
                                if i % 10 != 0:
                                    self.frame_buffer.put(frame)
                        self.frame_buffer.put(processed_image)

        self.stream_event.set()
        self.stream_thread.join()
        cap.release()

    def start_processing(self):
        if not self.video_processing:
            self.video_processing = True
            self.stream_event.clear()
            self.frame_buffer.queue.clear()
            self.process_thread = threading.Thread(target=self.process_video)
            self.process_thread.start()

    def stop_processing(self):
        if self.video_processing:
            self.video_processing = False
            self.video_streaming = False
            self.stream_event.set()
            if self.process_thread:
                self.process_thread.join()
            if self.stream_thread:
                self.stream_thread.join()
            self.process_thread = None
            self.stream_thread = None
            self.frame_buffer.queue.clear()

    def start_streaming(self):
        if not self.video_streaming:
            self.video_streaming = True

    def stop_streaming(self):
        if self.video_streaming:
            self.video_streaming = False


class FallDetectionManager:
    def __init__(self):
        self.cameras = {}

    def add_camera(self, camera_id, rtmp_url, push_url):
        if camera_id not in self.cameras:
            self.cameras[camera_id] = FallDetectionCamera(camera_id, rtmp_url, push_url)

    def remove_camera(self, camera_id):
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            camera.stop_processing()
            camera.stop_streaming()
            del self.cameras[camera_id]

    def start_processing(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id].start_processing()

    def stop_processing(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop_processing()

    def start_streaming(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id].start_streaming()

    def stop_streaming(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop_streaming()

    def get_all_cameras(self):
        return {camera_id: {"rtmp_url": camera.rtmp_url, "push_url": camera.push_url,
                            "processing": camera.video_processing, "streaming": camera.video_streaming}
                for camera_id, camera in self.cameras.items()}


fall_detection_manager = FallDetectionManager()


@app.route('/')
def hello_world():
    """
    Check if Fall Detection System is running
    ---
    responses:
      200:
        description: A simple message indicating the system is running
    """
    return 'Fall Detection System is running!'


@app.route('/api/falldetect/add_camera', methods=['POST'])
def add_camera():
    """
    Add new cameras for fall detection
    ---
    tags:
      - Fall Detection
    parameters:
      - in: body
        name: body
        schema:
          type: object
          properties:
            cameras:
              type: array
              items:
                type: object
                properties:
                  camera_id:
                    type: string
                  rtmp_url:
                    type: string
                  push_url:
                    type: string
    responses:
      200:
        description: Cameras added successfully
        schema:
          type: object
          properties:
            message:
              type: string
            added_cameras:
              type: array
              items:
                type: string
    """
    data = request.json
    added_cameras = []
    for camera in data.get('cameras', []):
        camera_id = camera.get('camera_id')
        rtmp_url = camera.get('rtmp_url')
        push_url = camera.get('push_url')
        if camera_id and rtmp_url and push_url:
            fall_detection_manager.add_camera(camera_id, rtmp_url, push_url)
            added_cameras.append(camera_id)

    return jsonify({
        "message": "Cameras added successfully",
        "added_cameras": added_cameras
    }), 200


@app.route('/api/falldetect/<camera_id>/start_process', methods=['POST'])
def start_processing(camera_id):
    """
    Start video processing for a specific camera
    ---
    tags:
      - Fall Detection
    parameters:
      - name: camera_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Video processing started
        schema:
          type: object
          properties:
            status:
              type: string
    """
    fall_detection_manager.start_processing(camera_id)
    return jsonify({"status": "Video processing started"}), 200


@app.route('/api/falldetect/<camera_id>/stop_process', methods=['POST'])
def stop_processing(camera_id):
    """
    Stop video processing for a specific camera
    ---
    tags:
      - Fall Detection
    parameters:
      - name: camera_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Video processing stopped
        schema:
          type: object
          properties:
            status:
              type: string
    """
    fall_detection_manager.stop_processing(camera_id)
    return jsonify({"status": "Video processing stopped"}), 200


@app.route('/api/falldetect/<camera_id>/start_stream', methods=['POST'])
def start_streaming(camera_id):
    """
    Start video streaming for a specific camera
    ---
    tags:
      - Fall Detection
    parameters:
      - name: camera_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Video streaming started
        schema:
          type: object
          properties:
            status:
              type: string
    """
    fall_detection_manager.start_streaming(camera_id)
    return jsonify({"status": "Video streaming started"}), 200


@app.route('/api/falldetect/<camera_id>/stop_stream', methods=['POST'])
def stop_streaming(camera_id):
    """
    Stop video streaming for a specific camera
    ---
    tags:
      - Fall Detection
    parameters:
      - name: camera_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Video streaming stopped
        schema:
          type: object
          properties:
            status:
              type: string
    """
    fall_detection_manager.stop_streaming(camera_id)
    return jsonify({"status": "Video streaming stopped"}), 200


@app.route('/api/falldetect/get_cameras', methods=['GET'])
def get_cameras():
    """
    Get all cameras
    ---
    tags:
      - Fall Detection
    responses:
      200:
        description: List of all cameras
        schema:
          type: object
          additionalProperties:
            type: object
            properties:
              rtmp_url:
                type: string
              push_url:
                type: string
              processing:
                type: boolean
              streaming:
                type: boolean
    """
    return jsonify(fall_detection_manager.get_all_cameras()), 200


# 新增的 ngrok 相关函数
def init_ngrok():
    ngrok_path = r"C:\ngrok\ngrok.exe"
    conf.get_default().ngrok_path = ngrok_path
    try:
        tunnel = ngrok.connect(5000)
        public_url = tunnel.public_url
        print(f"ngrok tunnel available at: {public_url}")
        return public_url
    except Exception as e:
        print(f"Error starting ngrok: {e}")
        import traceback
        traceback.print_exc()
        return None


def send_ngrok_url_to_server(ngrok_url, server_ip, server_port):
    server_endpoint = f"http://{server_ip}:{server_port}/geturl/register_ngrok"
    payload = {
        "ngrok_url": ngrok_url,
        "service_name": "falldetect"  # You can change this to identify your service
    }
    print(payload)
    try:
        response = requests.post(server_endpoint, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"Successfully sent ngrok URL to server at {server_ip}:{server_port}")
            return True
        else:
            print(f"Failed to send ngrok URL. Server responded with status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error sending ngrok URL to server: {e}")
        return False


# 修改 init_app 函数来使用这个新函数
def init_app():
    public_url = str(init_ngrok())
    print(public_url)
    if public_url:
        server_ip = "10.60.131.218"  # 替换为实际的公网 IP
        server_port = 8080  # 替换为服务器上实际使用的端口

        # 设置一个定时器来延迟发送 ngrok URL
        def delayed_send():
            time.sleep(5)  # 等待5秒
            send_ngrok_url_to_server(public_url, server_ip, server_port)

        # 使用 Timer 对象来延迟执行
        from threading import Timer
        timer = Timer(5, delayed_send)
        timer.start()


# 初始化应用
init_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
