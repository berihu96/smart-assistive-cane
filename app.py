import cv2
import torch
import time
import numpy as np
import subprocess
import queue
import threading
import serial
import serial.tools.list_ports
from ultralytics import YOLO

# ==========================================
# 0. CONFIGURATION & SENSOR THRESHOLDS
# ==========================================
SERIAL_PORT = r'\\.\COM10'  # Adjust to match your Arduino COM port
BAUD_RATE = 9600

# Sensor & Vision Parameters
MIN_DIST_CM = 15.0      # Max speed (PWM 255) at or below this distance
MAX_DIST_CM = 80.0      # Motor completely turns OFF (PWM 0) at or above this distance

# FIX 1: Lowered height thresholds so smaller/distant objects trigger vibration
MIN_BOX_HEIGHT_PX = 20   # Bounding box >= 20px starts motor at minimum vibration
MAX_BOX_HEIGHT_PX = 220  # Bounding box >= 220px ramps motor to 100% PWM (255)

arduino = None
serial_lock = threading.Lock()
last_sent_cmd = None
latest_ir_distance = 999.0

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print(f"[INFO] Serial connection established on {SERIAL_PORT}")
except Exception as e:
    print(f"[WARNING] Serial port {SERIAL_PORT} unavailable ({e}). Running in vision-only mode.")

def send_motor_command(cmd_str: str):
    """Transmits string commands ('M<PWM>' or 'R') over serial."""
    global last_sent_cmd
    if arduino and arduino.is_open:
        if cmd_str != last_sent_cmd:  # Send only on state change to prevent serial flooding
            with serial_lock:
                try:
                    arduino.write(f"{cmd_str}\n".encode('utf-8'))
                    arduino.flush()
                    last_sent_cmd = cmd_str
                except Exception as e:
                    print(f"[ERROR] Serial write error: {e}")

def serial_reader_worker():
    """Background thread to read IR distance telemetry from Arduino."""
    global latest_ir_distance
    while arduino and arduino.is_open:
        try:
            with serial_lock:
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("IR_CM:"):
                        val_str = line.split(":")[1]
                        latest_ir_distance = float(val_str)
        except Exception:
            pass
        time.sleep(0.02)

if arduino and arduino.is_open:
    sr_thread = threading.Thread(target=serial_reader_worker, daemon=True)
    sr_thread.start()


# ==========================================
# 1. SPEECH ENGINE
# ==========================================
current_priority = 999
current_speech_process = None
speech_lock = threading.Lock()
last_spoken_time = {}

def speak(text, priority_level, cooldown_sec=1.5):
    """Non-blocking text-to-speech using PowerShell SpeechSynthesizer."""
    global current_speech_process, current_priority
    current_time = time.time()
    
    if text in last_spoken_time:
        if current_time - last_spoken_time[text] < cooldown_sec:
            return

    with speech_lock:
        is_playing = False
        if current_speech_process is not None:
            if current_speech_process.poll() is None:
                is_playing = True
            else:
                current_priority = 999

        if is_playing and priority_level > current_priority:
            return

        if is_playing and priority_level <= current_priority:
            try:
                current_speech_process.terminate()
            except Exception:
                pass

        last_spoken_time[text] = current_time
        current_priority = priority_level

        clean_text = text.replace("'", "").replace('"', "")
        ps_command = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = 2; "
            f"$synth.Speak('{clean_text}')"
        )

        current_speech_process = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", ps_command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )


# ==========================================
# 2. NAVIGATION THREAD
# ==========================================
class GuidanceTimerThread(threading.Thread):
    def __init__(self, destination, interval_seconds=10.0):
        super().__init__(daemon=True)
        self.destination = destination
        self.interval_seconds = interval_seconds
        self.running = True

        self.steps = [
            f"Navigation started to {self.destination}. Walk straight ahead.",
            f"Heading toward {self.destination}. Continue straight.",
            f"Turn slight left in 5 meters toward {self.destination}.",
            f"Turn left now. Continue straight.",
            f"Walk straight ahead. You are nearing {self.destination}.",
            f"You have arrived at {self.destination}."
        ]
        self.step_idx = 0

    def run(self):
        while self.running:
            if self.step_idx < len(self.steps):
                instruction = self.steps[self.step_idx]
                speak(instruction, priority_level=2, cooldown_sec=0)
                self.step_idx += 1
            else:
                speak(f"You have reached {self.destination}.", priority_level=2, cooldown_sec=0)
            
            time.sleep(self.interval_seconds)


# ==========================================
# 3. MODELS
# ==========================================
print("[INFO] Loading YOLOv8 Model...")
yolo_model = YOLO("yolov8n.pt")

print("[INFO] Loading MiDaS Depth Model...")
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform


# ==========================================
# 4. ASYNC DEPTH WORKER
# ==========================================
frame_queue = queue.Queue(maxsize=1)
depth_queue = queue.Queue(maxsize=1)

def depth_worker():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img)

        with torch.no_grad():
            prediction = midas(input_batch)

            if len(prediction.shape) == 2:
                prediction = prediction.unsqueeze(0).unsqueeze(0)
            elif len(prediction.shape) == 3:
                prediction = prediction.unsqueeze(1)

            prediction = torch.nn.functional.interpolate(
                prediction,
                size=(480, 640),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        if depth_queue.full():
            try: depth_queue.get_nowait()
            except queue.Empty: pass
        depth_queue.put(depth_map)

depth_thread = threading.Thread(target=depth_worker, daemon=True)
depth_thread.start()


# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam feed.")
        return

    destination = input("Enter your target destination: ").strip()
    if not destination:
        destination = "the target destination"

    guidance_thread = GuidanceTimerThread(destination, interval_seconds=10.0)
    guidance_thread.start()

    latest_depth = None

    # FIX 2: Expanded target class list so indoor items trigger YOLO detection
    target_classes = [
        "person", "car", "bus", "truck", "chair", "table", "laptop", 
        "cell phone", "bottle", "cup", "keyboard", "mouse", "backpack", 
        "potted plant", "couch", "bed", "tv", "door"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Empty frame received.")
            continue

        if frame_queue.empty():
            frame_queue.put(frame.copy())

        if not depth_queue.empty():
            latest_depth = depth_queue.get_nowait()

        # Step 1: Detect objects via YOLO
        results = yolo_model(frame, stream=True, verbose=False)
        
        max_box_height = 0
        closest_pos = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = yolo_model.names[cls_id]
                confidence = float(box.conf[0])

                if confidence > 0.25 and class_name in target_classes:
                    box_height = y2 - y1
                    center_x = (x1 + x2) // 2
                    
                    if center_x < 210:
                        position = "left"
                    elif 210 <= center_x <= 430:
                        position = "straight ahead"
                    else:
                        position = "right"

                    if box_height > max_box_height:
                        max_box_height = box_height
                        closest_pos = position

                    # Visual feedback bounding boxes
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} ({box_height}px)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Step 2: Proportional PWM Calculation (Vision vs IR)
        # Vision-based PWM logic
        if max_box_height <= MIN_BOX_HEIGHT_PX:
            vision_pwm = 0
        elif max_box_height >= MAX_BOX_HEIGHT_PX:
            vision_pwm = 255
        else:
            frac = (max_box_height - MIN_BOX_HEIGHT_PX) / (MAX_BOX_HEIGHT_PX - MIN_BOX_HEIGHT_PX)
            vision_pwm = int(frac * 255)

        # Hardware IR Sensor PWM logic
        if latest_ir_distance >= MAX_DIST_CM:
            ir_pwm = 0
        elif latest_ir_distance <= MIN_DIST_CM:
            ir_pwm = 255
        else:
            frac = (MAX_DIST_CM - latest_ir_distance) / (MAX_DIST_CM - MIN_DIST_CM)
            ir_pwm = int(frac * 255)

        # Merge PWM requirements (highest intensity wins)
        final_pwm = max(vision_pwm, ir_pwm)
        motor_cmd = f"M{final_pwm}"

        # Voice Guidance Alerts when an obstacle is nearby
        if max_box_height > 120 or latest_ir_distance < 35.0:
            if closest_pos == "straight ahead":
                speak("Obstacle straight ahead. Rotate right.", priority_level=0, cooldown_sec=1.5)
                motor_cmd = "R"
            elif closest_pos == "left":
                speak("Obstacle on left. Turn right.", priority_level=1, cooldown_sec=2.0)
            elif closest_pos == "right":
                speak("Obstacle on right. Turn left.", priority_level=1, cooldown_sec=2.0)

        # FIX 3: Terminal Debugging Output
        print(f"[DEBUG] Box: {max_box_height}px | Vision PWM: {vision_pwm} | IR: {latest_ir_distance:.1f}cm | IR PWM: {ir_pwm} | Sent: {motor_cmd}")

        # Send finalized motor command to Arduino
        send_motor_command(motor_cmd)

        # Visual Telemetry Overlay
        cv2.putText(frame, f"IR Distance: {latest_ir_distance:.1f} cm", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Max Box Height: {max_box_height} px", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Motor Speed: {motor_cmd} (PWM: {final_pwm})", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Smart Assistive Stick - Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            guidance_thread.running = False
            send_motor_command("M0")
            break

    if arduino and arduino.is_open:
        arduino.close()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()