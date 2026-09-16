// ==========================================
// SMART STICK - ARDUINO HARDWARE SKETCH
// ==========================================
const int IR_PIN = A2;          // Sharp IR Analog Input
const int MOTOR_PWM_PIN = 9;    // Motor / Haptic Driver Pin (PWM)

// Active Detection Thresholds (in Centimeters)
const float MIN_DISTANCE_CM = 10.0; // Urgent threshold -> Full Speed
const float MAX_DISTANCE_CM = 50.0; // Outer threshold -> Stops

unsigned long lastIRTime = 0;
unsigned long lastSerialTime = 0;
char lastYoloCmd = '0'; // Default '0' = Clear

float currentDistance = 999.0;

void setup() {
  Serial.begin(9600);
  pinMode(MOTOR_PWM_PIN, OUTPUT);
  analogWrite(MOTOR_PWM_PIN, 0); // Force Motor OFF at boot
}

float getDistanceCM() {
  long sum = 0;
  for (int i = 0; i < 5; i++) {
    sum += analogRead(IR_PIN);
  }
  int rawValue = sum / 5;

  // Sharp GP2Y0A21 Range Cutoffs:
  // Raw values < 90 correspond to open space (> 80cm)
  // Raw values > 650 indicate electrical shorts or invalid close range
  if (rawValue < 90 || rawValue > 650) {
    return 999.0;
  }

  float volts = rawValue * (5.0 / 1023.0);
  float distanceCm = 27.61 * pow(volts, -1.173);

  return distanceCm;
}

void loop() {
  // 1. Read IR Sensor & Transmit Clean Telemetry (Every 80ms)
  if (millis() - lastIRTime > 80) {
    currentDistance = getDistanceCM();

    // Clean transmission format for app.py
    Serial.print("IR_CM:");
    Serial.println(currentDistance);

    lastIRTime = millis();
  }

  // 2. Read Serial Control Commands from Python
  while (Serial.available() > 0) {
    lastYoloCmd = Serial.read();
    lastSerialTime = millis(); // Reset watchdog timer
  }

  // Watchdog: Reset YOLO state if Python disconnects or drops for 500ms
  if (millis() - lastSerialTime > 500) {
    lastYoloCmd = '0';
  }

  // 3. Priority Control Matrix
  int motorSpeed = 0;

  if (lastYoloCmd == '2') {
    // Priority 1: YOLO Vehicle Override (Maximum Vibration)
    motorSpeed = 255;
  } 
  else if (currentDistance >= MIN_DISTANCE_CM && currentDistance <= MAX_DISTANCE_CM) {
    // Priority 2: IR Physical Proximity (Proportional speed)
    long mapped = map((long)currentDistance, (long)MAX_DISTANCE_CM, (long)MIN_DISTANCE_CM, 120, 255);
    motorSpeed = constrain(mapped, 120, 255);
  } 
  else if (lastYoloCmd == '1') {
    // Priority 3: YOLO General Object (Medium Vibration)
    motorSpeed = 180;
  } 
  else {
    // Priority 4: Safe / No Detection
    motorSpeed = 0;
  }

  analogWrite(MOTOR_PWM_PIN, motorSpeed);
}
