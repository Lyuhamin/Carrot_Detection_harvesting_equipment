#include <Servo.h>

Servo gate;
const int SERVO_PIN = 9;
const int RELAY_PIN = 7;  // 선택: 릴레이라면 HIGH/LOW

// 서보 각도
const int CLOSE_ANGLE = 0;
const int OPEN_ANGLE  = 60;

String buf;

void setup() {
  Serial.begin(115200);
  gate.attach(SERVO_PIN);
  gate.write(CLOSE_ANGLE);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW); // 릴레이 OFF
}

void triggerCut() {
  // 서보 열기
  gate.write(OPEN_ANGLE);
  delay(250); // 기구에 맞춰 가감
  // 릴레이 ON (옵션)
  digitalWrite(RELAY_PIN, HIGH);
  delay(150); 
  digitalWrite(RELAY_PIN, LOW);
  // 서보 닫기
  gate.write(CLOSE_ANGLE);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      if (buf == "CUT") {
        triggerCut();
        Serial.println("OK");
      } else if (buf == "IDLE") {
        // 필요시 대기 로직
      }
      buf = "";
    } else if (c != '\r') {
      buf += c;
    }
  }
}
