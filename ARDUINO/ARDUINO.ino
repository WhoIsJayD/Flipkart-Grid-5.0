#include "ServoEasing.hpp"
#include "StringSplitter.h"
#include <EEPROM.h>

// Servo objects
ServoEasing S1;
ServoEasing S3;
ServoEasing S4;
ServoEasing S5;
ServoEasing S6;
const int relay1 = 7;
const int relay2 = 6;
const int relay3 = 5;
const int relay4 = 4;
const int a = 3;
const int b = 2;
const int pwm = 11;
const int dir = 8;
const int m1 = A0;
const int m3 = A3;
const int m4 = A1;
const int m5 = A2;
const int m6 = A4;
int lastTarget1 = 0;
volatile int lastpos, pos = 0;
float target[7] = {};

enum State {
  IDLE,
  MOVING
};
State currentState = State::IDLE;

void encoder() {
  pos += digitalRead(b) > 0 ? -1 : 1;
}

void setup() {
  Serial.begin(115200);
 
  // analogWrite(pwm,50);

  pinMode(a, INPUT);
  pinMode(b, INPUT);
  pinMode(dir, OUTPUT);
  pinMode(relay1, OUTPUT);
  pinMode(relay2, OUTPUT);
  pinMode(relay3, OUTPUT);
  pinMode(relay4, OUTPUT);

  // Servo settings
  S1.attach(m1, 0);
  S3.attach(m3, 0);
  S4.attach(m4, 0);
  S5.attach(m5, 0);
  S6.attach(m6, 0);

  S1.setEasingType(EASE_CUBIC_IN_OUT);
  S3.setEasingType(EASE_CUBIC_IN_OUT);
  S4.setEasingType(EASE_CUBIC_IN_OUT);
  S5.setEasingType(EASE_CUBIC_IN_OUT);
  S6.setEasingType(EASE_CUBIC_IN_OUT);
  S1.easeTo(0, 15);
  S3.easeTo(0, 15);
  S4.easeTo(0, 15);
  S5.easeTo(0, 15);
  S6.easeTo(0, 15);
  attachInterrupt(digitalPinToInterrupt(a), encoder, RISING);
}

void loop() {

  analogWrite(relay1, 255);
  analogWrite(relay2, 255);
  analogWrite(relay3, 255);
  analogWrite(relay4, 255);

  if (currentState == State::IDLE) {
    waitForInput();
  }

  if (currentState == State::MOVING) {
    moveMotors();
  }
}

void waitForInput() {
  if (Serial.available() > 0) {

    String input_string = Serial.readStringUntil('\n');

    Serial.print("Received input: ");
    Serial.println(input_string);

    if (input_string.startsWith("MOTOR,")) {
      Serial.println(input_string.substring(6));
      StringSplitter splitter(input_string.substring(6), ',', 7);
      int itemCount = splitter.getItemCount();
      Serial.println(itemCount);

      if (itemCount == 7) {
        for (int i = 0; i < itemCount; i++) {
          float item = splitter.getItemAtIndex(i).toFloat();
          target[i] = item;
        }

        Serial.println("Moving to the target positions...");

        // Debugging the values of target
        for (int i = 0; i < 7; i++) {
          Serial.print("target[");
          Serial.print(i);
          Serial.print("] = ");
          Serial.println(target[i]);
        }

        target[1] = map(target[1], 0, 360, 0, 8192);
        currentState = State::MOVING;
      } else {
        Serial.println("Invalid number of target positions.");
      }
    }
  }
}

void moveMotors() {
  S1.easeTo(target[0], 20);
  S3.easeTo(target[2], 20);
  S4.easeTo(target[3], 20);
  S5.easeTo(target[4], 20);
  S6.easeTo(target[5], 20);

  int e = target[1] - pos;
  if (e >= 135) {
    digitalWrite(dir, LOW);
    analogWrite(pwm, 20);
  } else if (e < 0) {
    digitalWrite(dir, HIGH);
    analogWrite(pwm, 20);
  } else {

    

      if (lastTarget1 < target[1]) {
        Serial.println("previous value is lower, set direction HIGH");
        analogWrite(pwm, 5);
        digitalWrite(dir, HIGH);
      } else if (lastTarget1 > target[1]) {
        Serial.println("previous value is higher, set direction LOW");
        analogWrite(pwm, 5);
        digitalWrite(dir, HIGH);
      } else {
        Serial.println("0");
        analogWrite(pwm, 0);
      }
    
    lastTarget1 = target[1];



    currentState = State::IDLE;
    Serial.println("Reached target positions.");
    lastpos = pos;
    EEPROM.update(25, lastpos / 32.1254902);
    if (target[6] == 1) {
      Serial.println("Suction Stopped !");
      analogWrite(relay1, 0);
      analogWrite(relay2, 0);
      analogWrite(relay3, 0);
      analogWrite(relay4, 0);
    } else {
      Serial.println("Suction As it is !");
      analogWrite(relay1, 255);
      analogWrite(relay2, 255);
      analogWrite(relay3, 255);
      analogWrite(relay4, 255);
    }

    Serial.println("DONE");
  }
}
