

# Flipkart Grid 5.0 - Robotics Challenge at IIT Madras

## Overview

This repository contains the code developed for the Flipkart Grid 5.0 competition, held at IIT Madras, where our team was selected for the finals. The project involves robotic manipulation and object detection using a combination of computer vision, inverse kinematics, and motor control.

## Project Description

The project showcases a robot capable of picking up objects, processing them, and placing them in designated zones. The robot uses multiple camera feeds for object detection, and the inverse kinematics calculations enable precise motor control for object manipulation.

## Features

- **Inverse Kinematics**: Calculates the angles for robotic arm joints to reach a specific point in 3D space.
- **Object Detection**: Utilizes custom-trained models for detecting and localizing objects.
- **Motor Control**: Communicates with an Arduino to control the robotic arm's motors.
- **Multi-Threading**: Manages multiple camera feeds concurrently for real-time object detection.

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Queue
- Arduino communication library
- ESP32 camera module

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/WhoIsJayD/Flipkart-Grid-5.0
   cd Flipkart-Grid-5.0
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the Arduino and ESP32 camera modules are connected and configured correctly.

## Usage

1. **Inverse Kinematics Calculation**:
   The `inverse_kinematics` function calculates the necessary angles for the robotic arm joints to reach a specific point (x, y, z).

2. **Object Detection**:
   The `perform_object_detection` function initiates object detection using multiple camera feeds. It runs for a specified duration or until a QR code is detected.

3. **Motor Control**:
   The `write_motor_poses` function sends calculated motor positions to the Arduino and ensures the command is executed.

4. **Main Loop**:
   The `main` function orchestrates the robot's state machine, transitioning through various states like initializing, picking up objects, and placing them in designated zones.

To run the project, execute the main script:

```bash
python main.py
```

## State Machine

The robot operates in several states:
- **INITIALIZING**: Sets the robot to the initial pose.
- **PICKING_UP**: Moves the robotic arm to pick up an object.
- **PROCESSING_PAUSED**: Pauses processing to analyze detected objects.
- **PICKED_UP**: Confirms the object is picked up and calculates new poses.
- **PICK_UP_PICKED**: Adjusts the arm for object placement.
- **DROP_ZONE_PICKED**: Moves the arm to the drop zone.
- **ESP_POSE_PICKED**: Positions the arm for ESP32 camera detection.

## License and Permissions

This code is made publicly available for reference purposes in my resume. You are free to refer to it, but you may not use or modify the code without explicit permission.

## Contact

For any inquiries or permissions, please contact:

- **Name**: Jaydeep Solanki
- **Email**: jaydeep.solankee@yahoo.com
- **LinkedIn**: https://www.linkedin.com/in/jaydeep-solanki-79ab61253/

## Team Members

- **[Jaydeep Solanki](https://www.linkedin.com/in/jaydeep-solanki-79ab61253/)** (EC, Nirma University)
- **[Vivek Samani](https://www.linkedin.com/in/vivek-samani-0957a127a/)** (EC, Nirma University)
- **[Satyam Rana](https://www.linkedin.com/in/satyam-rana-690692256/)** (EC, Nirma University)
- **[Sneh Shah](https://www.linkedin.com/in/sneh-shah-b8177828a/)** (EC, Nirma University)
- **[Ameya Kale](https://www.linkedin.com/in/ameya-kale-5228a8257/)** (EI, Nirma University)
- **[Keshav Vyas](https://www.linkedin.com/in/keshav-vyas-b194b4259/)** (EE, Nirma University)

We are students from [Institute of Technology, Nirma University, Ahmedabad](https://nirmauni.ac.in).

## Acknowledgments

We would like to thank Flipkart for organizing the Flipkart Grid 5.0 competition and providing this opportunity to showcase our skills.
