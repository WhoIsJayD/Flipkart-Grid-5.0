# arduino_communication.py

import serial
import time

class ArduinoComm:

    def __init__(self, port='COM9', baud_rate=115200):
        self.ser = serial.Serial(port, baud_rate)
        time.sleep(2)

    def write_data(self, data):
        if not isinstance(data, str):
            data = str(data)  # Convert data to string if it's not already

        print("Writing data:", data)
        self.ser.write(data.encode("utf-8") + b'\n')
        print("Data written!")

    def read_data(self):
        print("Data reading started!")
        data = self.ser.readline().decode("latin1", errors="replace").strip()
        print("Read data:", data)
        while data != "DONE":
            data = self.ser.readline().decode("latin1", errors="replace").strip()
            print("Read data:", data)
        return data

    def close(self):
        self.ser.close()
