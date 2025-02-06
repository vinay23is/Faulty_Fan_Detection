import machine
import time
import network
from umqtt.simple import MQTTClient
import esp32
import numpy as np
import tensorflow.lite as tflite

# WiFi Credentials
SSID = "your_SSID"
PASSWORD = "your_PASSWORD"
BROKER = "your_mqtt_broker"
TOPIC = "fan/detection"

# Load ML Model
interpreter = tflite.Interpreter(model_path="model/fan_fault_detection.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define Microphone Input
mic = machine.ADC(machine.Pin(36))
mic.atten(machine.ADC.ATTN_11DB)

# Connect to WiFi
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(SSID, PASSWORD)
    while not wlan.isconnected():
        time.sleep(1)
    print("Connected to WiFi")

# Initialize MQTT Client
client = MQTTClient("esp32_client", BROKER)

# Fan Fault Detection Logic
def detect_fault():
    while True:
        sound_level = mic.read()
        input_data = np.array([[sound_level]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        if prediction[0] > 0.5:
            client.publish(TOPIC, "Fault Detected")
            print("Anomalous fan behavior detected!")
        else:
            client.publish(TOPIC, "Fan Running Smoothly")
        
        time.sleep(2)

# Main Loop
connect_wifi()
client.connect()
detect_fault()
