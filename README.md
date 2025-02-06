# Faulty Fan Detection System (Embedded ML & IoT)

## Overview
This project detects anomalies in fan sounds using an **ESP32 + INMP441** microphone and **TensorFlow Lite**.
If an anomaly is detected, an alert is sent via **MQTT**.

## Features
- **Continuous Sound Monitoring** 🎤
- **ML-Based Anomaly Detection** 🔍
- **ESP32 MQTT Integration** 🌐
- **Edge AI using TensorFlow Lite** 🤖

## Repository Structure
```
- /firmware          # ESP32 Code
- /model             # ML Training and TFLite Model
- /data/normal       # Placeholder for normal fan sound data
- /data/anomalous    # Placeholder for anomalous fan sound data
- README.md          # Documentation
```

## How to Use
1. **Train the Model**: Run `model/train_model.py`
2. **Convert Model to TensorFlow Lite**: Run `model/convert_model.py`
3. **Deploy Firmware to ESP32**: Flash the ESP32 with `firmware/main.py`
4. **Monitor Fan Status**: Connect ESP32 to MQTT and receive alerts.
