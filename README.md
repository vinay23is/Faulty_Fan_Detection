# Faulty Fan Detection System (Embedded ML & IoT)

## Project Overview
The **Faulty Fan Detection System** is an **embedded machine learning (ML) and IoT**-based project designed to monitor and detect anomalies in fan operation. It utilizes an **ESP32 microcontroller** and an **INMP441 microphone** to continuously listen to fan sounds. The system processes the audio data, detects anomalies using a trained **TensorFlow Lite** model, and sends alerts via **MQTT** if abnormal behavior is detected.

## Key Features
- 📡 **Real-time Fan Monitoring** using ESP32
- 🎤 **Continuous Sound Analysis** via INMP441 Microphone
- 🔍 **ML-Based Anomaly Detection** using TensorFlow Lite
- 🌐 **MQTT Integration** for Remote Notifications
- 🚀 **Edge AI Deployment** on Embedded Devices

## Model Architecture
```
Model: "sequential_1"
_________________________________
 Layer (type)      Output Shape     Param #  
=================================
 dense_3 (Dense)   (None, 256)      36864   
 dropout (Dropout) (None, 256)      0       
 dense_4 (Dense)   (None, 128)      32896   
 dropout_1 (Dropout) (None, 128)    0       
 dense_5 (Dense)   (None, 64)       8256    
 dense_6 (Dense)   (None, 1)        65      
=================================
Total params: 78,081
Trainable params: 78,081
Non-trainable params: 0
```

## Model Performance
### Classification Report
```
              precision    recall  f1-score   support
0             0.97        0.94     0.96       1345
1             0.94        0.96     0.95       1140
accuracy      0.95        0.95     0.95       2485
macro avg     0.95        0.95     0.95       2485
weighted avg  0.95        0.95     0.95       2485
```

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Precision-Recall Curve
![Precision-Recall Curve](precision_recall_curve.png)
Best Threshold for F1-Score: **0.2982**

## Dataset Summary
```
Number of Normal Files: 6522
Number of Anomaly Files: 1476
```

## Inference & Performance Metrics
```
Average Inference Latency: 0.08166 seconds
Total FLOPs: 155713
Total MACs: 77856.5
```

## How It Works
1. **Data Processing & Model Training**:
   - The model is trained using **MFCC features** extracted from audio recordings.
   - Class imbalance is handled using **SMOTEENN**.
2. **Model Deployment**:
   - Converted to **TensorFlow Lite** for ESP32 compatibility.
   - Loaded into ESP32 firmware for **on-device inference**.
3. **Continuous Sound Monitoring**:
   - The ESP32 continuously listens to fan noise.
   - If an anomaly is detected, an alert is sent via **MQTT**.

## Repository Structure
```
- /firmware          # ESP32 Code
- /model             # ML Training and TFLite Model
- /data/normal       # Placeholder for normal fan sound data
- /data/anomalous    # Placeholder for anomalous fan sound data
- README.md          # Documentation
```

## Getting Started
1. **Train the Model**: Run `model/train_model.py`
2. **Convert Model to TensorFlow Lite**: Run `model/convert_model.py`
3. **Deploy Firmware to ESP32**: Flash the ESP32 with `firmware/main.py`
4. **Monitor Fan Status**: Connect ESP32 to MQTT and receive alerts.

## Acknowledgments
- **TensorFlow Lite** for ML deployment on ESP32
- **Librosa** for audio feature extraction
- **MQTT** for real-time communication

## Contributors
- Vinay Dodla
  
