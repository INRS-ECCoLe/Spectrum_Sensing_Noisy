# **Deep Learning–Based Spectrum Sensing Using RML2016, RML2018, HisarMod2019, and PanoRadioHF Dataset**

## **Overview**

This project establishes a framework for robust, low-SNR, privacy-preserving spectrum sensing suitable for 6G cognitive radio and intelligent RF monitoring research. The methodology repurposes major radio modulation datasets for binary spectrum sensing tasks (Idle vs. Occupied) and evaluates various Deep Learning architectures.

## **Phase 1: Dataset Preparation and Problem Formulation**

* **Foundation Datasets:** \* RadioML (RML2016.10b and RML2018.01a)  
  * HisarMod2019  
  * PanoRadioHF  
* **Problem Definition:** The learning objective is redefined as a binary classification task:  
  * **H0:** Idle channel (Noise only)  
  * **H1:** Occupied channel (Signal \+ Noise)  
* **Data Augmentation:** AWGN-only samples are synthetically generated to represent idle channels (H0) to balance the datasets.

## **Phase 2: Robustness Analysis**

To ensure reliability in real-world conditions, the models are rigorously evaluated under:

* **Fading:** Signal strength variations.  
* **CFO (Carrier Frequency Offset):** Frequency synchronization errors.  
* **Channel Impairments:** Various realistic channel distortions.  
* **Generalization:** Analysis of model performance across unseen SNR levels and channel conditions.

## **Phase 3: Feature Representation and Preprocessing**

Two distinct feature extraction pipelines are employed based on the model architecture:

1. **Raw IQ Sequences (128 × 2):** Used for sequence-based models (LSTM, PETCGDNN, GRU).  
2. **Time–Frequency Representations:** STFT/Spectrograms used for CNN-based models.  
* **Preprocessing:** All samples undergo **Power-Normalization** to ensure fairness across different SNR levels.

## **Phase 4: Deep Learning Models**

The framework evaluates four distinct deep learning architectures, each offering unique advantages:

* **CNN (Convolutional Neural Network):** Specialized in learning spectral occupancy patterns from time-frequency images.  
* **LSTM (Long Short-Term Memory):** Designed to capture long-term temporal dependencies in IQ sequences.  
* **GRU (Gated Recurrent Unit):** Offers efficient temporal modeling with lower computational cost than LSTM.  
* **PETCGDNN:** A hybrid architecture specifically engineered for robustness in low-SNR environments.

*All models output a probability score indicating the presence of a signal.*

## **Phase 5: Performance Evaluation**

Models are benchmarked against classical detectors (Energy Detection, Cyclostationary Feature Detection) using the following metrics:

* **Classification Metrics:** Precision, F1 Score, Accuracy.  
* **Sensing Metrics:** \* **Pd:** Probability of Detection.  
  * **Pfa:** Probability of False Alarm.  
  * **ROC:** Receiver Operating Characteristic curves.  
* **Regression Metrics:** NMSE (Normalized Mean Squared Error).

## **Final Outcome**

This comprehensive framework demonstrates a deep learning-based approach to spectrum sensing that is resilient to low-SNR conditions and channel impairments, paving the way for advanced **6G cognitive radio** and **intelligent RF monitoring** applications.