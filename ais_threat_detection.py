import os
import csv
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import tkinter as tk
from tkinter import ttk
import psutil
import time
from tqdm import tqdm

# Define paths
processed_dir = r'E:\AIS\AI_PROJECT\processed_data'
log_file = r'E:\AIS\AI_PROJECT\threats.log'

# Load data from CSV files
def load_data(filename, num_samples=None, is_threat=False, shuffle=False):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found. Ensure it exists in the specified directory.")
    patterns = []
    labels = []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header
            if header != ['src_bytes', 'dst_bytes', 'rate', 'label']:
                raise ValueError(f"Invalid header in {filename}. Expected ['src_bytes', 'dst_bytes', 'rate', 'label'], got {header}")
            data = list(reader)
            if shuffle:
                random.shuffle(data)  # Shuffle the data if requested
            for row in data:
                pattern = [float(row[0]), float(row[1]), float(row[2])]
                label = int(row[3])
                patterns.append(pattern)
                labels.append(label)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error reading {filename}: {e}. Ensure the file is in the correct format (src_bytes,dst_bytes,rate,label).")
    if num_samples is not None and len(patterns) > num_samples:
        patterns = patterns[:num_samples]
        labels = labels[:num_samples]
    return patterns, labels

# Detector class for AIS
class Detector:
    def __init__(self, features, radius):
        self.features = features
        self.radius = radius

    def matches(self, pattern):
        distance = np.sqrt(sum((f - p) ** 2 for f, p in zip(self.features, pattern)))
        return distance <= self.radius

# AIS class for a specific threat type
class AIS:
    def __init__(self, threat_type, num_detectors=10, radius=0.5):  # Increased radius
        self.threat_type = threat_type
        self.detectors = []
        self.num_detectors = num_detectors
        self.radius = radius

    def train(self, self_patterns, threat_patterns):
        print(f"Training AIS for {self.threat_type}...")
        # Use threat patterns to guide detector placement
        threat_centroid = np.mean(threat_patterns, axis=0) if threat_patterns else np.array([1.0, 1.0, 1.0])
        while len(self.detectors) < self.num_detectors:
            # Generate a detector near the threat centroid with some randomness
            detector_features = [
                threat_centroid[i] + random.uniform(-0.3, 0.3) for i in range(3)
            ]
            # Ensure detector features are within the valid range [0, 2]
            detector_features = [max(0, min(2, f)) for f in detector_features]
            detector = Detector(detector_features, self.radius)
            
            # Check if detector matches any self pattern
            if not any(detector.matches(self_p) for self_p in self_patterns):
                # Check if detector matches the threat pattern
                if any(detector.matches(threat_p) for threat_p in threat_patterns):
                    self.detectors.append(detector)
        print(f"Generated {len(self.detectors)} detectors for {self.threat_type}.")

    def detect(self, pattern):
        return any(detector.matches(pattern) for detector in self.detectors)

# Multi-AIS class to handle multiple threat types
class MultiAIS:
    def __init__(self, threats):
        self.ais_dict = {threat: AIS(threat) for threat in threats}

    def train(self, self_patterns, threat_patterns_dict):
        for threat, ais in self.ais_dict.items():
            ais.train(self_patterns, threat_patterns_dict[threat])

    def detect(self, pattern):
        for threat, ais in self.ais_dict.items():
            if ais.detect(pattern):
                return threat
        return None

# Visualize the AIS detectors and patterns in 3D
def visualize_ais(multi_ais, self_patterns, test_patterns):
    # Reduce the number of test patterns for visualization
    if len(test_patterns) > 1000:
        test_patterns = random.sample(test_patterns, 1000)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')  # 3D plot
    
    # Plot self patterns
    self_x = [p[0] for p in self_patterns]
    self_y = [p[1] for p in self_patterns]
    self_z = [p[2] for p in self_patterns]
    ax.scatter(self_x, self_y, self_z, c='green', label='Self Patterns')
    
    # Plot detectors
    for threat, ais in multi_ais.ais_dict.items():
        det_x = [d.features[0] for d in ais.detectors]
        det_y = [d.features[1] for d in ais.detectors]
        det_z = [d.features[2] for d in ais.detectors]
        ax.scatter(det_x, det_y, det_z, label=f'{threat} Detectors')
    
    # Plot test patterns
    test_x = [p[0] for p in test_patterns]
    test_y = [p[1] for p in test_patterns]
    test_z = [p[2] for p in test_patterns]
    ax.scatter(test_x, test_y, test_z, c='red', label='Test Patterns', alpha=0.5)
    
    ax.set_xlabel('src_bytes')
    ax.set_ylabel('dst_bytes')
    ax.set_zlabel('rate')
    plt.legend()
    plt.title("AIS Threat Detection Visualization (3D)")
    plt.show()

# Train a classifier for comparison
def train_classifier(self_patterns, threat_patterns_dict):
    print("Training classifier...")
    # Balance the training data
    num_self = len(self_patterns)
    threat_patterns = []
    for patterns in threat_patterns_dict.values():
        threat_patterns.extend(patterns)
    num_threat = len(threat_patterns)
    min_samples = min(num_self, num_threat)
    
    # Sample equal numbers of self and threat patterns
    self_samples = self_patterns[:min_samples]
    threat_samples = threat_patterns[:min_samples]
    
    X = self_samples + threat_samples
    y = [0] * len(self_samples) + [1] * len(threat_samples)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

# Evaluate the AIS on the test set
def evaluate_ais(multi_ais, clf, test_patterns, test_labels):
    print("Evaluating AIS on test set...")
    predictions = []
    for pattern in tqdm(test_patterns, desc="Evaluating"):
        threat = multi_ais.detect(pattern)
        if threat:
            predictions.append(1)  # Threat detected
        else:
            predictions.append(clf.predict([pattern])[0])  # Fallback to classifier
    print(f"Predictions: {predictions[:10]}...")
    print(f"True Labels: {test_labels[:10]}...")
    # Calculate metrics with zero_division set to 0
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

# Real-time monitoring GUI
class ThreatDetectionGUI:
    def __init__(self, root, multi_ais, clf):
        self.root = root
        self.multi_ais = multi_ais
        self.clf = clf
        self.root.title("Real-Time Threat Detection")
        
        self.label = ttk.Label(root, text="Monitoring System Metrics...")
        self.label.pack(pady=10)
        
        self.text = tk.Text(root, height=10, width=50)
        self.text.pack(pady=10)
        
        self.monitoring = True
        self.update_metrics()

    def update_metrics(self):
        if not self.monitoring:
            return
        
        # Simulate system metrics (in a real scenario, these would be actual system metrics)
        cpu = psutil.cpu_percent(interval=1) / 100  # Normalize to [0, 1]
        memory = psutil.virtual_memory().percent / 100  # Normalize to [0, 1]
        network = psutil.net_io_counters().bytes_sent / 1e6  # Normalize (example)
        network = min(network / 10, 1)  # Rough normalization to [0, 1]
        
        pattern = [cpu, memory, network]
        
        # Detect threats
        threat = self.multi_ais.detect(pattern)
        if threat:
            message = f"Threat Detected: {threat} at {time.ctime()}\n"
            self.text.insert(tk.END, message)
            with open(log_file, 'a') as f:
                f.write(message)
        else:
            prediction = self.clf.predict([pattern])[0]
            if prediction == 1:
                message = f"Potential Threat Detected (Classifier) at {time.ctime()}\n"
                self.text.insert(tk.END, message)
                with open(log_file, 'a') as f:
                    f.write(message)
        
        self.text.insert(tk.END, f"CPU: {cpu:.2f}, Memory: {memory:.2f}, Network: {network:.2f}\n")
        self.text.see(tk.END)
        self.root.after(1000, self.update_metrics)  # Update every second

    def stop(self):
        self.monitoring = False

# Main execution
def main():
    try:
        # Load data
        self_patterns, _ = load_data(os.path.join(processed_dir, 'normal_data.csv'), num_samples=1000)
        threat_patterns_dict = {
            'Malware': load_data(os.path.join(processed_dir, 'threat_malware.csv'), num_samples=500, is_threat=True)[0],
            'DDoS': load_data(os.path.join(processed_dir, 'threat_ddos.csv'), num_samples=500, is_threat=True)[0]
        }
        # Load test data with shuffling to ensure a mix of normal and attack samples
        test_patterns, test_labels = load_data(os.path.join(processed_dir, 'test_data.csv'), num_samples=1000, shuffle=True)

        # Train Multi-AIS
        threats = ['Malware', 'DDoS']
        multi_ais = MultiAIS(threats)
        multi_ais.train(self_patterns, threat_patterns_dict)

        # Train classifier
        clf = train_classifier(self_patterns, threat_patterns_dict)

        # Evaluate
        evaluate_ais(multi_ais, clf, test_patterns, test_labels)

        # Visualize
        visualize_ais(multi_ais, self_patterns, test_patterns)

        # Start real-time monitoring
        root = tk.Tk()
        app = ThreatDetectionGUI(root, multi_ais, clf)
        root.protocol("WM_DELETE_WINDOW", app.stop)
        root.mainloop()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()