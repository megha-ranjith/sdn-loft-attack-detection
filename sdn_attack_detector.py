"""
SDN Low-Rate Flow Table Overflow (LOFT) Attack Detection System
Using Machine Learning
Author: MTech CSE Student - Micro Project
Subject: Advanced Computer Networks 
"""

import numpy as np
import pandas as pd
from datetime import datetime
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class SDNFlowTableSimulator:
    """Simulates an SDN switch with flow table"""
    
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.flow_table = []
        self.total_flows = 0
        
    def add_flow(self, flow):
        """Add flow entry to the table"""
        if len(self.flow_table) < self.capacity:
            self.flow_table.append(flow)
            self.total_flows += 1
            return True
        return False
    
    def get_occupancy(self):
        """Get current flow table occupancy percentage"""
        return (len(self.flow_table) / self.capacity) * 100
    
    def remove_flow(self, index):
        """Remove flow entry from table"""
        if 0 <= index < len(self.flow_table):
            self.flow_table.pop(index)
            return True
        return False
    
    def clear_malicious_flows(self, malicious_indices):
        """Remove all detected malicious flows"""
        for idx in sorted(malicious_indices, reverse=True):
            self.remove_flow(idx)

class FlowGenerator:
    """Generate normal and attack traffic flows"""
    
    @staticmethod
    def generate_normal_flow():
        """Generate characteristics of normal network flow"""
        return {
            'packet_count': random.randint(50, 500),
            'byte_count': random.randint(5000, 50000),
            'duration': random.uniform(1.0, 30.0),
            'src_port': random.randint(1024, 65535),
            'dst_port': random.choice([80, 443, 22, 21, 25]),
            'protocol': random.choice([6, 17]),  # TCP or UDP
            'flow_type': 'normal'
        }
    
    @staticmethod
    def generate_attack_flow():
        """Generate characteristics of LOFT attack flow"""
        return {
            'packet_count': random.randint(1, 10),  # Low packet count
            'byte_count': random.randint(100, 1000),  # Low byte count
            'duration': random.uniform(0.01, 0.5),  # Very short duration
            'src_port': random.randint(1024, 65535),
            'dst_port': random.randint(1024, 65535),  # Random ports
            'protocol': random.choice([6, 17]),
            'flow_type': 'attack'
        }

class LOFTDetector:
    """Machine Learning based LOFT attack detector"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_columns = ['packet_count', 'byte_count', 'duration', 
                                'src_port', 'dst_port', 'protocol']
        
    def prepare_dataset(self, num_samples=1000):
        """Generate training dataset"""
        print(f"{Colors.OKCYAN}[INFO] Generating training dataset...{Colors.ENDC}")
        
        flows = []
        labels = []
        
        # Generate 70% normal flows and 30% attack flows
        for _ in range(int(num_samples * 0.7)):
            flow = FlowGenerator.generate_normal_flow()
            flows.append([flow[col] for col in self.feature_columns])
            labels.append(0)  # Normal = 0
            
        for _ in range(int(num_samples * 0.3)):
            flow = FlowGenerator.generate_attack_flow()
            flows.append([flow[col] for col in self.feature_columns])
            labels.append(1)  # Attack = 1
        
        return np.array(flows), np.array(labels)
    
    def train(self):
        """Train the ML model"""
        print(f"{Colors.OKBLUE}[TRAINING] Training Machine Learning Model...{Colors.ENDC}")
        
        X, y = self.prepare_dataset(num_samples=2000)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"{Colors.OKGREEN}[SUCCESS] Model Training Complete!{Colors.ENDC}")
        print(f"  ├─ Accuracy:  {accuracy*100:.2f}%")
        print(f"  ├─ Precision: {precision*100:.2f}%")
        print(f"  ├─ Recall:    {recall*100:.2f}%")
        print(f"  └─ F1-Score:  {f1*100:.2f}%")
        
        self.is_trained = True
        return accuracy, precision, recall, f1
    
    def predict(self, flows):
        """Predict if flows are normal or malicious"""
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        
        features = []
        for flow in flows:
            features.append([flow[col] for col in self.feature_columns])
        
        predictions = self.model.predict(np.array(features))
        return predictions

class SDNController:
    """SDN Controller with ML-based attack detection"""
    
    def __init__(self):
        self.switch = SDNFlowTableSimulator(capacity=100)
        self.detector = LOFTDetector()
        self.stats = {
            'total_flows': 0,
            'normal_flows': 0,
            'attack_flows': 0,
            'detected_attacks': 0,
            'blocked_attacks': 0
        }
    
    def initialize(self):
        """Initialize and train the detection system"""
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}  SDN LOFT Attack Detection System  {Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        
        print(f"{Colors.OKCYAN}[INIT] Initializing SDN Controller...{Colors.ENDC}")
        print(f"[INIT] Flow Table Capacity: {self.switch.capacity} entries")
        
        # Train the ML model
        accuracy, precision, recall, f1 = self.detector.train()
        
        print(f"\n{Colors.OKGREEN}[READY] System is ready for attack detection!{Colors.ENDC}\n")
        return accuracy, precision, recall, f1
    
    def simulate_traffic(self, duration=30, attack_probability=0.3):
        """Simulate network traffic with potential attacks"""
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Starting Traffic Simulation (Duration: {duration}s){Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            iteration += 1
            
            # Generate traffic (mix of normal and attack)
            if random.random() < attack_probability:
                flow = FlowGenerator.generate_attack_flow()
                self.stats['attack_flows'] += 1
            else:
                flow = FlowGenerator.generate_normal_flow()
                self.stats['normal_flows'] += 1
            
            self.stats['total_flows'] += 1
            
            # Add to flow table
            added = self.switch.add_flow(flow)
            
            # Periodically check for attacks
            if iteration % 5 == 0:
                self.detect_and_mitigate()
            
            # Display status every 3 seconds
            if iteration % 10 == 0:
                self.display_status()
            
            time.sleep(0.3)  # Simulate time between flows
        
        print(f"\n{Colors.OKGREEN}[COMPLETE] Traffic simulation finished!{Colors.ENDC}\n")
    
    def detect_and_mitigate(self):
        """Detect attacks in flow table and mitigate"""
        if len(self.switch.flow_table) == 0:
            return
        
        # Predict which flows are malicious
        predictions = self.detector.predict(self.switch.flow_table)
        
        malicious_indices = [i for i, pred in enumerate(predictions) if pred == 1]
        
        if len(malicious_indices) > 0:
            self.stats['detected_attacks'] += len(malicious_indices)
            self.stats['blocked_attacks'] += len(malicious_indices)
            
            print(f"{Colors.WARNING}[ALERT] Detected {len(malicious_indices)} malicious flows!{Colors.ENDC}")
            print(f"{Colors.FAIL}[ACTION] Removing malicious flows...{Colors.ENDC}")
            
            # Remove malicious flows
            self.switch.clear_malicious_flows(malicious_indices)
            
            print(f"{Colors.OKGREEN}[SUCCESS] Malicious flows removed!{Colors.ENDC}\n")
    
    def display_status(self):
        """Display current system status"""
        occupancy = self.switch.get_occupancy()
        
        print(f"{Colors.OKCYAN}{'─'*70}{Colors.ENDC}")
        print(f"Flow Table Occupancy: {occupancy:.1f}% ({len(self.switch.flow_table)}/{self.switch.capacity})")
        print(f"Total Flows Processed: {self.stats['total_flows']}")
        print(f"Normal: {self.stats['normal_flows']} | Attacks: {self.stats['attack_flows']}")
        print(f"Detected & Blocked: {self.stats['blocked_attacks']}")
        print(f"{Colors.OKCYAN}{'─'*70}{Colors.ENDC}\n")
    
    def display_final_report(self):
        """Display final statistics and report"""
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}  FINAL DETECTION REPORT  {Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        
        detection_rate = (self.stats['detected_attacks'] / max(self.stats['attack_flows'], 1)) * 100
        
        print(f"Total Flows Processed:     {self.stats['total_flows']}")
        print(f"├─ Normal Flows:           {self.stats['normal_flows']}")
        print(f"└─ Attack Flows:           {self.stats['attack_flows']}\n")
        
        print(f"Attack Detection:")
        print(f"├─ Attacks Detected:       {self.stats['detected_attacks']}")
        print(f"├─ Attacks Blocked:        {self.stats['blocked_attacks']}")
        print(f"└─ Detection Rate:         {detection_rate:.2f}%\n")
        
        print(f"Final Flow Table Occupancy: {self.switch.get_occupancy():.1f}%")
        
        if detection_rate > 80:
            print(f"\n{Colors.OKGREEN}✓ System Performance: EXCELLENT{Colors.ENDC}")
        elif detection_rate > 60:
            print(f"\n{Colors.WARNING}✓ System Performance: GOOD{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}✗ System Performance: NEEDS IMPROVEMENT{Colors.ENDC}")
        
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

def main():
    """Main execution function"""
    # Create SDN Controller
    controller = SDNController()
    
    # Initialize and train system
    controller.initialize()
    
    # Wait for user to start simulation
    input(f"{Colors.BOLD}Press ENTER to start traffic simulation...{Colors.ENDC}")
    
    # Simulate traffic with attacks
    controller.simulate_traffic(duration=30, attack_probability=0.3)
    
    # Display final report
    controller.display_final_report()
    
    print(f"{Colors.OKGREEN}[INFO] Project demonstration complete!{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Thank you for using SDN LOFT Attack Detection System{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
