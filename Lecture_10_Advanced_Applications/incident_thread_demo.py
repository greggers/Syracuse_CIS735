import json
import datetime
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import networkx as nx
import matplotlib.pyplot as plt

@dataclass
class ThreadNode:
    """Represents a node in the digital thread"""
    id: str
    timestamp: str
    node_type: str
    data: Dict[str, Any]
    source: str
    parent_ids: List[str]

class DigitalThreadTracker:
    """
    Tracks the digital thread of a cybersecurity incident from detection to resolution
    """
    
    def __init__(self):
        self.thread_nodes = []
        self.graph = nx.DiGraph()
    
    def add_node(self, node_type: str, data: Dict[str, Any], 
                 source: str, parent_ids: List[str] = None):
        """Add a new node to the digital thread"""
        if parent_ids is None:
            parent_ids = []
            
        node = ThreadNode(
            id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now().isoformat(),
            node_type=node_type,
            data=data,
            source=source,
            parent_ids=parent_ids
        )
        
        self.thread_nodes.append(node)
        self.graph.add_node(node.id, **asdict(node))
        
        # Add edges from parent nodes
        for parent_id in parent_ids:
            self.graph.add_edge(parent_id, node.id)
        
        return node.id
    
    def simulate_incident_thread(self):
        """Simulate a complete cybersecurity incident digital thread"""
        
        # 1. Initial network anomaly detection
        anomaly_id = self.add_node(
            node_type="anomaly_detection",
            data={
                "alert_type": "unusual_network_traffic",
                "source_ip": "192.168.1.100",
                "destination_ip": "external_malicious_ip",
                "traffic_volume": "500MB",
                "confidence": 0.85
            },
            source="network_ids"
        )
        
        # 2. ML model prediction
        ml_prediction_id = self.add_node(
            node_type="ml_prediction",
            data={
                "model_type": "random_forest",
                "prediction": "malware_communication",
                "probability": 0.92,
                "features_used": ["packet_size", "frequency", "destination_reputation"]
            },
            source="ml_threat_detector",
            parent_ids=[anomaly_id]
        )
        
        # 3. Threat intelligence correlation
        threat_intel_id = self.add_node(
            node_type="threat_intelligence",
            data={
                "ioc_match": True,
                "threat_family": "APT29",
                "campaign": "CozyBear_2024",
                "confidence": "high"
            },
            source="threat_intel_platform",
            parent_ids=[ml_prediction_id]
        )
        
        # 4. Forensic analysis
        forensics_id = self.add_node(
            node_type="forensic_analysis",
            data={
                "artifacts_found": ["suspicious_registry_keys", "encrypted_payload"],
                "timeline": "infection_started_3_days_ago",
                "affected_systems": ["workstation_A", "server_B"]
            },
            source="forensic_analyst",
            parent_ids=[threat_intel_id]
        )
        
        # 5. Containment action
        containment_id = self.add_node(
            node_type="containment",
            data={
                "action": "network_isolation",
                "systems_isolated": ["192.168.1.100", "192.168.1.101"],
                "firewall_rules_updated": True
            },
            source="incident_response_team",
            parent_ids=[forensics_id]
        )
        
        # 6. Remediation
        remediation_id = self.add_node(
            node_type="remediation",
            data={
                "malware_removed": True,
                "systems_patched": ["workstation_A", "server_B"],
                "backup_restored": True,
                "verification_scan": "clean"
            },
            source="security_team",
            parent_ids=[containment_id]
        )
        
        return remediation_id
    
    def visualize_thread(self):
        """Visualize the digital thread as a graph"""
        plt.figure(figsize=(15, 10))
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Color nodes by type
        node_colors = {
            'anomaly_detection': 'red',
            'ml_prediction': 'blue',
            'threat_intelligence': 'orange',
            'forensic_analysis': 'green',
            'containment': 'purple',
            'remediation': 'gold'
        }
        
        colors = [node_colors.get(self.graph.nodes[node]['node_type'], 'gray') 
                 for node in self.graph.nodes()]
        
        # Draw the graph
        nx.draw(self.graph, pos, node_color=colors, node_size=1000, 
                with_labels=False, arrows=True, arrowsize=20, 
                edge_color='gray', alpha=0.7)
        
        # Add labels
        labels = {node: self.graph.nodes[node]['node_type'] 
                 for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Cybersecurity Incident Digital Thread")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def get_lineage(self, node_id: str):
        """Get the complete lineage of a specific node"""
        ancestors = list(nx.ancestors(self.graph, node_id))
        lineage = []
        
        for ancestor_id in ancestors + [node_id]:
            node_data = self.graph.nodes[ancestor_id]
            lineage.append({
                'id': ancestor_id,
                'type': node_data['node_type'],
                'timestamp': node_data['timestamp'],
                'source': node_data['source']
            })
        
        return sorted(lineage, key=lambda x: x['timestamp'])

# Demo execution
if __name__ == "__main__":
    tracker = DigitalThreadTracker()
    final_node_id = tracker.simulate_incident_thread()
    
    print("Digital Thread Simulation Complete!")
    print(f"Total nodes in thread: {len(tracker.thread_nodes)}")
    
    # Show lineage
    lineage = tracker.get_lineage(final_node_id)
    print("\nIncident Lineage:")
    for item in lineage:
        print(f"  {item['timestamp']}: {item['type']} from {item['source']}")
    
    # Visualize
    tracker.visualize_thread()