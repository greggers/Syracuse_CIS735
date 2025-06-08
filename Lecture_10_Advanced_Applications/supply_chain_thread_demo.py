import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import networkx as nx

@dataclass
class SupplyChainComponent:
    """Represents a component in the software supply chain"""
    name: str
    version: str
    source: str
    hash_value: str
    dependencies: List[str]
    security_scan_results: Dict[str, any]
    timestamp: str

class SupplyChainDigitalThread:
    """
    Tracks digital thread for software supply chain security
    """
    
    def __init__(self):
        self.components = {}
        self.thread_events = []
        self.vulnerability_alerts = []
    
    def add_component(self, component: SupplyChainComponent):
        """Add a component to the supply chain thread"""
        self.components[f"{component.name}:{component.version}"] = component
        
        self.thread_events.append({
            'timestamp': component.timestamp,
            'event_type': 'component_added',
            'component': component.name,
            'version': component.version,
            'source': component.source,
            'hash': component.hash_value,
            'security_status': component.security_scan_results.get('status', 'unknown')
        })
    
    def simulate_supply_chain(self):
        """Simulate a complete software supply chain with security tracking"""
        
        # Base OS layer
        os_component = SupplyChainComponent(
            name="ubuntu",
            version="20.04",
            source="canonical_registry",
            hash_value=hashlib.sha256("ubuntu:20.04".encode()).hexdigest()[:16],
            dependencies=[],
            security_scan_results={
                'status': 'clean',
                'vulnerabilities': 0,
                'last_scan': datetime.now().isoformat()
            },
            timestamp=datetime.now().isoformat()
        )
        self.add_component(os_component)
        
        # Python runtime
        python_component = SupplyChainComponent(
            name="python",
            version="3.9.7",
            source="python_foundation",
            hash_value=hashlib.sha256("python:3.9.7".encode()).hexdigest()[:16],
            dependencies=["ubuntu:20.04"],
            security_scan_results={
                'status': 'clean',
                'vulnerabilities': 0,
                'last_scan': datetime.now().isoformat()
            },
            timestamp=(datetime.now() + timedelta(minutes=1)).isoformat()
        )
        self.add_component(python_component)
        
        # ML libraries
        sklearn_component = SupplyChainComponent(
            name="scikit-learn",
            version="1.0.2",
            source="pypi",
            hash_value=hashlib.sha256("scikit-learn:1.0.2".encode()).hexdigest()[:16],
            dependencies=["python:3.9.7"],
            security_scan_results={
                'status': 'clean',
                'vulnerabilities': 0,
                'last_scan': datetime.now().isoformat()
            },
            timestamp=(datetime.now() + timedelta(minutes=2)).isoformat()
        )
        self.add_component(sklearn_component)
        
        # Potentially vulnerable component
        requests_component = SupplyChainComponent(
            name="requests",
            version="2.25.1",
            source="pypi",
            hash_value=hashlib.sha256("requests:2.25.1".encode()).hexdigest()[:16],
            dependencies=["python:3.9.7"],
            security_scan_results={
                'status': 'vulnerable',
                'vulnerabilities': 1,
                'cve_ids': ['CVE-2023-32681'],
                'severity': 'medium',
                'last_scan': datetime.now().isoformat()
            },
            timestamp=(datetime.now() + timedelta(minutes=3)).isoformat()
        )
        self.add_component(requests_component)
        
        # Application layer
        app_component = SupplyChainComponent(
            name="malware_detector_app",
            version="1.0.0",
            source="internal_build",
            hash_value=hashlib.sha256("malware_detector_app:1.0.0".encode()).hexdigest()[:16],
            dependencies=["scikit-learn:1.0.2", "requests:2.25.1"],
            security_scan_results={
                'status': 'at_risk',
                'vulnerabilities': 1,
                'inherited_vulnerabilities': ['CVE-2023-32681'],
                'last_scan': datetime.now().isoformat()
            },
            timestamp=(datetime.now() + timedelta(minutes=4)).isoformat()
        )
        self.add_component(app_component)
        
        # Simulate vulnerability discovery
        self.simulate_vulnerability_discovery()
        
        return app_component
    
    def simulate_vulnerability_discovery(self):
        """Simulate discovery of new vulnerability in supply chain"""
        vulnerability_alert = {
            'timestamp': (datetime.now() + timedelta(hours=1)).isoformat(),
            'event_type': 'vulnerability_discovered',
            'cve_id': 'CVE-2023-32681',
            'affected_component': 'requests:2.25.1',
            'severity': 'medium',
            'description': 'Potential security vulnerability in requests library',
            'impact_assessment': self.assess_vulnerability_impact('requests:2.25.1')
        }
        
        self.vulnerability_alerts.append(vulnerability_alert)
        self.thread_events.append(vulnerability_alert)
    
    def assess_vulnerability_impact(self, vulnerable_component: str):
        """Assess impact of vulnerability across supply chain"""
        impact = {
            'directly_affected': [vulnerable_component],
            'downstream_affected': [],
            'risk_level': 'medium'
        }
        
        # Find all components that depend on the vulnerable component
        for comp_key, component in self.components.items():
            if vulnerable_component in component.dependencies:
                impact['downstream_affected'].append(comp_key)
        
        if len(impact['downstream_affected']) > 0:
            impact['risk_level'] = 'high'
        
        return impact
    
    def visualize_supply_chain(self):
        """Visualize the supply chain dependency graph"""
        G = nx.DiGraph()
        
        # Add nodes
        for comp_key, component in self.components.items():
            status = component.security_scan_results.get('status', 'unknown')
            G.add_node(comp_key, 
                      name=component.name,
                      version=component.version,
                      status=status)
        
        # Add edges (dependencies)
        for comp_key, component in self.components.items():
            for dep in component.dependencies:
                G.add_edge(dep, comp_key)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Color nodes by security status
        node_colors = {
            'clean': 'green',
            'vulnerable': 'red',
            'at_risk': 'orange',
            'unknown': 'gray'
        }
        
        colors = [node_colors.get(G.nodes[node]['status'], 'gray') for node in G.nodes()]
        
        # Draw the graph
        nx.draw(G, pos, node_color=colors, node_size=2000, 
                with_labels=False, arrows=True, arrowsize=20,
                edge_color='gray', alpha=0.7)
        
        # Add labels
        labels = {node: f"{G.nodes[node]['name']}\n{G.nodes[node]['version']}" 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Software Supply Chain Digital Thread\n(Green=Clean, Red=Vulnerable, Orange=At Risk)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def generate_sbom(self):
        """Generate Software Bill of Materials (SBOM)"""
        sbom = {
            'sbom_version': '1.0',
            'generated': datetime.now().isoformat(),
            'components': [],
            'vulnerabilities': [],
            'digital_thread_hash': self.calculate_thread_hash()
        }
        
        for comp_key, component in self.components.items():
            sbom['components'].append({
                'name': component.name,
                'version': component.version,
                'source': component.source,
                'hash': component.hash_value,
                'dependencies': component.dependencies,
                'security_status': component.security_scan_results
            })
        
        for alert in self.vulnerability_alerts:
            sbom['vulnerabilities'].append(alert)
        
        return sbom
    
    def calculate_thread_hash(self):
        """Calculate hash of entire digital thread for integrity"""
        thread_data = json.dumps(self.thread_events, sort_keys=True)
        return hashlib.sha256(thread_data.encode()).hexdigest()
    
    def export_thread_report(self, filename: str = "supply_chain_thread_report.json"):
        """Export complete supply chain digital thread report"""
        report = {
            'report_generated': datetime.now().isoformat(),
            'supply_chain_summary': {
                'total_components': len(self.components),
                'vulnerable_components': len([c for c in self.components.values() 
                                            if c.security_scan_results.get('status') == 'vulnerable']),
                'at_risk_components': len([c for c in self.components.values() 
                                         if c.security_scan_results.get('status') == 'at_risk']),
                'thread_integrity_hash': self.calculate_thread_hash()
            },
            'sbom': self.generate_sbom(),
            'complete_thread': self.thread_events,
            'vulnerability_alerts': self.vulnerability_alerts
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Supply chain digital thread report exported to {filename}")
        return filename

# Demo execution
if __name__ == "__main__":
    print("=== Supply Chain Digital Thread Demo ===\n")
    
    # Initialize supply chain tracker
    supply_chain = SupplyChainDigitalThread()
    
    # Simulate supply chain
    app_component = supply_chain.simulate_supply_chain()
    
    print(f"Supply chain simulation complete!")
    print(f"Total components: {len(supply_chain.components)}")
    print(f"Vulnerability alerts: {len(supply_chain.vulnerability_alerts)}")
    
    # Show component summary
    print(f"\n=== Component Security Status ===")
    for comp_key, component in supply_chain.components.items():
        status = component.security_scan_results.get('status', 'unknown')
        vulns = component.security_scan_results.get('vulnerabilities', 0)
        print(f"  {component.name} v{component.version}: {status} ({vulns} vulnerabilities)")
    
    # Show vulnerability impact
    if supply_chain.vulnerability_alerts:
        print(f"\n=== Vulnerability Impact Analysis ===")
        for alert in supply_chain.vulnerability_alerts:
            print(f"CVE: {alert['cve_id']}")
            print(f"  Affected: {alert['affected_component']}")
            print(f"  Severity: {alert['severity']}")
            impact = alert['impact_assessment']
            print(f"  Downstream impact: {len(impact['downstream_affected'])} components")
            print(f"  Risk level: {impact['risk_level']}")
    
    # Generate and show SBOM
    sbom = supply_chain.generate_sbom()
    print(f"\n=== Software Bill of Materials (SBOM) ===")
    print(f"SBOM Version: {sbom['sbom_version']}")
    print(f"Components tracked: {len(sbom['components'])}")
    print(f"Vulnerabilities: {len(sbom['vulnerabilities'])}")
    print(f"Thread integrity hash: {sbom['digital_thread_hash'][:16]}...")
    
    # Visualize supply chain
    supply_chain.visualize_supply_chain()
    
    # Export complete report
    report_file = supply_chain.export_thread_report()
    
    print(f"\n=== Digital Thread Verification ===")
    print(f"Thread integrity hash: {supply_chain.calculate_thread_hash()[:16]}...")
    print(f"Total thread events: {len(supply_chain.thread_events)}")
    
    # Show chronological thread
    print(f"\n=== Chronological Digital Thread ===")
    for event in sorted(supply_chain.thread_events, key=lambda x: x['timestamp']):
        timestamp = datetime.fromisoformat(event['timestamp']).strftime('%H:%M:%S')
        event_type = event['event_type'].replace('_', ' ').title()
        if 'component' in event:
            print(f"  {timestamp}: {event_type} - {event['component']} v{event['version']}")
        else:
            print(f"  {timestamp}: {event_type} - {event.get('cve_id', 'N/A')}")
