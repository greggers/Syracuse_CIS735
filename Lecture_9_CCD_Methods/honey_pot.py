import socket
import threading
import time
import random
import json
import logging
from datetime import datetime
import hashlib
import os

class HoneyPot:
    """
    Educational Honeypot demonstrating deception techniques in cybersecurity.
    This honeypot simulates vulnerable services to attract and study attackers.
    """
    
    def __init__(self, host='localhost', base_port=8080):
        self.host = host
        self.base_port = base_port
        self.active_connections = {}
        self.attack_log = []
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('honeypot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Deception tactics configuration
        self.deception_config = {
            'fake_services': {
                'ssh': {'port': 2222, 'banner': 'SSH-2.0-OpenSSH_7.4'},
                'ftp': {'port': 2121, 'banner': '220 Welcome to FTP Server'},
                'telnet': {'port': 2323, 'banner': 'Welcome to Telnet Service'},
                'http': {'port': 8080, 'banner': 'HTTP/1.1 200 OK\r\nServer: Apache/2.4.41'}
            },
            'fake_vulnerabilities': [
                'SQL Injection in login form',
                'Directory traversal in file upload',
                'Buffer overflow in authentication',
                'Cross-site scripting in search'
            ],
            'fake_files': [
                'passwords.txt', 'config.ini', 'database.sql', 
                'admin_backup.zip', 'user_data.csv'
            ]
        }
    
    def generate_fake_system_info(self):
        """
        Generate fake system information to deceive attackers
        """
        fake_systems = [
            {
                'os': 'Ubuntu 18.04.5 LTS',
                'kernel': 'Linux 4.15.0-112-generic',
                'services': ['apache2', 'mysql', 'ssh', 'ftp'],
                'users': ['admin', 'root', 'www-data', 'mysql']
            },
            {
                'os': 'Windows Server 2016',
                'version': '10.0.14393',
                'services': ['IIS', 'SQL Server', 'RDP', 'SMB'],
                'users': ['Administrator', 'Guest', 'IIS_IUSRS']
            },
            {
                'os': 'CentOS 7.8',
                'kernel': 'Linux 3.10.0-1127',
                'services': ['httpd', 'mysqld', 'sshd', 'vsftpd'],
                'users': ['root', 'apache', 'mysql', 'ftp']
            }
        ]
        return random.choice(fake_systems)
    
    def simulate_ssh_honeypot(self, port=2222):
        """
        Simulate an SSH service with intentional vulnerabilities
        """
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, port))
            server_socket.listen(5)
            
            self.logger.info(f"SSH Honeypot listening on {self.host}:{port}")
            
            while self.running:
                try:
                    client_socket, addr = server_socket.accept()
                    thread = threading.Thread(
                        target=self.handle_ssh_connection, 
                        args=(client_socket, addr)
                    )
                    thread.daemon = True
                    thread.start()
                except socket.error:
                    break
                    
        except Exception as e:
            self.logger.error(f"SSH Honeypot error: {e}")
        finally:
            server_socket.close()
    
    def handle_ssh_connection(self, client_socket, addr):
        """
        Handle SSH connections with deceptive responses
        """
        try:
            # Log the connection attempt
            attack_info = {
                'timestamp': datetime.now().isoformat(),
                'source_ip': addr[0],
                'source_port': addr[1],
                'service': 'SSH',
                'attack_type': 'Connection Attempt'
            }
            
            # Send fake SSH banner
            banner = "SSH-2.0-OpenSSH_7.4\r\n"
            client_socket.send(banner.encode())
            
            # Simulate authentication attempts
            auth_attempts = 0
            common_passwords = ['admin', 'password', '123456', 'root', 'admin123']
            
            while auth_attempts < 3:
                try:
                    data = client_socket.recv(1024).decode('utf-8', errors='ignore')
                    if not data:
                        break
                    
                    # Log the attempt
                    attack_info['data'] = data[:100]  # First 100 chars
                    attack_info['auth_attempt'] = auth_attempts + 1
                    
                    # Simulate weak password checking (deception)
                    if any(pwd in data.lower() for pwd in common_passwords):
                        # Pretend to accept weak credentials (honeypot deception)
                        response = "Authentication successful\r\n"
                        client_socket.send(response.encode())
                        
                        # Provide fake shell environment
                        self.simulate_fake_shell(client_socket, addr)
                        break
                    else:
                        response = "Authentication failed\r\n"
                        client_socket.send(response.encode())
                    
                    auth_attempts += 1
                    
                except socket.error:
                    break
            
            self.log_attack(attack_info)
            
        except Exception as e:
            self.logger.error(f"SSH handler error: {e}")
        finally:
            client_socket.close()
    
    def simulate_fake_shell(self, client_socket, addr):
        """
        Simulate a fake shell environment to study attacker behavior
        """
        fake_system = self.generate_fake_system_info()
        current_dir = "/home/admin"
        
        # Send fake welcome message
        welcome = f"""
Welcome to {fake_system['os']}
Last login: {datetime.now().strftime('%a %b %d %H:%M:%S %Y')} from {addr[0]}
admin@honeypot:~$ """
        
        client_socket.send(welcome.encode())
        
        while True:
            try:
                command = client_socket.recv(1024).decode('utf-8', errors='ignore').strip()
                if not command:
                    break
                
                # Log the command
                command_info = {
                    'timestamp': datetime.now().isoformat(),
                    'source_ip': addr[0],
                    'service': 'SSH_Shell',
                    'command': command,
                    'directory': current_dir
                }
                self.log_attack(command_info)
                
                # Process fake commands
                response = self.process_fake_command(command, current_dir, fake_system)
                client_socket.send(response.encode())
                
                # Update current directory if cd command
                if command.startswith('cd '):
                    new_dir = command[3:].strip()
                    if new_dir == '..':
                        current_dir = '/'.join(current_dir.split('/')[:-1]) or '/'
                    elif new_dir.startswith('/'):
                        current_dir = new_dir
                    else:
                        current_dir = f"{current_dir}/{new_dir}".replace('//', '/')
                
            except socket.error:
                break
    
    def process_fake_command(self, command, current_dir, fake_system):
        """
        Process commands in fake shell environment
        """
        cmd_parts = command.split()
        if not cmd_parts:
            return "admin@honeypot:~$ "
        
        cmd = cmd_parts[0].lower()
        
        # Simulate common commands with fake responses
        if cmd == 'ls':
            fake_files = random.sample(self.deception_config['fake_files'], 3)
            fake_dirs = ['Documents', 'Downloads', 'Desktop']
            response = '\n'.join(fake_dirs + fake_files) + '\n'
            
        elif cmd == 'pwd':
            response = f"{current_dir}\n"
            
        elif cmd == 'whoami':
            response = "admin\n"
            
        elif cmd == 'id':
            response = "uid=1000(admin) gid=1000(admin) groups=1000(admin),4(adm),24(cdrom),27(sudo)\n"
            
        elif cmd == 'uname':
            response = f"{fake_system['kernel']}\n"
            
        elif cmd == 'ps':
            fake_processes = [
                "  PID TTY          TIME CMD",
                " 1234 pts/0    00:00:01 bash",
                " 5678 pts/0    00:00:00 apache2",
                " 9012 pts/0    00:00:00 mysql"
            ]
            response = '\n'.join(fake_processes) + '\n'
            
        elif cmd == 'netstat':
            fake_connections = [
                "Active Internet connections",
                "Proto Recv-Q Send-Q Local Address           Foreign Address         State",
                "tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN",
                "tcp        0      0 0.0.0.0:80              0.0.0.0:*               LISTEN",
                "tcp        0      0 0.0.0.0:3306            0.0.0.0:*               LISTEN"
            ]
            response = '\n'.join(fake_connections) + '\n'
            
        elif cmd == 'cat' and len(cmd_parts) > 1:
            filename = cmd_parts[1]
            if filename == 'passwords.txt':
                response = "admin:password123\nroot:admin\nuser:123456\n"
            elif filename == 'config.ini':
                response = "[database]\nhost=localhost\nuser=root\npassword=secret123\n"
            else:
                response = f"cat: {filename}: No such file or directory\n"
                
        elif cmd in ['exit', 'quit', 'logout']:
            response = "Connection closed.\n"
            return response
            
        else:
            response = f"{cmd}: command not found\n"
        
        return response + "admin@honeypot:~$ "
    
    def simulate_web_honeypot(self, port=8080):
        """
        Simulate a web server with fake vulnerabilities
        """
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, port))
            server_socket.listen(5)
            
            self.logger.info(f"Web Honeypot listening on {self.host}:{port}")
            
            while self.running:
                try:
                    client_socket, addr = server_socket.accept()
                    thread = threading.Thread(
                        target=self.handle_web_connection, 
                        args=(client_socket, addr)
                    )
                    thread.daemon = True
                    thread.start()
                except socket.error:
                    break
                    
        except Exception as e:
            self.logger.error(f"Web Honeypot error: {e}")
        finally:
            server_socket.close()
    
    def handle_web_connection(self, client_socket, addr):
        """
        Handle HTTP requests with fake vulnerable responses
        """
        try:
            request = client_socket.recv(4096).decode('utf-8', errors='ignore')
            if not request:
                return
            
            # Log the request
            attack_info = {
                'timestamp': datetime.now().isoformat(),
                'source_ip': addr[0],
                'service': 'HTTP',
                'request': request.split('\n')[0] if request else 'Empty request'
            }
            
            # Analyze request for attack patterns
            attack_patterns = {
                'sql_injection': ['union', 'select', 'drop', 'insert', "'", '"'],
                'xss': ['<script>', 'javascript:', 'onerror=', 'onload='],
                'directory_traversal': ['../', '..\\', '%2e%2e', 'etc/passwd'],
                'command_injection': ['|', ';', '&&', '$(', '`']
            }
            
            detected_attacks = []
            request_lower = request.lower()
            
            for attack_type, patterns in attack_patterns.items():
                if any(pattern in request_lower for pattern in patterns):
                    detected_attacks.append(attack_type)
            
            attack_info['detected_attacks'] = detected_attacks
            
            # Generate fake vulnerable response
            if 'sql_injection' in detected_attacks:
                response = self.generate_fake_sql_response()
            elif 'directory_traversal' in detected_attacks:
                response = self.generate_fake_file_response()
            else:
                response = self.generate_fake_web_page()
            
            client_socket.send(response.encode())
            self.log_attack(attack_info)
            
        except Exception as e:
            self.logger.error(f"Web handler error: {e}")
        finally:
            client_socket.close()
    
    def generate_fake_sql_response(self):
        """
        Generate fake SQL injection response to deceive attackers
        """
        fake_data = [
            "admin:5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "user1:ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f",
            "guest:2c6ee24b09816a6f14f95d1698b24ead",
        ]
        
        response = """HTTP/1.1 200 OK
Content-Type: text/html
Server: Apache/2.4.41

<html><body>
<h2>Database Query Result</h2>
<table border="1">
<tr><th>Username</th><th>Password Hash</th></tr>
"""
        
        for entry in fake_data:
            username, hash_val = entry.split(':')
            response += f"<tr><td>{username}</td><td>{hash_val}</td></tr>\n"
        
        response += """</table>
<p><i>Note: This is a honeypot. All data is fake.</i></p>
</body></html>"""
        
        return response
    
    def generate_fake_file_response(self):
        """
        Generate fake file content for directory traversal attacks
        """
        fake_passwd = """root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
bin:x:2:2:bin:/bin:/usr/sbin/nologin
sys:x:3:3:sys:/dev:/usr/sbin/nologin
admin:x:1000:1000:Admin User:/home/admin:/bin/bash
mysql:x:999:999:MySQL Server:/var/lib/mysql:/bin/false
www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin"""
        
        response = f"""HTTP/1.1 200 OK
Content-Type: text/plain
Server: Apache/2.4.41

{fake_passwd}

<!-- This is a honeypot - fake /etc/passwd file -->
"""
        return response
    
    def generate_fake_web_page(self):
        """
        Generate fake vulnerable web page
        """
        response = """HTTP/1.1 200 OK
Content-Type: text/html
Server: Apache/2.4.41

<!DOCTYPE html>
<html>
<head>
    <title>Corporate Admin Panel</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .login-box { border: 1px solid #ccc; padding: 20px; width: 300px; }
        .vulnerable { color: red; font-size: 12px; }
    </style>
</head>
<body>
    <h1>Corporate Admin Panel</h1>
    <div class="login-box">
        <h3>Administrator Login</h3>
        <form action="/login" method="POST">
            <p>Username: <input type="text" name="username" value="admin"></p>
            <p>Password: <input type="password" name="password"></p>
            <p><input type="submit" value="Login"></p>
        </form>
        <p class="vulnerable">Debug: SQL Query = SELECT * FROM users WHERE username='admin' AND password='[INPUT]'</p>
    </div>
    
    <h3>File Manager</h3>
    <p><a href="/files?file=config.txt">View Config</a></p>
    <p><a href="/files?file=../../../etc/passwd">System Files</a></p>
    
    <h3>Search</h3>
    <form action="/search" method="GET">
        <input type="text" name="q" placeholder="Search...">
        <input type="submit" value="Search">
    </form>
    <p class="vulnerable">Note: Search results displayed without sanitization</p>
    
    <!-- Honeypot: Intentionally vulnerable web application -->
    <!-- Real vulnerabilities: SQL injection, XSS, Directory traversal -->
</body>
</html>"""
        return response
    
    def log_attack(self, attack_info):
        """
        Log attack information for analysis
        """
        self.attack_log.append(attack_info)
        self.logger.info(f"Attack logged: {attack_info}")
        
        # Save to JSON file for analysis
        with open('attack_log.json', 'w') as f:
            json.dump(self.attack_log, f, indent=2)
    
    def start_honeypot(self):
        """
        Start all honeypot services
        """
        self.running = True
        self.logger.info("Starting Honeypot Services...")
        
        # Start different service threads
        services = [
            threading.Thread(target=self.simulate_ssh_honeypot, args=(2222,)),
            threading.Thread(target=self.simulate_web_honeypot, args=(8080,)),
            threading.Thread(target=self.monitor_attacks)
        ]
        
        for service in services:
            service.daemon = True
            service.start()
        
        self.logger.info("All honeypot services started!")
        return services
    
    def stop_honeypot(self):
        """
        Stop honeypot services
        """
        self.running = False
        self.logger.info("Stopping honeypot services...")
    
    def monitor_attacks(self):
        """
        Monitor and analyze attack patterns
        """
        while self.running:
            time.sleep(30)  # Check every 30 seconds
            
            if len(self.attack_log) > 0:
                recent_attacks = [
                    attack for attack in self.attack_log 
                    if (datetime.now() - datetime.fromisoformat(attack['timestamp'])).seconds < 300
                ]
                
                if recent_attacks:
                    self.analyze_attack_patterns(recent_attacks)
    
    def analyze_attack_patterns(self, attacks):
        """
        Analyze attack patterns and generate alerts
        """
        # Count attacks by source IP
        ip_counts = {}
        service_counts = {}
        
        for attack in attacks:
            ip = attack.get('source_ip', 'unknown')
            service = attack.get('service', 'unknown')
            
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
            service_counts[service] = service_counts.get(service, 0) + 1
        
        # Generate alerts for suspicious activity
        for ip, count in ip_counts.items():
            if count > 5:  # More than 5 attempts in 5 minutes
                self.logger.warning(f"ALERT: High activity from IP {ip}: {count} attempts")
        
        # Log service targeting
        for service, count in service_counts.items():
            if count > 3:
                self.logger.info(f"Service {service} targeted {count} times")
    
    def generate_attack_report(self):
        """
        Generate comprehensive attack analysis report
        """
        if not self.attack_log:
            print("No attacks logged yet.")
            return
        
        print("\n" + "="*60)
        print("HONEYPOT ATTACK ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        total_attacks = len(self.attack_log)
        unique_ips = len(set(attack.get('source_ip') for attack in self.attack_log))
        
        print(f"Total Attack Attempts: {total_attacks}")
        print(f"Unique Source IPs: {unique_ips}")
        
        # Service targeting analysis
        services = {}
        for attack in self.attack_log:
            service = attack.get('service', 'unknown')
            services[service] = services.get(service, 0) + 1
        
        print(f"\nServices Targeted:")
        for service, count in sorted(services.items(), key=lambda x: x[1], reverse=True):
            print(f"  {service}: {count} attempts")
        
        # Attack type analysis
        attack_types = {}
        for attack in self.attack_log:
            detected = attack.get('detected_attacks', [])
            for attack_type in detected:
                attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
        
        if attack_types:
            print(f"\nAttack Types Detected:")
            for attack_type, count in sorted(attack_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {attack_type.replace('_', ' ').title()}: {count} attempts")
        
        # Top attacking IPs
        ip_counts = {}
        for attack in self.attack_log:
            ip = attack.get('source_ip', 'unknown')
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
        
        print(f"\nTop Attacking IPs:")
        for ip, count in sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {ip}: {count} attempts")
        
        # Recent activity
        recent_attacks = [
            attack for attack in self.attack_log 
            if (datetime.now() - datetime.fromisoformat(attack['timestamp'])).seconds < 3600
        ]
        
        print(f"\nRecent Activity (last hour): {len(recent_attacks)} attempts")
        
        print("\n" + "="*60)
    
    def demonstrate_deception_techniques(self):
        """
        Demonstrate the deception techniques used in the honeypot
        """
        print("\n" + "="*60)
        print("HONEYPOT DECEPTION TECHNIQUES DEMONSTRATION")
        print("="*60)
        
        print("\n1. SERVICE IMPERSONATION:")
        print("   - Fake SSH service on port 2222")
        print("   - Fake web server on port 8080")
        print("   - Realistic service banners and responses")
        
        print("\n2. FAKE VULNERABILITY EXPOSURE:")
        print("   - Intentional SQL injection points")
        print("   - Directory traversal opportunities")
        print("   - Weak authentication mechanisms")
        print("   - Exposed configuration files")
        
        print("\n3. SYSTEM INFORMATION DECEPTION:")
        fake_system = self.generate_fake_system_info()
        print(f"   - Fake OS: {fake_system['os']}")
        print(f"   - Fake Services: {', '.join(fake_system['services'])}")
        print(f"   - Fake Users: {', '.join(fake_system['users'])}")
        
        print("\n4. BEHAVIORAL DECEPTION:")
        print("   - Accepts weak passwords to encourage further exploration")
        print("   - Provides fake sensitive data (passwords, configs)")
        print("   - Simulates realistic command responses")
        print("   - Logs all attacker activities for analysis")
        
        print("\n5. INTELLIGENCE GATHERING:")
        print("   - Tracks attack patterns and techniques")
        print("   - Identifies common attack tools and methods")
        print("   - Maps attacker behavior and objectives")
        print("   - Provides early warning of new threats")
        
        print("\n" + "="*60)

def simulate_attacks(honeypot_host='localhost'):
    """
    Simulate various attacks against the honeypot for demonstration
    """
    print("\n" + "="*50)
    print("SIMULATING ATTACKS FOR DEMONSTRATION")
    print("="*50)
    
    # Simulate SSH brute force
    print("\n1. Simulating SSH Brute Force Attack...")
    try:
        ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ssh_socket.connect((honeypot_host, 2222))
        
        # Try common passwords
        passwords = ['admin', 'password', '123456']
        for pwd in passwords:
            ssh_socket.send(f"password:{pwd}\n".encode())
            response = ssh_socket.recv(1024).decode()
            print(f"   Tried password '{pwd}': {response.strip()}")
            time.sleep(1)
        
        ssh_socket.close()
    except Exception as e:
        print(f"   SSH simulation failed: {e}")
    
    # Simulate web attacks
    print("\n2. Simulating Web Attacks...")
    web_attacks = [
        "GET /?id=1' UNION SELECT * FROM users-- HTTP/1.1\r\nHost: localhost\r\n\r\n",
        "GET /files?file=../../../etc/passwd HTTP/1.1\r\nHost: localhost\r\n\r\n",
        "GET /search?q=<script>alert('XSS')</script> HTTP/1.1\r\nHost: localhost\r\n\r\n"
    ]
    
    for i, attack in enumerate(web_attacks, 1):
        try:
            web_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            web_socket.connect((honeypot_host, 8080))
            web_socket.send(attack.encode())
            response = web_socket.recv(1024).decode()
            print(f"   Attack {i}: {attack.split()[1]} - Response received")
            web_socket.close()
            time.sleep(1)
        except Exception as e:
            print(f"   Web attack {i} failed: {e}")

def main():
    """
    Main function to demonstrate the honeypot
    """
    print("="*60)
    print("EDUCATIONAL HONEYPOT DEMONSTRATION")
    print("Concealment, Camouflage, and Deception (CCD) Methods")
    print("="*60)
    
    # Create honeypot instance
    honeypot = HoneyPot()
    
    # Demonstrate deception techniques
    honeypot.demonstrate_deception_techniques()
    
    # Start honeypot services
    services = honeypot.start_honeypot()
    
    try:
        print(f"\nHoneypot is running...")
        print(f"SSH Honeypot: telnet {honeypot.host} 2222")
        print(f"Web Honeypot: http://{honeypot.host}:8080")
        print("\nWaiting for attacks... (Press Ctrl+C to stop)")
        
        # Wait a moment for services to start
        time.sleep(2)
        
        # Simulate some attacks for demonstration
        simulate_attacks(honeypot.host)
        
        # Wait for more activity
        time.sleep(10)
        
        # Generate attack report
        honeypot.generate_attack_report()
        
    except KeyboardInterrupt:
        print("\nShutting down honeypot...")
    finally:
        honeypot.stop_honeypot()
        print("Honeypot stopped.")
        
        # Final report
        if honeypot.attack_log:
            print(f"\nFinal Statistics:")
            print(f"Total attacks logged: {len(honeypot.attack_log)}")
            print(f"Attack log saved to: attack_log.json")

if __name__ == "__main__":
    main()

