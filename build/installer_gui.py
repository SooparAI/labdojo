#!/usr/bin/env python3
"""
Lab Dojo GUI Installer
Cross-platform installer with tkinter GUI
Can be packaged with PyInstaller to create .exe (Windows) or .app (macOS)

Build commands:
  Windows: pyinstaller --onefile --windowed --icon=logo.ico --name="LabDojo_Installer" installer_gui.py
  macOS:   pyinstaller --onefile --windowed --icon=logo.icns --name="LabDojo_Installer" installer_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import sys
import os
import platform
import threading
import shutil
from pathlib import Path

VERSION = "0.1.2"
GITHUB_REPO = "https://github.com/SooparAI/labdojo"

class InstallerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Lab Dojo v{VERSION} Installer")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Header
        header = tk.Frame(root, bg="#1a1a2e", height=80)
        header.pack(fill=tk.X)
        
        title_label = tk.Label(
            header,
            text=f"Lab Dojo v{VERSION}",
            font=("Arial", 24, "bold"),
            bg="#1a1a2e",
            fg="#00ff88"
        )
        title_label.pack(pady=20)
        
        subtitle = tk.Label(
            header,
            text="AI Research Workstation for Pathology",
            font=("Arial", 10),
            bg="#1a1a2e",
            fg="#ffffff"
        )
        subtitle.pack()
        
        # Progress area
        progress_frame = tk.Frame(root, padx=20, pady=20)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(
            progress_frame,
            text="Ready to install",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=10)
        
        self.progress = ttk.Progressbar(
            progress_frame,
            mode='indeterminate',
            length=500
        )
        self.progress.pack(pady=10)
        
        # Log output
        self.log_text = scrolledtext.ScrolledText(
            progress_frame,
            height=15,
            width=70,
            font=("Courier", 9),
            bg="#f5f5f5"
        )
        self.log_text.pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(root, padx=20, pady=10)
        button_frame.pack(fill=tk.X)
        
        self.install_btn = tk.Button(
            button_frame,
            text="Install & Launch",
            command=self.start_installation,
            bg="#00ff88",
            fg="#000000",
            font=("Arial", 12, "bold"),
            width=20,
            height=2
        )
        self.install_btn.pack(side=tk.LEFT, padx=5)
        
        self.quit_btn = tk.Button(
            button_frame,
            text="Quit",
            command=root.quit,
            font=("Arial", 12),
            width=10,
            height=2
        )
        self.quit_btn.pack(side=tk.RIGHT, padx=5)
        
    def log(self, message):
        """Add message to log window"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def run_command(self, cmd, description):
        """Run a shell command and log output"""
        self.log(f"\n[{description}]")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.stdout:
                self.log(result.stdout.strip())
            if result.returncode != 0:
                if result.stderr:
                    self.log(f"ERROR: {result.stderr.strip()}")
                return False
            return True
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            return False
            
    def check_python(self):
        """Check if Python 3.9+ is installed"""
        self.status_label.config(text="[1/5] Checking Python...")
        self.log("Checking Python installation...")
        
        try:
            result = subprocess.run(
                [sys.executable, "--version"],
                capture_output=True,
                text=True
            )
            version = result.stdout.strip()
            self.log(f"✓ {version}")
            return True
        except:
            self.log("✗ Python not found!")
            messagebox.showerror(
                "Python Required",
                "Python 3.9+ is required.\nPlease install from python.org"
            )
            return False
            
    def install_dependencies(self):
        """Install Python dependencies"""
        self.status_label.config(text="[2/5] Installing dependencies...")
        self.log("\nInstalling Python packages...")
        
        packages = ["fastapi", "uvicorn", "aiohttp", "aiosqlite", "pydantic"]
        cmd = f"{sys.executable} -m pip install --quiet --upgrade pip {' '.join(packages)}"
        
        if self.run_command(cmd, "pip install"):
            self.log("✓ Dependencies installed")
            return True
        return False
        
    def check_ollama(self):
        """Check if Ollama is installed"""
        self.status_label.config(text="[3/5] Checking Ollama...")
        self.log("\nChecking Ollama...")
        
        if shutil.which("ollama"):
            self.log("✓ Ollama found")
            
            # Start Ollama if not running
            if platform.system() == "Windows":
                subprocess.Popen(["ollama", "serve"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            return True
        else:
            self.log("⚠ Ollama not found")
            self.log("Please install from: https://ollama.com/download")
            messagebox.showwarning(
                "Ollama Recommended",
                "Ollama is recommended for local AI.\nInstall from: https://ollama.com/download\n\nYou can continue without it."
            )
            return True
            
    def setup_config(self):
        """Create config directory"""
        self.status_label.config(text="[4/5] Setting up config...")
        self.log("\nCreating config directory...")
        
        config_dir = Path.home() / ".labdojo" / "logs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        self.log(f"✓ Config: {config_dir.parent}")
        return True
        
    def launch_labdojo(self):
        """Launch Lab Dojo"""
        self.status_label.config(text="[5/5] Launching Lab Dojo...")
        self.log("\nStarting Lab Dojo...")
        
        # Find labdojo.py in the same directory as this installer
        script_dir = Path(__file__).parent.parent
        labdojo_py = script_dir / "labdojo.py"
        
        if not labdojo_py.exists():
            self.log(f"✗ labdojo.py not found at {labdojo_py}")
            messagebox.showerror("Error", f"labdojo.py not found!\nExpected at: {labdojo_py}")
            return False
            
        self.log(f"✓ Found: {labdojo_py}")
        self.log("\nLab Dojo will open at http://localhost:8080")
        
        # Launch in background
        subprocess.Popen([sys.executable, str(labdojo_py)])
        
        return True
        
    def installation_thread(self):
        """Run installation steps in background thread"""
        self.progress.start()
        self.install_btn.config(state=tk.DISABLED)
        
        steps = [
            self.check_python,
            self.install_dependencies,
            self.check_ollama,
            self.setup_config,
            self.launch_labdojo
        ]
        
        for step in steps:
            if not step():
                self.progress.stop()
                self.install_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Installation failed")
                return
                
        self.progress.stop()
        self.status_label.config(text="✓ Installation complete!")
        self.log("\n✓ Lab Dojo is running!")
        
        messagebox.showinfo(
            "Success",
            "Lab Dojo is now running!\n\nOpen your browser to:\nhttp://localhost:8080"
        )
        
    def start_installation(self):
        """Start installation in background thread"""
        thread = threading.Thread(target=self.installation_thread, daemon=True)
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = InstallerGUI(root)
    root.mainloop()
