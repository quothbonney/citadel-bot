#!/usr/bin/env python3
"""
Standalone risk monitor viewer - run this in a separate terminal to watch spreads.

Usage:
    python monitor_viewer.py                    # Watch in terminal (updates every second)
    python monitor_viewer.py --gui              # Open GUI window
    python monitor_viewer.py --interval 5       # Update every 5 seconds
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from auth import connect, wait_for_market
from log_config import init_logging
from params import StrategyParams, DEFAULT_PARAMS_PATH
from risk_monitor import SpreadMonitor
from settings import settings


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def display_terminal(monitor: SpreadMonitor, case: dict):
    """Display monitor stats in terminal with formatting."""
    clear_screen()
    print("=" * 80)
    print(f"RISK MONITOR - Period {case.get('period', 0)}, Tick {case.get('tick', 0)}")
    print("=" * 80)
    print()
    
    summary = monitor.get_summary()
    if not summary:
        print("No data available yet...")
        return
    
    for pair_id, stats in summary.items():
        # Color coding based on spread magnitude
        spread = stats['current']
        std = stats['std']
        z_score = (spread - stats['mean']) / std if std > 0 else 0
        
        # Determine color/status
        if abs(z_score) > 2:
            status = "⚠️  HIGH"
        elif abs(z_score) > 1:
            status = "⚡ MEDIUM"
        else:
            status = "✓  NORMAL"
        
        print(f"\n{pair_id}")
        print("-" * 80)
        print(f"  Current Spread: {spread:>12.6f}  [{status}]")
        print(f"  Mean:           {stats['mean']:>12.6f}")
        print(f"  Std Dev:        {stats['std']:>12.6f}")
        print(f"  Z-Score:        {z_score:>12.2f}")
        print(f"  Min:            {stats['min']:>12.6f}")
        print(f"  Max:            {stats['max']:>12.6f}")
        print(f"  Samples:        {stats['count']:>12d}")
    
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit")


def display_gui(monitor: SpreadMonitor, client, update_interval: float = 1.0):
    """Display monitor in a GUI window using tkinter."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("Error: tkinter not available. Install with: pip install tk")
        return
    
    root = tk.Tk()
    root.title("Risk Monitor - Spread Tracker")
    root.geometry("800x600")
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    
    # Header
    header = ttk.Label(main_frame, text="RISK MONITOR", font=("Arial", 16, "bold"))
    header.grid(row=0, column=0, columnspan=2, pady=10)
    
    # Status label
    status_label = ttk.Label(main_frame, text="Initializing...", font=("Arial", 10))
    status_label.grid(row=1, column=0, columnspan=2, pady=5)
    
    # Create text widget for stats
    text_widget = tk.Text(main_frame, wrap=tk.WORD, font=("Courier", 10), height=20)
    text_widget.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(2, weight=1)
    
    def update_display():
        try:
            case = client.get_case()
            if case.get('status') != 'ACTIVE':
                status_label.config(text="Market closed")
                root.after(int(update_interval * 1000), update_display)
                return
            
            portfolio = client.get_portfolio()
            monitor.update(portfolio, case)
            
            # Update status
            status_label.config(
                text=f"Period {case.get('period', 0)}, Tick {case.get('tick', 0)} - "
                     f"Last update: {time.strftime('%H:%M:%S')}"
            )
            
            # Update stats display
            text_widget.delete(1.0, tk.END)
            summary = monitor.get_summary()
            
            if not summary:
                text_widget.insert(tk.END, "No data available yet...\n")
            else:
                for pair_id, stats in summary.items():
                    spread = stats['current']
                    std = stats['std']
                    z_score = (spread - stats['mean']) / std if std > 0 else 0
                    
                    text_widget.insert(tk.END, f"\n{pair_id}\n")
                    text_widget.insert(tk.END, "-" * 70 + "\n")
                    text_widget.insert(tk.END, f"  Current Spread: {spread:>12.6f}\n")
                    text_widget.insert(tk.END, f"  Mean:           {stats['mean']:>12.6f}\n")
                    text_widget.insert(tk.END, f"  Std Dev:        {stats['std']:>12.6f}\n")
                    text_widget.insert(tk.END, f"  Z-Score:        {z_score:>12.2f}\n")
                    text_widget.insert(tk.END, f"  Min:            {stats['min']:>12.6f}\n")
                    text_widget.insert(tk.END, f"  Max:            {stats['max']:>12.6f}\n")
                    text_widget.insert(tk.END, f"  Samples:        {stats['count']:>12d}\n")
            
            root.after(int(update_interval * 1000), update_display)
        except Exception as e:
            status_label.config(text=f"Error: {e}")
            root.after(int(update_interval * 1000), update_display)
    
    # Start updates
    update_display()
    
    # Handle window close
    def on_closing():
        root.destroy()
        sys.exit(0)
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Risk monitor viewer for spread tracking')
    parser.add_argument('--params', '-p', default=str(DEFAULT_PARAMS_PATH), help='Strategy params file')
    parser.add_argument('--gui', '-g', action='store_true', help='Open GUI window instead of terminal')
    parser.add_argument('--interval', '-i', type=float, default=1.0, help='Update interval in seconds (default: 1.0)')
    parser.add_argument('--api-key', '-k', help='Override API key')
    parser.add_argument('--api-host', '-H', help='Override API host')
    args = parser.parse_args()
    
    # Minimal logging for standalone viewer
    init_logging(console_level='WARNING')
    
    # Load params
    params = StrategyParams.load(args.params)
    
    # Connect to RIT
    api_key = args.api_key or settings.api_key
    api_host = args.api_host or settings.api_host
    client = connect(api_key, api_host)
    wait_for_market(client)
    
    # Create monitor
    monitor = SpreadMonitor(params)
    
    if args.gui:
        # GUI mode
        display_gui(monitor, client, args.interval)
    else:
        # Terminal mode
        try:
            while True:
                case = client.get_case()
                if case.get('status') != 'ACTIVE':
                    print("Market closed.")
                    break
                
                portfolio = client.get_portfolio()
                monitor.update(portfolio, case)
                display_terminal(monitor, case)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")


if __name__ == '__main__':
    main()


