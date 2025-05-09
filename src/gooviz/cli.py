"""Command-line interface for Goo visualization."""

import argparse
from pathlib import Path
from .visualizer import create_visualization_dashboard

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Goo Simulation Visualization Dashboard")
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the HDF5 data file"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port number for the dashboard server (default: 8050)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the dashboard in debug mode"
    )
    
    args = parser.parse_args()
    
    # Validate filepath
    filepath = Path(args.filepath)
    if not filepath.exists():
        parser.error(f"File not found: {filepath}")
    if not filepath.suffix == '.h5':
        parser.error("File must be an HDF5 file (.h5)")
        
    # Run the dashboard
    create_visualization_dashboard(str(filepath))

if __name__ == "__main__":
    main() 