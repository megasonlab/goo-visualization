"""Example script demonstrating basic usage of the goo-visualization library."""

import matplotlib.pyplot as plt
from goo_vis import GooDataLoader, GooVisualizer, create_visualization_dashboard

def main():
    # Path to your Goo data file
    data_path = "path/to/your/data.h5"
    
    # Example 1: Basic cell position plot
    with GooDataLoader(data_path) as loader:
        visualizer = GooVisualizer(loader)
        
        # Plot cell positions colored by volume
        ax = visualizer.plot_cell_positions("frame_009", color_by="volume")
        plt.show()
        
        # Plot concentration heatmap for a specific molecule
        ax = visualizer.plot_concentration_heatmap("frame_009", "molecule_name")
        plt.show()
        
        # Plot time series of volume across multiple frames
        frames = [f"frame_{i:03d}" for i in range(1, 10)]
        ax = visualizer.plot_time_series(frames, "volume")
        plt.show()
        
        # Plot volume distribution
        ax = visualizer.plot_distribution("frame_009", "volume")
        plt.show()
    
    # Example 2: Create a comprehensive dashboard
    create_visualization_dashboard(data_path, "frame_009")

if __name__ == "__main__":
    main() 