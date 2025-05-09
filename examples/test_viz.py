#!/usr/bin/env python3
"""Test script for gooviz visualization."""

import matplotlib.pyplot as plt
from gooviz import GooDataLoader, GooVisualizer

def main():
    data_path = "examples/out.h5"
    frame = "frame_009"
    
    with GooDataLoader(data_path) as loader:
        visualizer = GooVisualizer(loader)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot cell positions with volume coloring
        ax1 = plt.subplot(221)
        visualizer.plot_cell_positions(frame, color_by='volume', ax=ax1)
        
        # Plot cell positions with pressure coloring
        ax2 = plt.subplot(222)
        visualizer.plot_cell_positions(frame, color_by='pressure', ax=ax2)
        
        # Plot volume distribution
        ax3 = plt.subplot(223)
        visualizer.plot_distribution(frame, 'volume', ax=ax3)
        
        # Plot pressure distribution
        ax4 = plt.subplot(224)
        visualizer.plot_distribution(frame, 'pressure', ax=ax4)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main() 