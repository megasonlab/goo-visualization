"""Interactive visualization module for Goo simulation data."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback_context
import logging
from typing import List, Optional, Dict, Any
from .data_loader import GooDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GooVisualizer:
    """Class for creating interactive visualizations of Goo simulation data."""
    
    def __init__(self, data_loader: GooDataLoader):
        """Initialize the visualizer with a data loader.
        
        Args:
            data_loader: GooDataLoader instance for accessing simulation data
        """
        self.data_loader = data_loader
        self.app = Dash(__name__)
        
        # Calculate global ranges for consistent scaling
        self._calculate_global_ranges()
        
        self._setup_dashboard()
        
    def _calculate_global_ranges(self):
        """Calculate maximum values and coordinate ranges across all frames for consistent scaling."""
        logger.info("Calculating global ranges across all frames...")
        frames = self.data_loader.get_available_frames()
        
        # Initialize with extreme values
        ranges = {
            'volume': {'min': float('inf'), 'max': float('-inf')},
            'pressure': {'min': float('inf'), 'max': float('-inf')}
        }
        
        for frame in frames:
            df = self.data_loader.get_frame_data(frame)
            # Update volume ranges
            ranges['volume']['min'] = min(ranges['volume']['min'], df['volume'].min())
            ranges['volume']['max'] = max(ranges['volume']['max'], df['volume'].max())
            # Update pressure ranges
            ranges['pressure']['min'] = min(ranges['pressure']['min'], df['pressure'].min())
            ranges['pressure']['max'] = max(ranges['pressure']['max'], df['pressure'].max())
            
        self.global_ranges = ranges
        logger.info(f"Global ranges calculated: {self.global_ranges}")
        
    def _create_cell_scatter_2d(self, df: pd.DataFrame, color_by: str, size_by: str) -> go.Figure:
        """Create a 2D interactive scatter plot of cell positions."""
        logger.info(f"Creating 2D cell scatter plot with {len(df)} cells")
        
        try:
            # Calculate size range using global min/max
            size_values = df[size_by].fillna(0)
            min_size = self.global_ranges[size_by]['min']
            max_size = self.global_ranges[size_by]['max']
            size_range = max_size - min_size
            
            if size_range == 0:
                scaled_sizes = np.full_like(size_values, 30)
            else:
                scaled_sizes = 10 + 40 * (size_values - min_size) / size_range
            
            color_values = df[color_by].fillna(0)
            color_max = self.global_ranges[color_by]['max']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['x'],
                y=df['y'],
                mode='markers',
                marker=dict(
                    size=scaled_sizes,
                    color=color_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_by),
                    cmin=0,
                    cmax=color_max
                ),
                text=[f"Cell: {name}<br>Volume: {v:.2f}<br>Pressure: {p:.2f}" 
                      for name, v, p in zip(df['name'], df['volume'], df['pressure'])],
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title='Cell Positions (2D View)',
                xaxis_title='X Position',
                yaxis_title='Y Position',
                showlegend=False,
                xaxis=dict(
                    range=[-15, 15],
                    constrain='domain'
                ),
                yaxis=dict(
                    range=[-15, 15],
                    scaleanchor='x',
                    scaleratio=1
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating 2D scatter plot: {e}")
            return go.Figure()

    def _create_cell_scatter_3d(self, df: pd.DataFrame, color_by: str, size_by: str) -> go.Figure:
        """Create a 3D interactive scatter plot of cell positions."""
        logger.info(f"Creating 3D cell scatter plot with {len(df)} cells")
        
        try:
            size_values = df[size_by].fillna(0)
            min_size = self.global_ranges[size_by]['min']
            max_size = self.global_ranges[size_by]['max']
            size_range = max_size - min_size
            
            if size_range == 0:
                scaled_sizes = np.full_like(size_values, 30)
            else:
                scaled_sizes = 10 + 40 * (size_values - min_size) / size_range
            
            color_values = df[color_by].fillna(0)
            color_max = self.global_ranges[color_by]['max']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=df['x'],
                y=df['y'],
                z=df['z'],
                mode='markers',
                marker=dict(
                    size=scaled_sizes,
                    color=color_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_by),
                    cmin=0,
                    cmax=color_max
                ),
                text=[f"Cell: {name}<br>Volume: {v:.2f}<br>Pressure: {p:.2f}" 
                      for name, v, p in zip(df['name'], df['volume'], df['pressure'])],
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title='Cell Positions (3D View)',
                scene=dict(
                    xaxis_title='X Position',
                    yaxis_title='Y Position',
                    zaxis_title='Z Position',
                    xaxis=dict(range=[-15, 15]),
                    yaxis=dict(range=[-15, 15]),
                    zaxis=dict(range=[-15, 15]),
                    aspectmode='cube'
                ),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating 3D scatter plot: {e}")
            return go.Figure()

    def _create_cell_count_plot(self, current_frame_idx: int = 0) -> go.Figure:
        """Create a plot showing the number of cells over time.
        
        Args:
            current_frame_idx: Index of the currently selected frame
            
        Returns:
            Plotly figure object
        """
        try:
            frames = self.data_loader.get_available_frames()
            cell_counts = []
            
            for frame in frames:
                df = self.data_loader.get_frame_data(frame)
                cell_counts.append(len(df))
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(frames))),
                y=cell_counts,
                mode='lines+markers',
                name='Cell Count'
            ))
            
            # Add vertical line for current frame
            fig.add_vline(
                x=current_frame_idx,
                line_dash="dash",
                line_color="red",
                opacity=0.5
            )
            
            fig.update_layout(
                title='Number of Cells Over Time',
                xaxis_title='Frame Number',
                yaxis_title='Number of Cells',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cell count plot: {e}")
            return go.Figure()  # Return empty figure on error
        
    def _create_concentration_heatmap(self, values: np.ndarray, metadata: Dict[str, Any], molecule: str) -> go.Figure:
        """Create an interactive heatmap of concentration data.
        
        Args:
            values: 2D array of concentration values
            metadata: Dictionary containing grid metadata
            molecule: Name of the molecule being visualized
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Creating concentration heatmap for {molecule} with shape {values.shape}")
        fig = go.Figure(data=go.Heatmap(
            z=values,
            colorscale='Viridis',
            colorbar=dict(title=f'{molecule} Concentration')
        ))
        
        fig.update_layout(
            title=f'{molecule} Concentration Grid',
            xaxis_title='X',
            yaxis_title='Y',
            showlegend=False
        )
        
        return fig
        
    def _create_gene_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create an interactive heatmap of gene expression data.
        
        Args:
            df: DataFrame containing cell data
            
        Returns:
            Plotly figure object
        """
        gene_cols = [col for col in df.columns if col.startswith('gene_')]
        logger.info(f"Creating gene heatmap with {len(gene_cols)} genes and {len(df)} cells")
        gene_data = df[gene_cols].T
        
        fig = go.Figure(data=go.Heatmap(
            z=gene_data.values,
            x=df['cell_id'],
            y=[col[5:] for col in gene_cols],  # Remove 'gene_' prefix
            colorscale='Viridis',
            colorbar=dict(title='Expression Level')
        ))
        
        fig.update_layout(
            title='Gene Expression Patterns',
            xaxis_title='Cell ID',
            yaxis_title='Gene',
            showlegend=False
        )
        
        return fig
        
    def _setup_dashboard(self):
        """Set up the interactive dashboard layout and callbacks."""
        # Get initial frames
        initial_frames = self.data_loader.get_available_frames()
        initial_options = [{'label': frame, 'value': frame} for frame in initial_frames]
        initial_value = initial_frames[0] if initial_frames else None
        
        # Define modern color scheme
        colors = {
            'background': '#f8f9fa',
            'text': '#2c3e50',
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'accent': '#e74c3c'
        }
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1('Goo Simulation Dashboard', 
                    style={
                        'color': colors['text'],
                        'textAlign': 'center',
                        'margin': '20px 0',
                        'fontFamily': 'Arial, sans-serif',
                        'fontWeight': 'bold'
                    }
                ),
                html.Hr(style={'borderColor': colors['primary'], 'width': '80%', 'margin': '0 auto'})
            ], style={'backgroundColor': 'white', 'padding': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Main content
            html.Div([
                # Left sidebar - Controls
                html.Div([
                    # Frame selection
                    html.Div([
                        html.H3('Frame Selection', 
                            style={'color': colors['text'], 'marginBottom': '10px', 'fontSize': '1.2em'}
                        ),
                        dcc.Slider(
                            id='frame-slider',
                            min=0,
                            max=len(initial_frames)-1,
                            value=0,
                            marks={
                                0: {'label': initial_frames[0]},
                                len(initial_frames)-1: {'label': initial_frames[-1]}
                            },
                            step=1,
                            included=False
                        )
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '10px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'marginBottom': '20px'
                    }),
                    
                    # Visualization controls
                    html.Div([
                        html.H3('Visualization Controls', 
                            style={'color': colors['text'], 'marginBottom': '15px', 'fontSize': '1.2em'}
                        ),
                        
                        html.Div([
                            html.Label('Color cells by:', 
                                style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}
                            ),
                            dcc.Dropdown(
                                id='color-dropdown',
                                options=[
                                    {'label': 'Volume', 'value': 'volume'},
                                    {'label': 'Pressure', 'value': 'pressure'}
                                ],
                                value='volume',
                                style={'marginBottom': '15px'}
                            ),
                            
                            html.Label('Size cells by:', 
                                style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}
                            ),
                            dcc.Dropdown(
                                id='size-dropdown',
                                options=[
                                    {'label': 'Volume', 'value': 'volume'},
                                    {'label': 'Pressure', 'value': 'pressure'}
                                ],
                                value='volume',
                                style={'marginBottom': '15px'}
                            ),
                            
                            html.Label('View Mode:', 
                                style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}
                            ),
                            dcc.RadioItems(
                                id='view-mode-toggle',
                                options=[
                                    {'label': '2D View', 'value': '2d'},
                                    {'label': '3D View', 'value': '3d'}
                                ],
                                value='2d',
                                labelStyle={
                                    'display': 'inline-block',
                                    'marginRight': '15px',
                                    'cursor': 'pointer'
                                }
                            )
                        ])
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '10px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                ], style={
                    'width': '25%',
                    'padding': '20px',
                    'position': 'sticky',
                    'top': '0',
                    'height': '100vh',
                    'overflowY': 'auto'
                }),
                
                # Right content - Visualizations
                html.Div([
                    # Cell scatter plot
                    html.Div([
                        dcc.Graph(
                            id='cell-scatter',
                            style={'height': '60vh'},
                            config={'displayModeBar': True}
                        )
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '10px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'marginBottom': '20px'
                    }),
                    
                    # Bottom row of plots
                    html.Div([
                        # Cell count plot
                        html.Div([
                            dcc.Graph(
                                id='cell-count-plot',
                                style={'height': '40vh'},
                                config={'displayModeBar': True}
                            )
                        ], style={'width': '50%', 'padding': '10px'}),
                        
                        # Concentration heatmap
                        html.Div([
                            dcc.Graph(
                                id='concentration-heatmap',
                                style={'height': '40vh'},
                                config={'displayModeBar': True}
                            )
                        ], style={'width': '50%', 'padding': '10px'})
                    ], style={'display': 'flex', 'marginBottom': '20px'}),
                    
                    # Gene heatmap
                    html.Div([
                        dcc.Graph(
                            id='gene-heatmap',
                            style={'height': '40vh'},
                            config={'displayModeBar': True}
                        )
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '10px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                ], style={
                    'width': '75%',
                    'padding': '20px',
                    'backgroundColor': colors['background']
                })
            ], style={'display': 'flex'})
        ], style={
            'backgroundColor': colors['background'],
            'minHeight': '100vh',
            'fontFamily': 'Arial, sans-serif'
        })
        
        @self.app.callback(
            Output('cell-scatter', 'figure'),
            [Input('frame-slider', 'value'),
             Input('color-dropdown', 'value'),
             Input('size-dropdown', 'value'),
             Input('view-mode-toggle', 'value')]
        )
        def update_cell_scatter(slider_value, color_by, size_by, view_mode):
            if slider_value is None:
                return go.Figure()
            frame = initial_frames[slider_value]
            logger.info(f"Updating cell scatter for frame {frame} in {view_mode} mode")
            df = self.data_loader.get_frame_data(frame)
            
            if view_mode == '3d':
                return self._create_cell_scatter_3d(df, color_by, size_by)
            else:
                return self._create_cell_scatter_2d(df, color_by, size_by)
            
        @self.app.callback(
            Output('cell-count-plot', 'figure'),
            Input('frame-slider', 'value')
        )
        def update_cell_count_plot(slider_value):
            if slider_value is None:
                return go.Figure()
            frame = initial_frames[slider_value]
            logger.info("Updating cell count plot")
            return self._create_cell_count_plot(slider_value)
            
        @self.app.callback(
            Output('concentration-heatmap', 'figure'),
            [Input('frame-slider', 'value'),
             Input('molecule-dropdown', 'value')]
        )
        def update_concentration_heatmap(slider_value, molecule):
            if slider_value is None or not molecule:
                return go.Figure()
            frame = initial_frames[slider_value]
            logger.info(f"Updating concentration heatmap for frame {frame}, molecule {molecule}")
            values, metadata = self.data_loader.get_concentration_grid(frame, molecule)
            return self._create_concentration_heatmap(values, metadata, molecule)
            
        @self.app.callback(
            Output('gene-heatmap', 'figure'),
            Input('frame-slider', 'value')
        )
        def update_gene_heatmap(slider_value):
            if slider_value is None:
                return go.Figure()
            frame = initial_frames[slider_value]
            logger.info(f"Updating gene heatmap for frame {frame}")
            df = self.data_loader.get_frame_data(frame)
            return self._create_gene_heatmap(df)
            
    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
            port: Port number to run the server on
        """
        self.app.run(debug=debug, port=port)

def create_visualization_dashboard(filepath: str, port: int = 8050, debug: bool = False) -> None:
    """Create and run an interactive visualization dashboard.
    
    Args:
        filepath: Path to the Goo data file
        port: Port number for the dashboard server
        debug: Whether to run in debug mode
    """
    with GooDataLoader(filepath) as loader:
        visualizer = GooVisualizer(loader)
        visualizer.run_server(debug=debug, port=port) 