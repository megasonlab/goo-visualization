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
logging.basicConfig(level=logging.WARNING)  # Change to WARNING to reduce info messages
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
        logger.warning("Calculating global ranges across all frames...")
        frames = self.data_loader.get_available_frames()
        
        # Initialize with extreme values
        ranges = {
            'volume': {'min': float('inf'), 'max': float('-inf')},
            'pressure': {'min': float('inf'), 'max': float('-inf')},
            'sphericity': {'min': float('inf'), 'max': float('-inf')},
            'aspect_ratio': {'min': float('inf'), 'max': float('-inf')},
            'division_frame': {'min': float('inf'), 'max': float('-inf')}
        }
        
        for frame in frames:
            df = self.data_loader.get_frame_data(frame)
            # Update ranges for each feature
            for feature in ranges.keys():
                if feature in df.columns:
                    ranges[feature]['min'] = min(ranges[feature]['min'], df[feature].min())
                    ranges[feature]['max'] = max(ranges[feature]['max'], df[feature].max())
            
        self.global_ranges = ranges
        logger.warning(f"Global ranges calculated: {self.global_ranges}")
        
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
                showlegend=False,
                height=400,  # Set consistent height
                margin=dict(l=50, r=20, t=50, b=50),  # Add margin for axis labels
                xaxis=dict(
                    range=[0, len(frames)],
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    range=[0, max(cell_counts) * 1.1],  # Add 10% padding
                    showgrid=True,
                    gridcolor='lightgray'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cell count plot: {e}")
            return go.Figure()

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
        
    def _create_gene_heatmap(self, df: pd.DataFrame, frame_idx: int, selected_cell: str = None) -> go.Figure:
        """Create an interactive line plot of gene expression data over time.
        
        Args:
            df: DataFrame containing cell data
            frame_idx: Current frame index
            selected_cell: Name of the cell to display (if None, show all cells)
            
        Returns:
            Plotly figure object
        """
        print("\n=== Starting gene expression plot creation ===")
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input DataFrame columns: {df.columns.tolist()}")
        
        # Get gene columns - looking for columns that start with 'gene_'
        gene_cols = [col for col in df.columns if col.startswith('gene_')]
        if not gene_cols:
            print("WARNING: No gene columns found in the data")
            # Create a figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="Gene data not available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title='Gene Expression Levels Over Time',
                xaxis_title='Frame Number',
                yaxis_title='Expression Level',
                showlegend=True,
                height=600
            )
            return fig
            
        print(f"\nFound {len(gene_cols)} gene columns: {gene_cols}")
        print(f"Sample gene data for first cell:")
        for gene in gene_cols:
            print(f"  {gene}: {df[gene].iloc[0]}")
        
        # Collect gene expression data across frames
        frames = self.data_loader.get_available_frames()
        print(f"\nTotal number of frames: {len(frames)}")
        
        # Dictionary to store gene data for each cell across frames
        cell_gene_data = {}
        cell_division_frames = {}
        
        print("\nCollecting gene data across frames:")
        for i, frame in enumerate(frames):
            if i % 50 == 0:  # Print progress every 50 frames
                print(f"Processing frame {i}/{len(frames)}")
            frame_df = self.data_loader.get_frame_data(frame)
            if not frame_df.empty:
                # For each cell in this frame
                for _, row in frame_df.iterrows():
                    cell_name = row['name']
                    if cell_name not in cell_gene_data:
                        cell_gene_data[cell_name] = {gene: [] for gene in gene_cols}
                        cell_division_frames[cell_name] = row['division_frame']
                    # Add gene values for this cell
                    for gene in gene_cols:
                        cell_gene_data[cell_name][gene].append(row[gene])
        
        # Create figure
        fig = go.Figure()
        
        # Add line plot for each cell's gene expression
        print("\nAdding traces to plot:")
        cells_to_plot = [selected_cell] if selected_cell else cell_gene_data.keys()
        for cell_name in cells_to_plot:
            if cell_name in cell_gene_data:
                gene_values = cell_gene_data[cell_name]
                division_frame = cell_division_frames[cell_name]
                
                # Add a line for each gene
                for gene in gene_cols:
                    gene_name = gene[5:]  # Remove 'gene_' prefix
                    values = gene_values[gene]
                    
                    # Create x-axis values starting from division frame
                    x_values = list(range(division_frame, division_frame + len(values)))
                    
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=values,
                        name=f"{gene_name}",
                        mode='lines',
                        line=dict(width=2),
                        showlegend=True
                    ))
        
        # Add vertical line for current frame
        fig.add_vline(
            x=frame_idx,
            line_dash="dash",
            line_color="red",
            opacity=0.5
        )
        
        fig.update_layout(
            title=f'Gene Expression Levels Over Time{" for " + selected_cell if selected_cell else ""}',
            xaxis_title='Frame Number',
            yaxis_title='Expression Level',
            showlegend=True,
            height=600,  # Make the plot taller
            legend=dict(
                groupclick="toggleitem",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )
        
        print("\n=== Finished creating gene expression plot ===")
        return fig
        
    def _create_volume_plot(self, current_frame_idx: int = 0) -> go.Figure:
        """Create a plot showing cell volumes over time, starting from their division frames.
        
        Args:
            current_frame_idx: Index of the currently selected frame
            
        Returns:
            Plotly figure object
        """
        try:
            frames = self.data_loader.get_available_frames()
            cell_volumes = {}  # Dictionary to store volume data for each cell
            max_volume = 0  # Track maximum volume for y-axis range
            
            # First pass: collect division frames and initial volumes
            for frame in frames:
                df = self.data_loader.get_frame_data(frame)
                for _, row in df.iterrows():
                    cell_name = row['name']
                    if cell_name not in cell_volumes:
                        cell_volumes[cell_name] = {
                            'division_frame': row['division_frame'],
                            'volumes': []
                        }
                    max_volume = max(max_volume, row['volume'])
            
            # Second pass: collect volume data for each cell
            for frame in frames:
                df = self.data_loader.get_frame_data(frame)
                for cell_name in cell_volumes:
                    cell_data = df[df['name'] == cell_name]
                    if not cell_data.empty:
                        cell_volumes[cell_name]['volumes'].append(cell_data['volume'].iloc[0])
                    else:
                        cell_volumes[cell_name]['volumes'].append(None)
            
            # Create the plot
            fig = go.Figure()
            
            # Add a line for each cell
            for cell_name, data in cell_volumes.items():
                # If division frame is NaN, start from frame 0
                start_frame = 0 if np.isnan(data['division_frame']) else int(data['division_frame'])
                
                # Find the last frame with valid volume data
                valid_volumes = [(i, v) for i, v in enumerate(data['volumes']) if v is not None]
                if not valid_volumes:
                    continue
                    
                # Get the range of frames with valid data
                start_idx = valid_volumes[0][0]
                end_idx = valid_volumes[-1][0] + 1
                
                # Create x-axis values and get corresponding volumes
                x_values = list(range(start_idx, end_idx))  # Use frame indices directly
                volumes = [data['volumes'][i] for i in range(start_idx, end_idx)]
                
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=volumes,
                    mode='lines',
                    line=dict(width=2, color='rgba(0, 0, 255, 0.3)'),  # Semi-transparent blue
                    name=cell_name,  # Keep cell name in trace
                    showlegend=False  # Hide from legend
                ))
            
            # Add vertical line for current frame
            fig.add_vline(
                x=current_frame_idx,
                line_dash="dash",
                line_color="red",
                opacity=0.5
            )
            
            # Update layout with proper axis ranges and labels
            fig.update_layout(
                title='Cell Volumes Over Time',
                xaxis_title='Frame Number',
                yaxis_title='Volume',
                showlegend=False,  # Hide the legend
                height=400,  # Set consistent height
                margin=dict(l=50, r=20, t=50, b=50),  # Add margin for axis labels
                xaxis=dict(
                    range=[0, len(frames)],
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    range=[0, max_volume * 1.1],  # Add 10% padding to max volume
                    showgrid=True,
                    gridcolor='lightgray'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating volume plot: {e}")
            return go.Figure()

    def _setup_dashboard(self):
        """Set up the interactive dashboard layout and callbacks."""
        # Get initial frames
        initial_frames = self.data_loader.get_available_frames()
        initial_options = [{'label': frame, 'value': frame} for frame in initial_frames]
        initial_value = initial_frames[0] if initial_frames else None
        
        # Get all unique cell names across all frames
        all_cells = set()
        for frame in initial_frames:
            df = self.data_loader.get_frame_data(frame)
            all_cells.update(df['name'].unique())
        cell_options = [{'label': name, 'value': name} for name in sorted(all_cells)]
        print(f"Found {len(cell_options)} unique cells for dropdown")
        
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
                                    {'label': 'Pressure', 'value': 'pressure'},
                                    {'label': 'Sphericity', 'value': 'sphericity'},
                                    {'label': 'Aspect Ratio', 'value': 'aspect_ratio'},
                                    {'label': 'Division Frame', 'value': 'division_frame'}
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
                                    {'label': 'Pressure', 'value': 'pressure'},
                                    {'label': 'Sphericity', 'value': 'sphericity'},
                                    {'label': 'Aspect Ratio', 'value': 'aspect_ratio'},
                                    {'label': 'Division Frame', 'value': 'division_frame'}
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
                        
                        # Volume plot
                        html.Div([
                            dcc.Graph(
                                id='volume-plot',
                                style={'height': '40vh'},
                                config={'displayModeBar': True}
                            )
                        ], style={'width': '50%', 'padding': '10px'})
                    ], style={'display': 'flex', 'marginBottom': '40px'}),  # Increased bottom margin
                    
                    # Gene expression plot with cell selection
                    html.Div([
                        html.Div([
                            html.Label('Select Cell:', 
                                style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}
                            ),
                            dcc.Dropdown(
                                id='cell-select-dropdown',
                                options=cell_options,
                                value=cell_options[0]['value'] if cell_options else None,
                                style={'marginBottom': '15px', 'width': '100%'}
                            )
                        ], style={'width': '300px', 'marginBottom': '20px'}),  # Increased bottom margin
                        dcc.Graph(
                            id='gene-heatmap',
                            style={'height': '40vh'},
                            config={'displayModeBar': True}
                        )
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '30px',  # Increased padding
                        'borderRadius': '10px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'marginTop': '20px'  # Added top margin
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
            [Input('frame-slider', 'value'),
             Input('cell-select-dropdown', 'value')]
        )
        def update_gene_heatmap(slider_value, selected_cell):
            if slider_value is None or selected_cell is None:
                return go.Figure()
            frame = initial_frames[slider_value]
            logger.info(f"Updating gene heatmap for frame {frame}, cell {selected_cell}")
            df = self.data_loader.get_frame_data(frame)
            return self._create_gene_heatmap(df, slider_value, selected_cell)
            
        @self.app.callback(
            Output('volume-plot', 'figure'),
            Input('frame-slider', 'value')
        )
        def update_volume_plot(slider_value):
            if slider_value is None:
                return go.Figure()
            logger.info("Updating volume plot")
            return self._create_volume_plot(slider_value)
            
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