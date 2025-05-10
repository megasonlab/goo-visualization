"""Data loading module for Goo simulation data."""

import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Change to WARNING to reduce info messages
logger = logging.getLogger(__name__)

class GooDataLoader:
    """Class to handle loading and processing of Goo simulation data."""
    
    def __init__(self, filepath: Union[str, Path]):
        """Initialize the data loader with the path to the Goo data file.
        
        Args:
            filepath: Path to the Goo data file (.h5 format)
        """
        self.filepath = Path(filepath)
        self._file = None
        logger.warning(f"Initializing GooDataLoader with file: {filepath}")
        
    def __enter__(self):
        try:
            self._file = h5py.File(self.filepath, 'r')
            logger.warning(f"Successfully opened HDF5 file: {self.filepath}")
            return self
        except Exception as e:
            logger.error(f"Failed to open HDF5 file: {e}")
            raise
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.warning("Closed HDF5 file")
            
    def get_available_frames(self) -> List[str]:
        """Get list of available frames in the dataset.
        
        Returns:
            List of frame names
        """
        if self._file is None:
            raise RuntimeError("DataLoader must be used as a context manager")
        frames = sorted(list(self._file.keys()))
        logger.warning(f"Found {len(frames)} frames")
        return frames
            
    def get_frame_data(self, frame: str) -> pd.DataFrame:
        """Get cell data for a specific frame.
        
        Args:
            frame: Frame identifier
            
        Returns:
            DataFrame containing cell data
        """
        # logger.warning(f"Loading data for frame {frame}")
        
        try:
            frame_group = self._file[frame]
            cells_group = frame_group['cells']
            
            # Initialize lists to store data
            names = []
            locations = []
            division_frames = []
            volumes = []
            pressures = []
            sphericities = []
            aspect_ratios = []
            gene_concs = defaultdict(list)
            
            # Iterate through each cell
            for cell_name in cells_group.keys():
                cell_group = cells_group[cell_name]
                try:
                    # Get basic properties
                    names.append(cell_group.attrs['name'])
                    locations.append(cell_group['loc'][:])
                    volumes.append(cell_group['volume'][()])
                except Exception as e:
                    logger.error(f"Error loading basic properties for cell {cell_name} in frame {frame}: {e}")
                    continue
                # Try to get optional shape features
                try:
                    division_frames.append(cell_group['division_frame'][()])
                except Exception:
                    division_frames.append(np.nan)
                try:
                    sphericities.append(cell_group['sphericity'][()])
                except Exception:
                    sphericities.append(np.nan)
                try:
                    aspect_ratios.append(cell_group['aspect_ratio'][()])
                except Exception:
                    aspect_ratios.append(np.nan)
                try:
                    pressures.append(cell_group['pressure'][()])
                except Exception:
                    pressures.append(np.nan)
                # Get gene concentrations
                for key in cell_group.keys():
                    if key.startswith('gene_') and key.endswith('_conc'):
                        gene_name = key[5:-5]  # remove 'gene_' and '_conc'
                        try:
                            gene_concs[gene_name].append(cell_group[key][()])
                        except Exception:
                            gene_concs[gene_name].append(np.nan)
            
            # If no cells, return empty DataFrame
            if not names:
                return pd.DataFrame()
            
            # Create the base DataFrame
            df = pd.DataFrame({
                'name': names,
                'x': [loc[0] for loc in locations],
                'y': [loc[1] for loc in locations],
                'z': [loc[2] for loc in locations],
                'volume': volumes,
                'pressure': pressures,
                'division_frame': division_frames,
                'sphericity': sphericities,
                'aspect_ratio': aspect_ratios
            })
            
            # Add gene concentrations
            for gene_name, concs in gene_concs.items():
                # Pad with np.nan if some cells are missing this gene
                while len(concs) < len(names):
                    concs.append(np.nan)
                df[f'gene_{gene_name}'] = concs
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading frame {frame}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def get_concentration_grid(self, frame_name: str, molecule: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get concentration grid data for a specific molecule.
        
        Args:
            frame_name: Name of the frame to load
            molecule: Name of the molecule (e.g., 'mol_A')
            
        Returns:
            Tuple of (grid_values, metadata) where metadata contains grid dimensions
        """
        if self._file is None:
            raise RuntimeError("DataLoader must be used as a context manager")
            
        try:
            frame_group = self._file[frame_name]
            mol_group = frame_group["concentration_grid"][molecule]
            
            values = mol_group["values"][:]
            dimensions = mol_group["dimensions"][:]
            
            metadata = {
                "dimensions": dimensions,
                "shape": values.shape,
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
            
            logger.info(f"Successfully loaded concentration grid for {molecule} in frame {frame_name}")
            return values, metadata
            
        except Exception as e:
            logger.error(f"Error loading concentration grid for {molecule} in frame {frame_name}: {e}")
            raise

    def get_available_molecules(self, frame_name: str) -> List[str]:
        """Get list of available molecules in the concentration grid.
        
        Args:
            frame_name: Name of the frame to check
            
        Returns:
            List of molecule names
        """
        if self._file is None:
            raise RuntimeError("DataLoader must be used as a context manager")
            
        try:
            frame_group = self._file[frame_name]
            molecules = list(frame_group["concentration_grid"].keys())
            logger.info(f"Found {len(molecules)} molecules in frame {frame_name}: {molecules}")
            return molecules
            
        except Exception as e:
            logger.error(f"Error getting molecules for frame {frame_name}: {e}")
            raise

    def get_available_genes(self, frame_name: str) -> List[str]:
        """Get list of available genes in the dataset.
        
        Args:
            frame_name: Name of the frame to check
            
        Returns:
            List of gene names
        """
        if self._file is None:
            raise RuntimeError("DataLoader must be used as a context manager")
            
        try:
            df = self.get_frame_data(frame_name)
            gene_cols = [col for col in df.columns if col.startswith("gene_")]
            genes = [col[5:] for col in gene_cols]  # Remove 'gene_' prefix
            logger.info(f"Found {len(genes)} genes in frame {frame_name}: {genes}")
            return genes
            
        except Exception as e:
            logger.error(f"Error getting genes for frame {frame_name}: {e}")
            raise 