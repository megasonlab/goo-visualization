"""Data loading module for Goo simulation data."""

import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
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
        logger.info(f"Initializing GooDataLoader with file: {filepath}")
        
    def __enter__(self):
        try:
            self._file = h5py.File(self.filepath, 'r')
            logger.info(f"Successfully opened HDF5 file: {self.filepath}")
            return self
        except Exception as e:
            logger.error(f"Failed to open HDF5 file: {e}")
            raise
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info("Closed HDF5 file")
            
    def get_available_frames(self) -> List[str]:
        """Get list of available frames in the dataset.
        
        Returns:
            List of frame names
        """
        if self._file is None:
            raise RuntimeError("DataLoader must be used as a context manager")
        frames = sorted(list(self._file.keys()))
        logger.info(f"Found {len(frames)} frames: {frames}")
        return frames
            
    def get_frame_data(self, frame_name: str) -> pd.DataFrame:
        """Load data for a specific frame into a pandas DataFrame.
        
        Args:
            frame_name: Name of the frame to load (e.g., 'frame_001')
            
        Returns:
            DataFrame containing cell data for the specified frame
        """
        if self._file is None:
            raise RuntimeError("DataLoader must be used as a context manager")
            
        try:
            frame_group = self._file[frame_name]
            cells_group = frame_group["cells"]
            logger.info(f"Loading data for frame {frame_name} with {len(cells_group)} cells")
            
            # Initialize data collection
            data = {
                "cell_id": [],
                "name": [],
                "x": [], "y": [], "z": [],
                "volume": [],
                "pressure": []
            }
            gene_concs = defaultdict(list)
            mol_concs = defaultdict(list)
            
            # Process each cell
            for cell_id, cell_group in cells_group.items():
                # Basic properties
                data["cell_id"].append(cell_id)
                cell_name = cell_group.attrs["name"]
                logger.info(f"Loading cell {cell_id} with name: {cell_name}")
                data["name"].append(cell_name)
                loc = cell_group["loc"][:]
                data["x"].append(loc[0])
                data["y"].append(loc[1])
                data["z"].append(loc[2])
                data["volume"].append(cell_group["volume"][()])
                try:
                    data["pressure"].append(cell_group["pressure"][()])
                except KeyError:
                    data["pressure"].append(np.nan)
                    
                # Gene and molecule concentrations
                for key in cell_group.keys():
                    if key.startswith("gene_") and key.endswith("_conc"):
                        gene_name = key[5:-5]
                        gene_concs[gene_name].append(cell_group[key][()])
                    elif key.startswith("mol_") and key.endswith("_conc"):
                        mol_name = key[4:-5]
                        mol_concs[mol_name].append(cell_group[key][()])
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Add gene and molecule concentrations
            for gene_name, concs in gene_concs.items():
                if len(concs) == len(df):  # Only add if lengths match
                    df[f"gene_{gene_name}"] = concs
            for mol_name, concs in mol_concs.items():
                if len(concs) == len(df):  # Only add if lengths match
                    df[f"{mol_name}_conc"] = concs
                
            logger.info(f"Successfully loaded frame data with {len(df)} cells")
            return df
            
        except Exception as e:
            logger.error(f"Error loading frame {frame_name}: {e}")
            raise

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