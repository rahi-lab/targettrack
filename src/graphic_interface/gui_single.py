import os
import rpyc 
from rpyc.utils.classic import obtain
import time
import numpy as np
import threading
import queue

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QTimer
from . import gui
from ..datasets_code.DataSet import DataSet
from .. import main_controller
from logging_config import setup_logger

logger = setup_logger(__name__)

class ChunkCache:
    """Thread-safe cache for dataset chunks with LRU eviction"""
    def __init__(self, max_size_mb: int = 1024):
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self._cache = OrderedDict()  # {cache_key: (data, size)}
        self._current_size = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used) 
                data, size = self._cache.pop(key)
                self._cache[key] = (data, size)
                return data
        return None
    
    def put(self, key: str, data: np.ndarray):
        size = data.nbytes
        with self._lock:
            # Remove old entry if exists
            if key in self._cache:
                _, old_size = self._cache.pop(key)
                self._current_size -= old_size
            
            # Evict until we have space
            while self._current_size + size > self.max_size and self._cache:
                old_key, (_, old_size) = self._cache.popitem(last=False)
                self._current_size -= old_size
            
            # Only store if it fits
            if size <= self.max_size:
                self._cache[key] = (data, size)
                self._current_size += size

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._current_size = 0

class LazyDataset:
    """Represents a remote HDF5 dataset with lazy loading and caching"""
    
    def __init__(self, conn, file_id: str, path: str, cache: ChunkCache):
        self.conn = conn
        self.file_id = file_id
        self.path = path
        self.cache = cache
        
        # Get dataset info
        info = obtain(self.conn.root.get_dataset_info(file_id, path))
        if info is None:
            self.exists = False
            return
            
        self.exists = True
        self.shape = obtain(info['shape'])
        self.dtype = np.dtype(obtain(info['dtype']))
        self.chunks = obtain(info['chunks'])

    

    def __getitem__(self, key):
        if not self.exists:
            return None

        # Convert key to proper slices
        slices = self._normalize_slices(key)
        
        # Generate cache key
        cache_key = f"{self.path}:{self._slice_key(slices)}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Load from server
        slice_info = [(s.start, s.stop, s.step) for s in slices]
        try:
            # Fetch the data from the server
            remote_data = self.conn.root.get_dataset_chunk(self.file_id, self.path, slice_info)
            
            # Materialize the remote data using rpyc's obtain
            data = obtain(remote_data)
            
            # Cache result if not too large (< 10MB)
            if data.nbytes < 10 * 1024 * 1024:
                self.cache.put(cache_key, data)
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching dataset chunk for {self.path}, slices {slice_info}: {e}")
            return None

    def _normalize_slices(self, key) -> Tuple[slice, ...]:
        """Convert input key into proper slice objects"""
        if isinstance(key, tuple):
            slices = []
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    slices.append(k)
                elif isinstance(k, int):
                    slices.append(slice(k, k + 1))
                else:
                    raise ValueError(f"Invalid slice type: {type(k)}")
        else:
            if isinstance(key, slice):
                slices = [key]
            elif isinstance(key, int):
                slices = [slice(key, key + 1)]
            else:
                raise ValueError(f"Invalid key type: {type(key)}")
        
        # Pad with full slices if needed
        while len(slices) < len(self.shape):
            slices.append(slice(None))
            
        return tuple(slices)

    def _slice_key(self, slices: Tuple[slice, ...]) -> str:
        """Convert slice objects to a string for caching"""
        return str([(s.start, s.stop, s.step) for s in slices])

class RemoteConnection:
    """Handles remote server connection through SSH tunnel with retry logic"""
    def __init__(self, port, max_retries=3, retry_delay=2):
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connect()

    def connect(self):
        """Establish connection with retry logic"""
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                logger.info(f"Connection attempt {attempt + 1}/{self.max_retries}")
                
                self.conn = rpyc.connect(
                    'localhost',
                    self.port,
                    config={
                        'sync_request_timeout': 3600,  # 1 hour timeout
                        'allow_pickle': True,
                    }
                )
                self.root = self.conn.root
                
                if not self.validate_connection():
                    raise ConnectionError("Failed to validate connection")
                
                # Get server info
                server_info = obtain(self.root.get_server_info())
                if 'error' in server_info:
                    logger.warning(f"Server status: {server_info['error']}")
                else:
                    device_type = "GPU" if server_info.get('cuda_available', False) else "CPU"
                    logger.info(f"Server running on {device_type}")
                    if device_type == "GPU" and 'device_name' in server_info:
                        logger.info(f"Server GPU: {server_info['device_name']}")
                
                logger.info("Successfully connected to remote server")
                return
                
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(f"Connection attempt failed: {str(e)}")
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    
        logger.error(f"Failed to connect after {self.max_retries} attempts")
        raise last_error

    def validate_connection(self):
        """Verify the connection is working properly"""
        try:
            return obtain(self.root.check_connection())
        except:
            return False
            
    def ping(self):
        """Test if connection is still alive"""
        try:
            return obtain(self.validate_connection())
        except:
            return False

    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

class RemoteH5File(DataSet):
    """HDF5 file proxy that loads data on demand from remote server"""
    
    def __init__(self, connection, filepath: str, node: Optional[str] = None):
        super().__init__()
        self.conn = connection
        self.filepath = filepath
        self.node = node
        
        # Initialize caches
        self._dataset_cache = {}  # {path: LazyDataset}
        self._chunk_cache = ChunkCache()
        self._attr_cache = {}  # Cache for small attributes
        
        # Check file exists
        if not self.conn.root.check_file_exists(filepath):
            raise FileNotFoundError(f"Remote file not found: {filepath}")
        
        # Open file and get metadata
        result = self.conn.root.open_h5(filepath)
        self.file_id = result['file_id']
        self.structure = result['structure']
        self._attrs = result['attributes']
                
        self._prefetch_queue = queue.Queue()
        self._stop_event = threading.Event()  # Stop signal
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def _get_dataset(self, path: str) -> Optional[LazyDataset]:
        """Get or create LazyDataset for given path"""
        if path not in self._dataset_cache:
            dataset = LazyDataset(
                self.conn,
                self.file_id,
                path,
                self._chunk_cache
            )
            if dataset.exists:
                self._dataset_cache[path] = dataset
            else:
                return None
        return self._dataset_cache[path]

    def __getitem__(self, path: str) -> Any:
        """Get dataset or attribute"""
        try:
            # Check attribute cache
            if path in self._attr_cache:
                return self._attr_cache[path]
            
            # Get dataset
            dataset = self._get_dataset(path)
            if dataset is None:
                if self._is_expected_missing(path):
                    return False
                raise KeyError(f"Dataset {path} not found")
                
            # Load full dataset
            result = dataset[:]
            return result
            
        except Exception as e:
            if not self._is_expected_missing(path):
                logger.warn(f"Error getting dataset {path}: {str(e)}")
            return None

    def _get_frame(self, t: int, col: str = "red") -> Optional[np.ndarray]:
      """
      Get frame and prefetch ±10 frames.
      Handles errors gracefully.
      """
      try:
          # Fetch the current frame
          dataset = self._get_dataset(f"{t}/frame")
          if dataset is None:
              logger.warning(f"Dataset for frame {t} not found.")
              return None

          frame = obtain(dataset[:])
          if frame is None:
              logger.warning(f"Frame data for {t} could not be retrieved.")
              return None

          # Extract the requested channel
          channel_data = frame[0] if col == "red" else frame[1]

          # Prefetch surrounding frames

          return channel_data

      except Exception as e:
          logger.error(f"Error getting frame {t}: {str(e)}")
          return None
    def _prefetch_worker(self):
      """Background worker to process prefetch requests."""
      while not self._stop_event.is_set():
          try:
              # Use timeout to periodically check for the stop event
              t, col = self._prefetch_queue.get(timeout=1)
              if t is None:  # Sentinel to break out of loop
                  break
              self._get_frame(t, col)  # Load the frame to cache it
              self._prefetch_queue.task_done()
          except queue.Empty:
              # Timeout reached; loop continues, checking for stop signal
              continue
          except Exception as e:
              logger.warning(f"Error during background prefetch: {e}")

    def prefetch_frames(self, t: int, col: str = "red"):
      """Queue ±10 frames for background prefetching."""
      start = max(0, t - 10)
      end = min(self.frame_num, t + 11)
      for frame in range(start, end):
          self._prefetch_queue.put((frame, col))

    def stop_prefetching(self, immediate: bool = False):
      """
      Stop the background prefetching thread.
      Args:
          immediate (bool): If True, force the thread to stop immediately.
      """
      if immediate:
          self._stop_event.set()  # Signal the thread to stop immediately
      else:
          self._prefetch_queue.put((None, None))  # Graceful stop using sentinel value

      self._prefetch_thread.join()
      logger.info("Prefetching thread stopped.")
    def _prefetch_frames(self, t: int, col: str = "red"):
      """
      Prefetch ±10 frames and cache them.
      """
      start = max(0, t - 10)  # Prevent accessing before the first frame
      end = min(self.frame_num, t + 11)  # Prevent accessing beyond the last frame
      logger.info(f"Prefetching frames {start} to {end - 1} ({col})")
      for i in range(start, end):
        try:
            # Generate a cache key for the frame
            cache_key = f"{i}/frame:{col}"

            # Skip if the frame is already cached
            if self._chunk_cache.get(cache_key) is not None:
                continue

            # Fetch the frame dataset
            dataset = self._get_dataset(f"{i}/frame")
            if dataset is None:
                continue

            frame = obtain(dataset[:])
            if frame is None:
                logger.debug(f"Skipping prefetch for frame {i}, data unavailable.")
                continue

            # Cache the requested channel
            channel_data = frame[0] if col == "red" else frame[1]
            self._chunk_cache.put(cache_key, channel_data)

        except Exception as e:
            logger.warning(f"Error during prefetch for frame {i}: {e}")

    def _get_mask(self, t: int) -> Optional[np.ndarray]:
        """Get mask with error handling"""
        try:
            mask_key = "coarse_mask" if self.coarse_seg_mode else "mask"
            dataset = self._get_dataset(f"{t}/{mask_key}")
            
            if dataset is None:
                return False
                
            mask = obtain(dataset[:])
            return False if mask is None else mask
            
        except Exception as e:
            logger.error(f"Error getting mask {t}: {str(e)}")
            return False
    @property
    def pointdat(self):
        """Fetch the point data from the dataset."""
        try:
            dataset = self._get_dataset("/pointdat")
            if dataset is not None:
                return obtain(dataset[:])  # Fetch all point data
            else:
                logger.warning("Point data not found in the file.")
                return np.full((self.frame_num, self.nb_neurons + 1, 3), np.nan)  # Default empty data
        except Exception as e:
            logger.error(f"Error fetching point data: {str(e)}")
            return None

    def available_NNdats(self) -> List[str]:
        """Return a list of available NN datasets"""
        nn_datasets = []
        for key in self.structure.get('net', {}).keys():
            if 'NN' in key:
                nn_datasets.append(key)
        return nn_datasets
    
    def get_available_methods(self) -> List[str]:
        """
        Return a list of available methods (keys starting with 'helper_') in the remote dataset.
        """
        try:
            # Fetch relevant keys directly from the server
            helper_keys = self.conn.root.get_helper_keys(self.file_id, prefix="helper_")
            return [key[7:] for key in helper_keys]  # Remove 'helper_' prefix
        except Exception as e:
            logger.error(f"Error retrieving available methods: {str(e)}")
            return []

    
    def get_NN_mask(self, t: int, NN_key: str) -> Optional[np.ndarray]:
        """Get neural network prediction mask"""
        try:
            dataset = self._get_dataset(f"net/{NN_key}/{t}/predmask")
            if dataset is None:
                return False
                
            return dataset[:]
            
        except Exception as e:
            logger.error(f"Error getting NN mask {t}: {str(e)}")
            return False
    
    def set_point_data(self):
        """
        Set the pointdat on the remote dataset.
        """
        try:
            # Check if point_data is set and validate consistency
            if not hasattr(self, 'point_data') or self.point_data is None:
                self.point_data = True
            elif not self.point_data:
                raise ValueError("Masks and point data would interfere.")

            # Write data to the dataset
            self.conn.root.write_dataset_point_data(self.file_id, "/point_data", [self.point_data])
            logger.info("Point data successfully updated in the remote dataset.")

        except Exception as e:
            logger.error(f"Error setting pointdat: {str(e)}")
            raise
    def send_pointdat_patch_to_server(self, frame, neuron, coord):
            """Send only the updated data to the remote server."""
            try:
                patch_data = {"frame": frame, "neuron": neuron, "coord": coord}
                self.conn.root.update_dataset_pointdat(self.file_id, path="/pointdat", patch_data=patch_data)
                logger.info(f"Patch sent for frame {frame}, neuron {neuron}: {coord}")
            except Exception as e:
                logger.error(f"Failed to send patch: {e}")
    
    def send_ci_int_patch_to_server(self, frame, settings):
        """Send only the updated data to the remote server."""
        try:
            patch_data = {"frame": frame, "settings": settings}
            self.conn.root.exposed_update_dataset_ci_int_t(self.file_id, patch_data, settings)
            logger.info(f"Patch sent for frame {frame}")
        except Exception as e:
            logger.error(f"Failed to send patch: {e}")

    def _is_expected_missing(self, path: str) -> bool:
        """Check if this is an expected missing path"""
        expected = ['/mask', '/coarse_mask', '/transform1', 
                   '/transfo_matrix', '/high', '/predmask']
        return any(path.endswith(suffix) for suffix in expected)

    def clear_cache(self):
        """Clear all caches"""
        self._dataset_cache.clear()
        self._chunk_cache.clear()
        self._attr_cache.clear()

    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'file_id'):
                self.conn.root.close_h5(self.file_id)
        finally:
            self.clear_cache()

    # Properties that use cached attributes
    @property
    def name(self) -> str:
        return self._attrs.get("name", os.path.splitext(os.path.basename(self.filepath))[0])

    @property 
    def path_from_GUI(self) -> str:
        return self.filepath

    @property
    def nb_channels(self) -> Optional[int]:
        return self._attrs.get("C")

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        return (
            obtain(self._attrs.get("W")),
            obtain(self._attrs.get("H")), 
            obtain(self._attrs.get("D"))
        )

    @property
    def nb_neurons(self) -> int:
        return obtain(self._attrs.get("N_neurons", 0))
    
    @nb_neurons.setter
    def nb_neurons(self, value: int):
        self._attrs["N_neurons"] = value

    @property
    def neuron_presence(self) -> Dict[int, List[int]]:
        """Get dictionary mapping neuron IDs to frames where they are present"""
        return self._attrs.get("neuron_presence", None)
    
    @neuron_presence.setter
    def neuron_presence(self, value: Dict[int, List[int]]):
        self._attrs["neuron_presence"] = value
   
    @property
    def ca_act(self) -> Optional[np.ndarray]:
        """
        Get calcium activity data (ci_int) from the remote dataset.
        """
        try:
            # if "/ci_int" not in self.structure:
            #     return None
            dataset = self._get_dataset("/ci_int")
            if dataset is None:
                return None
            return obtain(dataset[:])
        except Exception as e:
            logger.error(f"Error retrieving 'ci_int': {str(e)}")
            return None

    @ca_act.setter
    def ca_act(self, ca_activity: np.ndarray):
        """
        Set the ci_int data on the remote server.
        WARNING: The array of calcium intensity values is NOT in 1-indexing for neurons, 
        and the first dimension is neurons, the second is frames.

        Args:
            ca_activity: A numpy array with calcium activity data (neurons x frames).
        """
        try:
            # Validate shape consistency
            assert self.nb_neurons == ca_activity.shape[0], "Mismatch in number of neurons"
            assert len(self.frames) == ca_activity.shape[1], "Mismatch in number of frames"
            
            # Remove existing dataset if it exists
            if "/ci_int" in self.structure:
                self.conn.root.delete_dataset(self.file_id, "/ci_int")
            # Check if ca_activity is all NaNs
            if np.isnan(ca_activity).all():
                logger.warning("Skipping write: ca_activity is all NaNs")
                return

            # Write the data
            self.conn.root.create_dataset(
                self.file_id,
                "/ci_int",
                shape=ca_activity.shape,
                dtype="float32",
                compression=None
            )
            # self.conn.root.write_dataset(self.file_id, "/ci_int", ca_activity)
            # Update the structure locally
            logger.info("Calcium activity data updated on remote server")
        except Exception as e:
            logger.error(f"Error setting 'ci_int': {str(e)}")
            raise
    @property
    def frame_num(self) -> int:
        return obtain(self._attrs.get("T", 0))

    @property
    def frames(self) -> List[int]:
        return list(range(obtain(self._attrs.get("T", 0))))

    @property
    def real_neurites(self) -> List[int]:
        return list(range(1, obtain(self.nb_neurons) + 1))

    def save(self):
        """Changes are handled server-side"""
        pass

def loaddict(fn: str) -> Dict[str, str]:
    """Parse settings file into dictionary"""
    set_dict = {}
    with open(fn, "r") as f:
        for line in f:
            if line.strip():
                try:
                    key, value = line.strip().split("=")
                    set_dict[key] = value
                except ValueError:
                    logger.warning(f"Invalid line in settings file: {line}")
    return set_dict

class gui_single:
    def __init__(self, dataset_path: str, port: int, tunneled: bool = False, node: Optional[str] = None):
        """
        Initialize GUI with either local or remote dataset
        
        Args:
            dataset_path: Path to h5 file
            tunneled: Whether to use SSH-tunneled connection
            node: Optional compute node name for remote connections
        """
        super().__init__()
        
        # Load settings
        self.settings = loaddict(os.path.join("src", "parameters", "current_settings.dat"))
        self.connection = None

        try:
            if tunneled:
                # Remote mode via SSH tunnel
                logger.info(f"Connecting via SSH tunnel to {node}...")
                self.connection = RemoteConnection(port=port)
                self.dataset = RemoteH5File(
                    self.connection,
                    dataset_path,
                    node=node
                )
                logger.info("Connected to remote dataset")
            else:
                # Local mode - use h5Data directly
                from ..datasets_code.h5Data import h5Data
                self.dataset = h5Data(dataset_path)
                logger.info(f"Loaded local dataset: {dataset_path}")

            # Initialize controller and UI
            self.controller = main_controller.Controller(self.dataset, self.settings)
            self.initUI()
            self.controller.set_up()

        except Exception as e:
            if self.connection:
                self.connection.close()
            logger.error(f"Failed to initialize GUI: {str(e)}", exc_info=True)
            raise

    def initUI(self):
        """Initialize the user interface"""
        self.app = QApplication([])
        self.app.setWindowIcon(QIcon(os.path.join("src", "Images", "icon.png")))
        logger.debug("initUI")
        screen = self.app.primaryScreen()
        size = screen.size()
        self.settings["screen_w"] = size.width()
        self.settings["screen_h"] = size.height()
        self.fps = float(self.settings["fps"])

        self.gui = gui.gui(self.controller, self.settings, self)
        self.gui.show()

    def closeEvent(self, event):
        """Handle application closing"""
        
        # Stop the prefetching thread
        self.controller.data.stop_prefetching()
        reply = QMessageBox.question(
            self.gui, "Closing",
            "Save remaining annotations and Neural Networks?",
            QMessageBox.Save | QMessageBox.Close | QMessageBox.Cancel,
            QMessageBox.Save)
            
        try:
            if reply == QMessageBox.Close:
                ok, msg = self.gui.close("force", "")
                if ok:
                    self.dataset.close()
                    if self.connection:
                        self.connection.close()
                    logger.info(f"Closing:\n{msg}")
                    event.accept()
                else:
                    errdial = QErrorMessage()
                    errdial.showMessage('Error during close. Contact support.')
                    errdial.exec_()
                    
            elif reply == QMessageBox.Save:
                ok, msg = self.gui.close("save", "")
                if ok:
                    self.dataset.close()
                    if self.connection:
                        self.connection.close()
                    logger.info(f"Saving and closing:\n{msg}")
                    event.accept()
                else:
                    errdial = QErrorMessage()
                    errdial.showMessage(f'Save failed:\n{msg}')
                    errdial.exec_()
                    event.ignore()
                    
            elif reply == QMessageBox.Cancel:
                event.ignore()
                
        except Exception as e:
            logger.error(f"Error during close: {str(e)}")
            event.ignore()

    def handle_server_disconnect(self):
        """Handle server disconnection"""
        try:
            # Try to reconnect
            if self.connection and not self.connection.ping():
                logger.warning("Server connection lost, attempting to reconnect...")
                self.connection.connect()
                
                # Reinitialize dataset after reconnection
                self.dataset = RemoteH5File(
                    self.connection,
                    self.dataset.filepath,
                    node=self.dataset.node
                )
                self.controller.data = self.dataset
                
                logger.info("Successfully reconnected to server")
                
        except Exception as e:
            logger.error(f"Failed to reconnect to server: {str(e)}")
            
            # Show error dialog
            errdial = QErrorMessage()
            errdial.showMessage('Lost connection to server. Please save your work and restart the application.')
            errdial.exec_()

    def periodic_connection_check(self):
        """Periodically check server connection"""
        if self.connection:
            if not self.connection.ping():
                self.handle_server_disconnect()
                
