import rpyc
import logging
from pathlib import Path
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('points_client')

class PointsTrainingClient:
    def __init__(self, host='localhost', port=18861):
        """
        Initialize connection to points training server
        
        Args:
            host: Hostname of server
            port: Port number
        """
        try:
            self.conn = rpyc.connect(
                host, 
                port,
                config={
                    'sync_request_timeout': 3600  # 1 hour timeout
                }
            )
            self.root = self.conn.root
            
            # Test connection and GPU
            gpu_info = self.root.get_gpu_info()
            if 'error' in gpu_info:
                logger.warning(f"Server GPU status: {gpu_info['error']}")
            else:
                logger.info(f"Server GPU: {gpu_info['device_name']}")
                
        except Exception as e:
            logger.error(f"Failed to connect to server: {str(e)}")
            raise
    
    def train_points_network(self, h5_path, network_name, run_name, **training_params):
        """
        Train points prediction network on server
        
        Args:
            h5_path: Path to h5 file
            network_name: Name of network architecture to use
            run_name: Name for this training run
            training_params: Additional training parameters including:
                - num_epochs
                - batch_size
                - learning_rate
                - num_workers
                - training_frames
                - validation_frames
            
        Returns:
            dict with training results including loss history
        """
        logger.info(f"Starting points training run {run_name}")
        
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        
        try:
            # Create temporary copy of h5 file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                shutil.copy2(h5_path, tmp.name)
                tmp_path = tmp.name
            
            # Start training on server
            results = self.root.train_points_network(
                tmp_path,
                network_name,
                run_name,
                training_params
            )
            
            if results['success']:
                # Copy results back
                shutil.copy2(tmp_path, h5_path)
                logger.info(f"Training completed for {run_name}")
                return results
            else:
                logger.error(f"Training failed: {results['error']}")
                raise RuntimeError(results['error'])
                
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
        finally:
            # Cleanup temporary file
            if 'tmp_path' in locals():
                try:
                    Path(tmp_path).unlink()
                except:
                    pass

def test_connection(host='localhost', port=18861):
    """Test connection to server"""
    try:
        client = PointsTrainingClient(host, port)
        gpu_info = client.root.get_gpu_info()
        logger.info(f"Connected to server, GPU info: {gpu_info}")
        return True
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    else:
        host = 'localhost'
        
    success = test_connection(host)
    print("Connection test successful" if success else "Connection test failed")