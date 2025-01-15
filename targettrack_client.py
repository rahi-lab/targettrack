import rpyc
import os
import logging
from typing import Optional, Dict, Any
import time

logger = logging.getLogger('targettrack_client')

class TargetTrackClient:
    def __init__(self, host='localhost', port=18861):
        """
        Initialize connection to TargetTrack remote service
        """
        self.connection = None
        self.host = host
        self.port = port
        self.connect()

    def connect(self):
        """
        Establish connection to remote service
        """
        try:
            self.connection = rpyc.connect(
                self.host, 
                self.port,
                config={'sync_request_timeout': 300}  # 5 minute timeout
            )
            logger.info(f"Connected to remote service at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to remote service: {str(e)}")
            raise

    def train_network(self, config: Dict[str, Any], callback=None) -> str:
        """
        Start network training job on remote service
        
        Args:
            config: Training configuration dictionary
            callback: Optional callback function for progress updates
            
        Returns:
            job_id: Identifier for tracking training progress
        """
        try:
            # Validate config
            required = ['model_name', 'instance_name', 'dataset_path', 
                       'training_frames', 'validation_frames', 'epochs']
            missing = [k for k in required if k not in config]
            if missing:
                raise ValueError(f"Missing required config keys: {missing}")

            # Start training job
            job_id = self.connection.root.train_network(config)
            logger.info(f"Started training job {job_id}")

            # Monitor progress if callback provided
            if callback:
                self._monitor_training(job_id, callback)

            return job_id

        except Exception as e:
            logger.error(f"Error in train_network: {str(e)}")
            raise

    def _monitor_training(self, job_id: str, callback, poll_interval: int = 10):
        """
        Monitor training progress and call callback with updates
        """
        while True:
            try:
                status = self.get_job_status(job_id)
                callback(status)
                
                if status['status'] in ['completed', 'failed']:
                    break
                    
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring job {job_id}: {str(e)}")
                break

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of training job
        """
        try:
            return self.connection.root.get_job_status(job_id)
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            raise

    def process_frames(self, h5_path: str, frames: list, operation: str):
        """
        Run processing operation on frames
        
        Args:
            h5_path: Path to H5 dataset
            frames: List of frame indices to process
            operation: Operation to perform ('segment' or 'extract_features')
        """
        try:
            return self.connection.root.process_frames(h5_path, frames, operation)
        except Exception as e:
            logger.error(f"Error in process_frames: {str(e)}")
            raise

    def validate_dataset(self, h5_path: str) -> Dict[str, Any]:
        """
        Validate H5 dataset and return info
        """
        try:
            return self.connection.root.load_dataset(h5_path)
        except Exception as e:
            logger.error(f"Error validating dataset: {str(e)}")
            raise

    def close(self):
        """
        Close connection to remote service
        """
        if self.connection:
            self.connection.close()
