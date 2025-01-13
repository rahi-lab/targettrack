import rpyc
import numpy as np
import h5py
from rpyc.utils.server import ThreadedServer
import sys
import os
import logging
import torch
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('targettrack_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('targettrack_service')

class TargetTrackService(rpyc.Service):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.active_jobs = {}
        logger.info(f"Service initialized with device: {self.device}")

    def exposed_train_network(self, config):
        """
        Handle neural network training remotely
        Args:
            config (dict): Training configuration including:
                - model_name: Network architecture to use 
                - instance_name: Name for this training run
                - dataset_path: Path to H5 dataset
                - training_frames: Number of frames to use for training
                - validation_frames: Number for validation
                - epochs: Number of epochs to train
        Returns:
            job_id: Unique identifier for tracking this job
        """
        try:
            # Generate unique job ID
            job_id = f"{config['model_name']}_{config['instance_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Start training in separate thread
            import threading
            job_thread = threading.Thread(
                target=self._run_training,
                args=(job_id, config)
            )
            job_thread.start()
            
            self.active_jobs[job_id] = {
                'thread': job_thread,
                'status': 'running',
                'config': config
            }
            
            logger.info(f"Started training job {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Error starting training: {str(e)}", exc_info=True)
            raise

    def _run_training(self, job_id, config):
        """
        Internal method to run the actual training process
        """
        try:
            # Import here to avoid loading ML libraries until needed
            from src.neural_network_scripts import run_NNmasks_f
            
            # Setup training args similar to original script
            args = [
                config['dataset_path'],
                f"logs/{job_id}.log",
                "0", # No deformation
                str(config['epochs']),
                "0", # No old training set
                "0", # No deformed frames
                str(config['training_frames']),
                str(config['validation_frames'])
            ]

            # Run training
            run_NNmasks_f.run_training(args)
            
            self.active_jobs[job_id]['status'] = 'completed'
            logger.info(f"Completed training job {job_id}")

        except Exception as e:
            self.active_jobs[job_id]['status'] = 'failed'
            self.active_jobs[job_id]['error'] = str(e)
            logger.error(f"Training job {job_id} failed: {str(e)}", exc_info=True)

    def exposed_get_job_status(self, job_id):
        """
        Check status of a training job
        """
        if job_id not in self.active_jobs:
            return {'status': 'not_found'}
        return {
            'status': self.active_jobs[job_id]['status'],
            'error': self.active_jobs[job_id].get('error')
        }

    def exposed_process_frames(self, h5_path, frames, operation):
        """
        Handle frame processing operations
        Args:
            h5_path: Path to source H5 file
            frames: List of frame indices to process
            operation: Processing operation to perform
        """
        try:
            with h5py.File(h5_path, 'r+') as h5:
                if operation == 'segment':
                    from src.mask_processing.segmentation import Segmenter
                    segmenter = Segmenter(h5, h5.attrs['seg_params'])
                    results = segmenter.segment(frames)
                    return results
                elif operation == 'extract_features':
                    from src.mask_processing.features import FeatureBuilder
                    feature_builder = FeatureBuilder(h5)
                    results = feature_builder.extract_features(frames)
                    return results
                else:
                    raise ValueError(f"Unknown operation: {operation}")

        except Exception as e:
            logger.error(f"Error processing frames: {str(e)}", exc_info=True)
            raise

    def exposed_load_dataset(self, h5_path):
        """
        Load and validate dataset
        """
        try:
            with h5py.File(h5_path, 'r') as h5:
                # Validate required attributes
                required = ['W', 'H', 'D', 'C', 'T', 'N_neurons'] 
                missing = [attr for attr in required if attr not in h5.attrs]
                if missing:
                    raise ValueError(f"Missing required attributes: {missing}")
                    
                # Return basic dataset info
                return {
                    'frame_shape': (h5.attrs['W'], h5.attrs['H'], h5.attrs['D']),
                    'channels': h5.attrs['C'],
                    'frames': h5.attrs['T'],
                    'neurons': h5.attrs['N_neurons']
                }

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
            raise

def start_server(port=18861, host='0.0.0.0'):
    try:
        server = ThreadedServer(
            TargetTrackService,
            port=port,
            hostname=host,
            protocol_config={
                'allow_public_attrs': True,
                'allow_all_attrs': True
            }
        )
        logger.info(f"Starting server on {host}:{port}")
        server.start()

    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    start_server()