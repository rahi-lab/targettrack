import rpyc
from rpyc.utils.server import ThreadedServer
import logging
import socket
import torch
import h5py
import numpy as np
import os
from pathlib import Path
import shutil
import multiprocessing
import importlib
import subprocess
import sys
from typing import Dict, Any

def check_environment() -> Dict[str, Any]:
    """Check the environment configuration and return diagnostic info"""
    env_info = {}
    
    # Python info

    
    # PyTorch build info
    env_info['torch_version'] = torch.__version__
    env_info['torch_cuda_available'] = torch.cuda.is_available()
    if hasattr(torch, 'version'):
        env_info['torch_cuda_built'] = torch.version.cuda
        
    
        
    return env_info

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('points_server')

class PointsTrainingService(rpyc.Service):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.datadir = Path("data/data_temp")
        self.datadir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using device: {self.device}")
        
    def exposed_train_points_network(self, h5_path, network_name, run_name, training_params):
        """
        Train points prediction network
        
        Args:
            h5_path: Path to h5 file
            network_name: Name of network architecture to use
            run_name: Name for this training run
            training_params: Dict of training parameters including:
                - num_epochs
                - batch_size
                - learning_rate
                - num_workers (for dataloading)
                - training_frames
                - validation_frames
        """
        logger.info(f"Starting points training run {run_name}")
        
        try:
            # Import network architecture
            NetMod = importlib.import_module(network_name)
            
            # Setup temporary directory
            run_dir = self.datadir / f"{run_name}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)
            
            # Copy and load h5 file
            temp_h5 = run_dir / "data.h5"
            shutil.copy2(h5_path, temp_h5)
            
            h5 = h5py.File(temp_h5, "r+")
            
            # Get basic parameters
            shape = (h5.attrs["C"], h5.attrs["W"], h5.attrs["H"], h5.attrs["D"])
            num_classes = h5.attrs["N_neurons"] + 1
            
            # Initialize network
            net = NetMod.Net(n_channels=shape[0], num_classes=num_classes)
            net.to(device=self.device)
            
            # Setup training
            optimizer = torch.optim.Adam(net.parameters(), 
                                       lr=training_params.get('learning_rate', 0.0003),
                                       amsgrad=True)
            
            from src.neural_network_scripts.NNtools_pts import (
                TrainDataset, selective_ce, get_ious
            )
            
            # Create datasets
            train_dataset = TrainDataset(shape)
            # Add training data...
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=training_params.get('batch_size', 3),
                shuffle=True,
                num_workers=training_params.get('num_workers', 4)
            )
            
            # Training loop
            losses = []
            ious = []
            for epoch in range(training_params['num_epochs']):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = net(data)
                    loss = selective_ce(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                losses.append(epoch_loss / len(train_loader))
                
                # Calculate IoU periodically
                if epoch % 10 == 0:
                    with torch.no_grad():
                        iou = get_ious(output, target)
                        ious.append(iou)
                        
                logger.info(f"Epoch {epoch}: Loss = {losses[-1]:.4f}")
            
            # Save results back to h5
            identifier = f"net/{network_name}_{run_name}"
            if identifier in h5:
                del h5[identifier]
            group = h5.create_group(identifier)
            
            # Save model state
            self._save_model_state(h5, identifier, net.state_dict())
            
            # Save metrics
            metrics = {
                'loss_history': losses,
                'iou_history': ious,
            }
            for name, data in metrics.items():
                if isinstance(data, list):
                    data = np.array(data)
                group.create_dataset(name, data=data)
            
            h5.close()
            
            # Copy results back
            shutil.copy2(temp_h5, h5_path)
            
            # Cleanup
            shutil.rmtree(run_dir)
            
            return {
                'success': True,
                'loss_history': losses,
                'iou_history': ious
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_model_state(self, h5file, identifier, state_dict):
        """Save PyTorch model state to H5 file"""
        if f"{identifier}/model_state" in h5file:
            del h5file[f"{identifier}/model_state"]
        
        group = h5file.create_group(f"{identifier}/model_state")
        for key, tensor in state_dict.items():
            group.create_dataset(key, data=tensor.cpu().numpy())
    
    def exposed_get_gpu_info(self):
        """Get GPU information"""
        info = {}
        
        # Basic CUDA availability
        info['cuda_available'] = torch.cuda.is_available()
        if not info['cuda_available']:
            info['error'] = 'torch.cuda.is_available() returned False'
            return info
            
        try:
            # Device count and properties
            info['device_count'] = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            info['current_device'] = current_device
            info['device_name'] = torch.cuda.get_device_name(current_device)  
            
            # Get properties for each device
            devices = []
            for i in range(info['device_count']):
                dev_props = {
                    'name': torch.cuda.get_device_name(i),
                    'capability': torch.cuda.get_device_capability(i),
                    'total_memory': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_reserved': torch.cuda.memory_reserved(i)
                }
                devices.append(dev_props)
            info['devices'] = devices
            
            return info
            
        except Exception as e:
            info['error'] = f'Error getting GPU info: {str(e)}'
            return info

if __name__ == "__main__":
    port = 18861
    host = '0.0.0.0'
    
    node_hostname = socket.gethostname()
    logger.info(f"Starting points training server on {host}:{port}")
    logger.info(f"Connect using hostname: {node_hostname}")
    
    # Check and log environment status
    env_info = check_environment()
    
    logger.info("Environment Configuration:")
    logger.info(f"PyTorch: {env_info['torch_version']}")
    logger.info(f"CUDA Available: {env_info['torch_cuda_available']}")
    if 'torch_cuda_built' in env_info:
        logger.info(f"PyTorch CUDA Version: {env_info['torch_cuda_built']}")
    
    server = ThreadedServer(
        PointsTrainingService,
        hostname=host,
        port=port,
        protocol_config={
            'allow_public_attrs': True,
            'allow_all_attrs': True,
            'sync_request_timeout': 3600  # 1 hour timeout for long training jobs
        }
    )
    server.start()