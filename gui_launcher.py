#!/usr/bin/env python3
import sys
import re
import logging
import os
from PyQt5.QtWidgets import QApplication
from src.graphic_interface import gui_single

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gui_launcher')

def parse_remote_path(path):
    """
    Parse remote path for a tunneled connection
    
    Args:
        path: Should be in format node:/path/to/file.h5
            
    Returns:
        (node, filepath) or None if local path
    """
    # Check if path is remote (has node: prefix)
    path_match = re.match(r'^([^:]+):(.+)$', path)
    if not path_match:
        return None
        
    node, filepath = path_match.groups()
    logger.info(f"Using tunneled connection for node {node}")
    
    return node, filepath

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("  Local:  python gui_launcher.py /path/to/file.h5")
        print("  Remote: python gui_launcher.py node:/path/to/file.h5")
        print("\nNote: For remote files, ensure SSH tunnel is active to node:18861")
        print("\nExample:")
        print("  python gui_launcher.py dgx001:/om2/user/name/data.h5")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Parse path
    remote_info = parse_remote_path(dataset_path)

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("HPC Neural GUI") 
        if remote_info:
            node, filepath = remote_info
            # Connect via SSH tunnel on localhost
            gui = gui_single.gui_single(filepath, tunneled=True, node=node)
        else:
            # Local path
            gui = gui_single.gui_single(dataset_path)
            
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Failed to start GUI: {str(e)}")
        sys.exit(1)