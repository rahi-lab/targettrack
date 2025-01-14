#!/usr/bin/env python3
import sys
import re
import logging
import os
import argparse
from PyQt5.QtWidgets import QApplication
from src.graphic_interface import gui_single

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gui_launcher')

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str, help='Path to the dataset file')
parser.add_argument('-n', '--node', type=str, help='Port number for the server')
parser.add_argument('-p', '--full-path', help="Use full path instead of default dataset dir", action="store_true")
args = parser.parse_args()
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
    # if len(sys.argv) != 2:
    #     print("Usage:")
    #     print("  Local:  python gui_launcher.py /path/to/file.h5")
    #     print("  Remote: python gui_launcher.py node:/path/to/file.h5")
    #     print("\nNote: For remote files, ensure SSH tunnel is active to node:18861")
    #     print("\nExample:")
    #     print("  python gui_launcher.py dgx001:/om2/user/name/data.h5")
    #     sys.exit(1)
    print(os.getenv('DEFAULT_DATASET_DIR'))

    dataset_path = os.path.join(os.getenv('DEFAULT_DATASET_DIR'), args.dataset_path) if os.getenv('DEFAULT_DATASET_DIR') and not args.full_path else args.dataset_path
    dataset_path = args.node + ":" + dataset_path if args.node else dataset_path
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
            # gui.show()
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Failed to start GUI: {str(e)}")
        sys.exit(1)