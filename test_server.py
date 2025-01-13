import rpyc
from rpyc.utils.server import ThreadedServer
import logging
import socket

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_server')

class TestService(rpyc.Service):
    def exposed_hello(self):
        """Simple test method"""
        logger.info("Hello method called")
        return "Hello from cluster!"
    
    def exposed_add(self, x, y):
        """Test method with parameters"""
        result = x + y
        logger.info(f"Add method called with {x}, {y} = {result}")
        return result

if __name__ == "__main__":
    port = 18861
    # Bind to all interfaces
    host = '0.0.0.0'
    
    # Get actual hostname of compute node
    node_hostname = socket.gethostname()
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Connect using hostname: {node_hostname}")
    
    server = ThreadedServer(
        TestService, 
        hostname=host,
        port=port, 
        protocol_config={
            'allow_public_attrs': True,
            'allow_all_attrs': True
        }
    )
    server.start()
