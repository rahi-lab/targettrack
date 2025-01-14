# TargetTrack

TargetTrack is a user-friendly graphical interface designed for annotating 3D recordings of C. elegans and tracking neuronal activity. This fork enhances the original functionality by enabling the GUI to run on a local client machine while delegating backend processing to a remote server, optimizing performance and resource utilization.

## Features
•	Local GUI Execution: Run the graphical interface on your local machine for responsive interaction.
•	Remote Backend Processing: Offload computationally intensive tasks to a remote server, leveraging its processing power.
•	Secure SSH Tunneling: Establish secure connections between the client and server to ensure data integrity and confidentiality.

## Installation

### Prerequisites
•	Local Machine:
-	Python 3.8 or higher
-	Conda or Mamba for environment management
-	SSH access to the remote server

•	Remote Server:
-	Python 3.8 or higher
-	Necessary computational resources for backend processing

### Setup Steps
1.	**Clone the Repository:**
```bash
git clone https://github.com/kirjner/targettrack.git
cd targettrack
```

2.	**Set Up the Conda Environment on the Local Machine:**
```bash
mamba env create -f gui_env.yaml
mamba activate targettrack_gui
```

3.	Install Required Packages on the Remote Server:
Ensure the remote server has the necessary dependencies installed. You can use the provided server_env.yaml file to set up the environment:
```bash
mamba env create -f server_env.yaml
mamba activate targettrack_server
```

4.	**Configure SSH Access:**
Set up SSH keys and ensure that you can connect to the remote server without password prompts. Adjust SSH configurations as needed to facilitate seamless connections.

## Usage
1.	**Start the Remote Server:**
On the remote server, navigate to the targettrack directory and run:
```bash
python hpc_gpu_server.py
```

2.	**Establish an SSH Tunnel from the Local Machine:**
Set up port forwarding to securely connect to the remote server:
```bash
ssh -v -N -L 18861:remote_server_address:18861 your_username@remote_server_address
```

Replace remote_server_address with the actual address of your remote server and your_username with your username on that server.

3.	**Launch the GUI on the Local Machine:**
With the SSH tunnel active, run the GUI launcher:
```bash
python gui_launcher.py /path/to/your/datafile.h5
```

Replace /path/to/your/datafile.h5 with the path to your HDF5 data file.


## Acknowledgements

This project builds upon the original [TargetTrack](https://github.com/rahi-lab/targettrack) developed by the Rahi Lab.

For detailed information on system requirements and dependencies, please refer to the original TargetTrack README.
