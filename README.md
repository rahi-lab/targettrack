# Targettrack
This is the user manual for the graphical interface(GUI) for segmenting and editing *C. elegans* volumetric images.

# From the authors
We understand it could be exhaustive to install and run external code. If you encounter any problems, please do not hesitate to corefranciscopark@g.harvard.edu or :warning: Can you add your email Mahsa? :warning: for assistance, we will be happy to help even if it is a very simple problem.

# Requirements
## System Requirements
- Processor (CPU): Intel Core i5 or AMD Ryzen 5 or higher
- Operating System: Aesthetically best on Linux(Tested on Ubuntu, Mint), but all functionalities available on Mac and Windows. However, since a Mac cannot have a CUDA compatible Nvidia GPU, the GUI can only be used for annotation and proofreadin.
- Memory (RAM): 16GB+ (32GB+ recommended)
- Storage : SSD highly recommended for fast IO, please have at least 4x (size of your h5 data) available for smooth running
- :red_circle: GPU: (optional but highly recommended if training a network) NVIDIA 2080 Ti or higher recommended, a large-memory GPU is preferred over a fast boost clock GPU. Especially, gaming GPUs are enough for this purpose.

## Dependencies
- python=3.8.13  #3.9 and 3.10 works but hasn't been tested throughly
- pytorch=1.11.0 # 1.8 ~ 1.12 all seems to work, but is not guaranteed
- ipython :warning: Do we need this Mahsa? (not on our side)
- matplotlib>=3.5.2
- numpy>=1.23.4
- pandas :warning: version needed.
- scikit-image>=0.18.1
- scikit-learn>=0.23.0
- scipy>=1.9.1
- tqdm>=4.59.0
- sparse :warning: version needed
- nd2reader :warning: version needed
- PyQt5=5.12
- pyqtgraph>=0.11.0
- opencv-python>=4.5.1.48
- opencv-python-headless>=4.5.1.48
- h5py>=3.6.0
- albumentations>=0.5.2
- connected-components-3d>=3.2.0
- torchvision>=0.12.0
- alphashape :warning: version needed

If these packages are installed, the GUI and neural network is supposed to run without further installation process. However, if you encounter a bug, please follow the steps below.

# Installation Steps

1. Clone this repository ("git clone https://github.com/lpbsscientist/targettrack").
2. If you don't have conda or miniconda installed, download it from https://docs.conda.io/en/latest/miniconda.html.
3. In your command line, run each of the commands in install.txt (except the first, if you have already cloned the repository).
This will create a virtual environment and install the necessary packages.
4. Place your `.h5` data file in the "targettrack" folder then run the program from your command line with `python3 gui_launcher.py [dataset name]`,
where `[dataset name]` is the name of your file.

## Common installation errors
- We recommend using `pip install opencv-python` instead of conda to install opencv
- We recommend sticking to pytorch 1.11.0 installed from https://pytorch.org/get-started/locally/

# Preparing your data into our `.h5` format

## Exporting to hdf5 files from a different system or language

- The file must end with the suffix .h5
- All images should have the same (Channel,Width,Height,Depth) dimensions and should be saved as hdf5 datasets with name `0/frame` ~ `N-1/frame` for N images
- The dimension information should be saved as hdf5 attributes as `C`=Channel, `W`=Width, `H`=Height, `D`=Depth
- The hdf5 attribute `N_neurons` should be set to a integer >1 and the number of images should be saved as `T`=N 

## For python users
Please refer to the script src/assembleh5.py. (estimated reading time is 3 minutes) It is a very short script  generating a hdf5 file at `data/example.h5`

# Running demo for mask annotations
We guide you step-by-step through the demo:
1. Download the sample `.h5` file from https://drive.google.com/drive/folders/1-El9nexOvwNGAJw6uFFENGY1DqQ7tvxH?usp=sharing . This file is a denoised, aligned, and cropped movie of a freely moving worm in red channel. It has around 150 annotated frames and results of training the neural network on 5 of those frames.
2. Open the sample file using `python3 gui_launcher.py epfl10_CZANet_Final.h5`
  <p align="center"> 
  <img src="src/Images/start.png" width=600> 
  </p>
  
3. Check the `Overlay mask` checkbox to see the annotated frames' masks. Notice that the present neurons in each frame are marked with blue in the neuron bar on top and absent ones by red. 
  <p align="center"> 
  <img src="src/Images/OverlayMask.png" width=600> 
  </p>
  
4. Highlight the masks by pressing on their corresponding key in the neuron bar. The highlighted neurons' key becomes green as you can see in the figure below (orange when the highlighted neuron is absent). 
You can change the label of the highlighted neurons by pressing the `Renumber` button in the `Annotate` tab.
<p align="center"> 
<img src="src/Images/Highlight10.png" width=600> 
</p> 

5. In order to train the neural network, open the `NN` tab. Set the number of training set, validation set, and epochs in the corresponding boxes and press the `Train Mask Prediction Neural Network` button. 
Once you enter the name of the run, the program will copy the file in the `data/data_temp` folder and train the neural network on the new file. The neural network needs to be trained on a GPU with at least 6GB capacity. It takes around 10 minutes to train the neural network for 100 epochs with 6 training frames and 1 validation frame on Ubuntu 18.04.6 with GeForce RTX 2080 Ti graphics. 

<p align="center"> 
<img src="src/Images/NNTrain2.png" width=600> 
</p> 

**Note**: if you had an unseccessful or unfinished run, delete the temporary files in the `data_temp` directory before starting the next run.

6. To check the performance of the neural network, open the file in `data/data_temp`. Choose the run name under `Select NN masks`. You can see the predictions for all frames if you check the `Overlay mask` checkbox. Below you can see the NN predictions for frame 115 (left) by the run `CZANet_Final`, which was trained on 5 frames (right).
<p align="center"> 
<img src="src/Images/unannotatedFrame.png" width=400> 
<img src="src/Images/SeeNNresults.png" width=400> 
</p>

# Running demo for point annotations
We guide you step-by-step through the demo:
1. Download the sample `Harvard_184-15GT.h5` file from https://drive.google.com/drive/folders/1-El9nexOvwNGAJw6uFFENGY1DqQ7tvxH?usp=sharing, and move it to the data folder. This file is a difference of Gaussian(DoG) filtered, rotated and centered movie of a freely moving worm in red/green. No non rigid image transformation applied to the images. It has 15 ground truth annotated frames and 3002-15=2987 frames with neural network predictions.

2. Open the sample file using `python3 gui_launcher.py data/184-15GT.h5`
  <p align="center"> 
  <img src="src/Images/start-point.png" width=600> 
  </p>
  
3. Highlight the points by pressing on their corresponding key in the neuron bar or by clicking on it. The highlighted neurons' key becomes green as you can see in the figure below (orange when the highlighted neuron is absent). The Tracks tab also displays the presence of neurons.
You can change the label of the highlighted neurons by pressing the `Renumber` button in the `Annotate` tab. Press down the corresponding key to annotate a neuron.
<p align="center"> 
<img src="src/Images/Highlight-points.png" width=600> 
</p> 

4. In order to train the neural network, open the `NN` tab. Select 'NN' as the method. Set the number of minimal annotations to be a GT frame, steps to take, deformation parameters, etc (See documentation for more details) with the format "key=value;key=value;" and press the `Run` button. The program then makes temporary files in `data/data_temp` and train a neural network. Other pipelines can be integrated here.
<p align="center"> 
<img src="src/Images/NNtrain-points.png" width=600> 
</p> 

5. To check the performance of the neural network, choose 'NN' under `Select helper data`. You can see the predictions for all frames if you check the `Overlay mask` checkbox. Below you can see the NN predictions (left) which was trained on 15 frames (right).
<p align="center"> 
<img src="src/Images/unannotatedFrame-points.png" width=400> 
<img src="src/Images/SeeNNresults-points.png" width=400> 
</p>

# FAQ
:warning: to be added
