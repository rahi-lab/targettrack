The tutorial of this GUI is currently being written. To be finished by 16.03.2022.

# Targettrack
This is the user manual for the graphical interface for segmenting and editing *C. elegans* 3D images.

# Requirements
- python 3.8
- pyqt5, numpy, scipy, matplotlib, scikit-image, h5py

# Installation Steps

1. Clone this repository ("git clone https://github.com/lpbsscientist/targettrack").
2. If you don't have conda or miniconda installed, download it from https://docs.conda.io/en/latest/miniconda.html.
3. Follow the instructions of install.txt to create a virtual environment and install the necessary packages.
8. Run the program from your command line with `python3 gui_launcher.py [dataset name]`

# User Guide
### Preparing the h5 file
This program is designed for 3D movies in the format of `.h5` file. The `.h5` file should have T groups where T is the number of volumes in the movie. The volume at time t is saved the member 'frame' of group t. The `.h5` file should also contain the following attributes:\
"name"= name of the movie\
"C" = number of channels in the movie\
"W" = width of each volume\ 
"H" = height of each volume\
"D" = depth of each volume\
"T" = number of volumes in the movie\
"N_neurons" = number of neurons which is set to zero before annotation.
You can use the script `nd22h5.py` to convert the `.nd2` files obtained by Nikon Eclipse Ti2 spinning disc confocal microscope into proper `.h5` file format for the Targettrack GUI.
The program will save segmentation masks for each frame t as the member "mask" of the group t in the '.h5' file.

## The interface
After opening the '.h5' file, you will see the first volume of the movie as the main part of the window. Using the wheel of the mouse, you can go through different Z-stacks of your volume.
If the movie already contains masks for some frames, there will be a bar of neuron numbers on the top of the window. The neurons present in each frame are colored as blue and the absent ones are colored as red.\
On the bottom of the window, you see a slider for going through different volumes. You can also use the keys `m` and `n` on your keyboardd to go to the next or previous volume respectively.\
On the bottom right side of the window, there are multiple tabs for viewing, editing, or exporting the movie. 

### The neuron bar
If you are in the `overlay mask` mode, you can highlight each neuron by clicking on its corresponding number. The key of the highlighted neuron becomes green in the frames when the neuron is present and yellow when it is absent. You can also assign one of the keys `q`,`w`,`e`,`r`,`t`,`y` and`u` to any of the annotated neurons by clicking on the white button on top of the neuron number. After assigning the key, it will show up in the "track" tab on top right side of the window. There, you can see list of volumes where theneuron is present (marked by color blue).

### View tab
This tab is used for improving the visualization of the movie. You can view different aspects of the volume using the following checkboxes:\
`Show only first channel`: this option only displayes the first channel in multichannel movies.\
`Show only second channel`: this option only displayes the second channel in multichannel movies.\
`Autolevels`: autolevels the image for displaying.\
`Overlay mask`: displays the mask for the annotated frames.\
`Only NN mask`: displays the prediction of the neural network (if available) even for the ground truth images.\
`Aligned`: if rotation matrices have already been computed in the `Processing` tab, this option shows the aligned volume. We used the method introduced in  https://github.com/bing-jian/gmmreg-python for alignment of 3D volume images to have the neurons of the worm aligned as much as possible across frames.\
`Cropped`: if the alignment has already been done and the cropping region determined through `Processing` tab, this option shows displays the cropped image.\
`Blur image`: it applies blurring using difference of Gaussians method on the first channel of the image by default. If you choose `Show only second channel` checkbox, the second channel will be blurred. The parameters used for applying blurring are set from the slide bars below the check box.









