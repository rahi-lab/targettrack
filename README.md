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

### The interface
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


### Annotation tab
This tab is used for annotating the volumes. It can be used for correcting the predictions of the neural network or the results of the watershed segmentation method (implemented in `Processing` tab). There are two modes of annotating a region from scratch:\
`Mask annotation mode`: it uses the value in the `Threshold for adding regions` box as the threshold. If you rightclick on any pixel, a box will ask you to enter the number of the neuron you want to annotate. After entering the number, all the pixels around the clicked pixel that have higher values than the threshold will be labeled with that number. You can change the threshold either by entering a new threshold value in the threshold box or by middle clicking on a pixel. If you middle click on a certain pixel, the value of that pixel will be used as the threshold.\
`Boxing Mode`: it adds new regions to the mask by defining a box with desirable dimensions. You can set the length, width, height and the label of the box you want to add in the `Box detailes` box. After setting the dimensions and the label, you can leftclick on the pixel where the lower left corner of the box should be. If you choose 0 as the label of the box, it will work as an eraser for your masks.\

The actions of `Boxing Mode` and `Mask annotation mode` can be reversed by pressing the key `z`.

In addition to annotating from scratch, you can also change the labels of existing masks or delete them using the following buttons:\
`Renumber`: You first click on the number of the neuron you want to relabel on the neuron bar and highlight it. This will activate the `Renumber` button. Upon pressing the `Renumber` button you get asked to enter the new label you want to use for the neuron. After entering the new label, if the neuron you chose has only one connected component, it will be relabeled immediately. If it has multiple disjoint components, you are asked if you want to relabel all those components or only one of them. If you choose `cancel` all the components will be renumbered. If you choose `Ok`, then you have to right click on a pixel inside the region you want to renumber to only relabel that component and not the others.\
If you want to renumber a neuron in more than one frame, you can check the `Change within` checkbox and set the interval of the frames you want use for renumbering.\

`Delete`:You first click on the number of the neuron you want to relabel on the neuron bar and highlight it. This will activate the `Delete` button. Upon pressing the `Delete` button, the desired neuron will be deleted.
The actions of Deleting and renumbering can be reversed by pressing the key `z` if they are only applied on one frame.

`Permute`: You permute labels of more than one neuron at once. If you enter the list of neurons separated by "," then the label of each neuron wil change to the label of the one after it. For example if you enter `1,2,3` in the permute box, the label of neuron 1 changes to 2, 2 changes to 3 and 3 changes to 1.
### NN tab
This tab is designed to run the the neural network (NN) directly from the GUI. The GUI will make a copy of the `.h5` file in the `data/data_temp` folder and saves the result of the NN in that file.
To run the neural network, enter the number of validation,training set, and the epochs for running in the corresponding box and press the `Train Mask Prediction Neural network` button. Note that the sum of validation and tratining set should not be more than the total number of annotated frames.
##### Generating target frames
This can be done only after one successful run of the neural network. To generate the deformed frames, check the `Add deformation` checkbox and enter the number of deformed frames you want to generate in the `Number of target frames` box.
##### Checking NN results
The results of the NN are saved in the copied file in `data/data_temp` folder. If you open the file, check the `Overlay mask` checkbox in View tab, and choose the NN instance in the `Select NN masks` choice box, you can see the predictions of the neural network on all the unannotated frames. To see the prediction on the annotated ground truth frames, you can use the `Only NN mask` checkbox in View tab. If you want to focus on validation set, you can open the `Validation frames id` collapsible box to see which frames were assigned to the validation set in that run.

##### Postprocessing NN results
There are multiple modes of postprocessing you can use to improve NN predictions. You can enter the mode in the `Post-processing mode` choice box and one of the following post-processings are applied on all the selected frames.

-Mode 1: it goes through all the cells and whenever the masks of two cells are touching each other it relabels the smaller neuron to the label of the larger one. Only the neurons listed in the input box are exempted from this modification.

-Mode 2: this postprocessing mode is basically like the previous one but uses different connectivity criteria to decide whether the neurons are touching each other or not. If the neurons are only neighbors across Z direction, it doesn't relabel them. Only the neurons listed in the input box are exempted from this modification. 

-Mode 3: if any of the neurons listed in the box touch each other and form one connected component it renames the smaller ones to the largest one. 

-Mode 4: if a certain neuron has multiple disjoint components, it deletes the components that have smaller volumes.

-Mode 5: If any of the neurons neurons listed in the box touch each other and form one connected component it renames all the segments in the connected component to the first neuron in the list.

Finally, if after postprocessing the results look good enough, you can save it as ground truth mask using `Approve mask` button.

### Frame selection
This tab is designed mainly to choose a subset of frames for editing or annotating. You can either use a percentage of segmented, non-segmented, or all frames here or manually enter the exact id of the frames you want to choose. 
To choose a percentage of frames, write the percentage you want in the percentage box, fill one of the options `segmented frames`, `non segmented frames`, or `all` frames based on your need and push the `Select` button.

To choose a specific set of frames, enter the number id of those frames separated with a "," in the box at the bottom of the tab, fill the `manual selection` choice, and push the `Select` button.

You can also use the reference frame for alignment in thes tab. If you push `Use this frame as reference` button, the current frame will be used as a reference to align all other frames with respect to it in the Processing tab.

### Export/Import tab


### Processing tab

### IO tab







