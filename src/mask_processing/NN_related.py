import numpy as np
import scipy.ndimage as sim


def post_process_NN_masks(times, exempt_neurons, load_fun, save_fun):
    """
    MB added: to post process the predictions of NN for the selected frames as the ground truth
    it goes through all the frames and all the cells and whenever the mask of two cells are touching each other it
    relabels the smaller neuron to the label of the larger one
    :param times: iterable of the times to be post-processed
    :param exempt_neurons: neurons that you do not want to be considered fot this post-processing
    :param load_fun: callable such that load_fun(t) is the mask to be post-processed for time t
    :param save_fun: callable such that save_fun(t, mask) saves the post-processed mask for time t
    """
    for t in times:
        if True:
            mask = load_fun(t)
            if mask is not False:
                if True:  # for c in cell_list:
                    labelArray, numFtr = sim.label(
                        mask > 0)  # get all the disconnected components of the nonzero regions of mask
                    for i in range(numFtr + 1):
                        IfZero = False
                        submask = (labelArray == i)  # focus on each connected component separately
                        list = np.unique(
                            mask[submask])  # list of all cell ids corresponded to the connected component i
                        list = [int(k) for k in (set(list) - set(exempt_neurons))]
                        if np.sum(submask) < 3:  # if the component is only 1 or 2 pixels big:
                            for l in range(len(list)):
                                if np.sum(mask == list[l]) > 5:  # check if this is the only place any cell is
                                    # mentioned. If a cell is mentioned somewhere else set this component to zero.
                                    if list[l] not in exempt_neurons:
                                        mask[submask] = 0
                                        IfZero = True  # whether the component was set to zero   # TODO MB: IfZero is not used... did you want elif or rather if on the next line??
                        if len(list) > 1 and not IfZero:
                            Volume = np.zeros(len(list))
                            for l in range(len(list)):
                                Volume[l] = sum(mask[submask] == list[l])
                            BiggestCell = list[np.argmax(Volume)]
                            mask[submask] = BiggestCell
                save_fun(t, mask)
            else:
                print("There are no predictions for this frame")


def post_process_NN_masks2(times, exempt_neurons, load_fun, save_fun):
    """
    MB added: to post process the predictions of NN for the selected frames as the ground truth
    This postprocessing step is basically like the previous one but uses different connectivity
    criteria to decide whether the neurons are touching each other or not. If the neurons are only
     neighbors across Z direction, it doesn't relabel them
    :param times: iterable of the times to be post-processed
    :param exempt_neurons: neurons that you do not want to be considered fot this post-processing
    :param load_fun: callable such that load_fun(t) is the mask to be post-processed for time t
    :param save_fun: callable such that save_fun(t, mask) saves the post-processed mask for time t
    """
    s = [[[False, False, False],
          [False, True, False],
          [False, False, False]],
         [[False, True, False],
          [False, True, False],
          [False, True, False]],
         [[False, False, False],
          [False, True, False],
          [False, False, False]]]
    for t in times:
        mask = load_fun(t)
        if mask is not False:
            maskOrig = mask
            labelArray, numFtr = sim.label(mask > 0,
                                           structure=s)  # get all the disconnected components of the nonzero regions of mask
            Grandlist = np.unique(mask)  # list of all cell ids in the mask
            Grandlist = [int(k) for k in (set(Grandlist) - set(exempt_neurons))]
            for c in Grandlist:
                # get connected component of each neuron
                labelArray_c, numFtr_c = sim.label(maskOrig == c,
                                                   structure=s)  # get all the disconnected components of a certain cell class
                Vol = np.zeros([1, numFtr_c])
                for i in range(0, numFtr_c):
                    Vol[0, i] = np.sum(labelArray_c == i + 1)  # volume of each connected component of cell class c
                BigComp = np.argmax(Vol) + 1  # label of the biggest component of class
                Comp_c_BigComp = (labelArray_c == BigComp)  # biggest connected component labeled as c
                label_c_BigComp = np.unique(labelArray[Comp_c_BigComp])
                for i in range(1, numFtr_c + 1):
                    if not i == BigComp:
                        Comp_c_i = (labelArray_c == i)
                        label_c_i = np.unique(labelArray[Comp_c_i])  # label of c_i component in the first big labeling
                        connCom_containing_c_i = (labelArray == label_c_i)
                        cells_Connected_To_i = set(np.unique(maskOrig[connCom_containing_c_i])) - {c} - {0}
                        cells_Connected_To_i = [int(k) for k in cells_Connected_To_i]
                        if not label_c_i == label_c_BigComp:
                            if len(cells_Connected_To_i) == 1:  # if only one other cell is connected to c_i
                                mask[Comp_c_i] = cells_Connected_To_i[0]
                            elif len(cells_Connected_To_i) > 1:
                                Vol_c_i_conn = np.zeros([1, len(cells_Connected_To_i)])
                                for j in range(len(cells_Connected_To_i)):
                                    Vol_c_i_conn[0, j] = np.sum(
                                        connCom_containing_c_i & (mask == cells_Connected_To_i[j]))
                                mask[Comp_c_i] = int(cells_Connected_To_i[np.argmax(Vol_c_i_conn)])
            save_fun(t, mask)


def post_process_NN_masks3(times, neurons, load_fun, save_fun):
    """
    MB added: to post process the predictions of NN for the selected frames as the ground truth
     If any of the neurons in the vector "neurons" touch each other
     and form one connected component it renames the smaller ones to the largest one
    :param times: iterable of the times to be post-processed
    :param neurons: neurons that you want to postprocess.
    :param load_fun: callable such that load_fun(t) is the mask to be post-processed for time t
    :param save_fun: callable such that save_fun(t, mask) saves the post-processed mask for time t
    """
    Vol = np.zeros([1, len(neurons)])
    for t in times:
        mask = load_fun(t)
        if mask is not False:
            if True:#for c in cell_list:
                labelArray, numFtr = sim.label(mask>0)#get all the disconnected components of the nonzero regions of mask
                for i in range(1,numFtr+1):
                    submask =  (labelArray==i)#focus on each connected component separately
                    for k in range(len(neurons)):
                        Vol[0,k] = np.sum(mask[submask] == neurons[k])#volume of each of the chosen neurons in this component
                    MaxInd = np.argmax(Vol[0,:])
                    if not np.max(Vol[0,:])==0:
                        for k1 in range(len(neurons)):
                            if not k1==MaxInd:
                                k1_comp = (submask & (mask == neurons[k1]))
                                mask[k1_comp] = int(neurons[MaxInd])
            save_fun(t, mask)
        else:
            print("There are no predictions for this frame")


def post_process_NN_masks4(times, neurons, load_fun, save_fun):
    """
    MB added: to post process the predictions of NN for the selected frames as the ground truth
     if a certain neuron has multiple components, it deletes the components that have smaller volume.
    :param times: iterable of the times to be post-processed
    :param neurons: neurons that you want to postprocess.
    :param load_fun: callable such that load_fun(t) is the mask to be post-processed for time t
    :param save_fun: callable such that save_fun(t, mask) saves the post-processed mask for time t
    """
    for t in times:
        mask = load_fun(t)
        if mask is not False:
            for n in neurons:#for c in cell_list:
                labelArray, numFtr = sim.label(mask==n)#get all the disconnected components of the nonzero regions of mask
                if numFtr>1:
                    Vol = np.zeros([1,numFtr])
                    for i in range(1,numFtr+1):
                        Vol[0,i-1] =  np.sum(labelArray==(i))#focus on each connected component separately
                    MaxInd = np.argmax(Vol[0,:])
                    for k1 in range(numFtr):
                        if not k1==MaxInd:
                            print(k1+1)
                            k1_comp = (labelArray==(k1+1))
                            mask[k1_comp] = 0
            save_fun(t, mask)
        else:
            print("There are no predictions for this frame")


def post_process_NN_masks5(times, neurons, load_fun, save_fun):
    """
    MB added: to post process the predictions of NN for the selected frames as the ground truth
     If any of the neurons in the vector "neurons" touch each other,
     and form one connected component it renames all the segments in the connected component to the first neuron in "neurons" vector
    :param times: iterable of the times to be post-processed
    :param neurons: neurons that you want to postprocess.
    :param load_fun: callable such that load_fun(t) is the mask to be post-processed for time t
    :param save_fun: callable such that save_fun(t, mask) saves the post-processed mask for time t
    """
    Vol = np.zeros([1,len(neurons)])
    for t in times:
        mask = load_fun(t)
        if mask is not False:
            if True:#for c in cell_list:
                labelArray, numFtr = sim.label(mask>0)#get all the disconnected components of the nonzero regions of mask
                for i in range(1,numFtr+1):
                    submask =  (labelArray==i)#focus on each connected component separately
                    for k in range(len(neurons)):
                        Vol[0,k] = np.sum(mask[submask]==neurons[k])#volume of each of the chosen neurons in this component
                    if not np.max(Vol[0,:])==0 and not Vol[0,0]==0:
                        for k1 in range(len(neurons)):
                            if not k1==0:
                                k1_comp = (submask&(mask==neurons[k1]))
                                mask[k1_comp] = int(neurons[0])
            save_fun(t, mask)
        else:
            print("There are no predictions for this frame")
