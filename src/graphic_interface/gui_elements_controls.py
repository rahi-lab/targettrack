from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import QRect, QPoint, Qt
from pyqtgraph.parametertree import Parameter, ParameterTree

from ..helpers import QtHelpers
from ..parameters.parameters import Parameters
from typing import Dict
import numpy as np

#CFP added for integration
from src.methods import tracking_methods
from multiprocessing import Process,Pipe


class NeuronBar(QScrollArea):
    """
    This is the bar at the top, with the list of identified neurons and the key assigned to them
    """
    def __init__(self, controller, reserved_keys=None):
        """

        :param controller: main controller to report to
        :param reserved_keys: keys that may not be assigned as neuron keys (they are used by main gui as specific commands)
        """
        super().__init__()
        self.controller = controller
        self.controller.neuron_keys_registered_clients.append(self)
        self.controller.nb_neuron_registered_clients.append(self)
        self.controller.present_neurons_registered_clients.append(self)
        self.controller.highlighted_neuron_registered_clients.append(self)

        if reserved_keys is None:
            reserved_keys = []
        self.reserved_keys = reserved_keys

        self.removed_holder = QWidget()   # a holder, not displayed, to keep alive the widgets removed from other layouts

        self.setWidgetResizable(True)
        dummy = QWidget()
        self.neuron_bar_holderLayout = QHBoxLayout()
        self.separator = QLabel("|")
        self._create_contents()
        dummy.setLayout(self.neuron_bar_holderLayout)
        self.setWidget(dummy)

        self.neurons = {}
        self.keyed_neurons_from1 = set()   # set of neurons that have a key displayed
        self.setContentsMargins(0, 0, 0, 0)
        self.neuron_bar_holderLayout.setSpacing(0)

    def _create_contents(self):
        for i in range(self.neuron_bar_holderLayout.count() - 1, -1, -1):
            wid = self.neuron_bar_holderLayout.itemAt(i).widget()
            if wid is not None:
                wid.setParent(self.removed_holder)   # This is necessary to make separator go away
        self.unactivated_contents = QGridLayout()
        self.activated_contents = QGridLayout()
        self.neuron_bar_holderLayout.addLayout(self.activated_contents)
        self.neuron_bar_holderLayout.addWidget(self.separator)
        self.neuron_bar_holderLayout.addLayout(self.unactivated_contents)

    def change_neuron_keys(self, changes:list):
        """
        Changes (or sets if no previously existing, or deletes if key not given) the keys displayed for given neurons.
        :param changes: list of (neuron_idx_from1, new_key). For each such pair, new_key will be displayed instead of
            previous display for the neuron neuron_idx_from1. If the key is None, then no key will be displayed.
        """
        for neuron_idx_from1, key in changes:
            if key is None:
                self.neurons[neuron_idx_from1].set_text(" ")
                self.keyed_neurons_from1.discard(neuron_idx_from1)
            else:
                self.neurons[neuron_idx_from1].set_text(key)
                self.keyed_neurons_from1.add(neuron_idx_from1)
        self._restore_activated_neurons()

    def _restore_activated_neurons(self):
        """
        Re-creates the activated_neuron_bar_contents (a bar with only the neurons that have a key assigned??)
        """
        del self.activated_contents
        del self.unactivated_contents
        # del self.separator
        self._create_contents()
        for i_from1 in sorted(self.neurons.keys()):
            if i_from1 in self.keyed_neurons_from1:
                self.neurons[i_from1].install_in(self.activated_contents)
            else:
                self.neurons[i_from1].install_in(self.unactivated_contents)

    def change_nb_neurons(self, nb_neurons):
        """
        Changes the number of neurons in the neuron bar.
        :param nb_neurons: new number of neurons
        """
        # TODO AD this removes highlighting and present/absent color...!! do we want that??
        n_delete = len(self.neurons) - nb_neurons

        # SJR: if number of neurons decreased, reset neuron bar
        #MB: set this loop to false because it gives an error for frames with no annotations
        if False:#n_delete > 0:
            for i in reversed(range(self.neuron_bar_contents.count())):
                self.neuron_bar_contents.itemAt(i).widget().setParent(None)   # Todo: or use self.removed_holder if we want to keep the buttons, not sure about that (same remark as above)

        for i_from1 in range(1,nb_neurons+1):
            self.neurons[i_from1] = NeuronBarItem(i_from1, self)
        self._restore_activated_neurons()

    def _make_user_neuron_key(self, neuron_id_from1):
        """
        Asks the user a key to assign to neuron neuron_id_from1 and notifies self.controller
        """
        def fun():
            text, ok = QInputDialog.getText(self, 'Select key for neuron ' + str(neuron_id_from1), 'Key: ')
            # Todo AD: should the parent not be the main gui rather than self??
            if not ok:
                return
            if len(text) == 0:
                text = None
            elif len(text) != 1 or (text in self.reserved_keys):
                errdial = QErrorMessage()
                errdial.showMessage('Invalid key')
                errdial.exec_()
                return
            self.controller.assign_neuron_key(neuron_id_from1, text)
        return fun

    def _make_neuron_highlight(self, i_from1):
        def fun():
            self.controller.highlight_neuron(i_from1)
        return fun

    def change_present_neurons(self, present=None, added=None, removed=None):
        """
        Changes which of the neurons are present in current frame, as their corresponding buttons should be colored
        in blue instead of red.
        :param present: which neuron indices (from 1) are present, if given
        :param added: single neuron index (from 1) that was added, if given
        :param removed: single neuron index (from 1) that was removed, if given
        """
        # todo: don't reset all style sheet of all neurons every time?
        if present is not None:
            for i_from1, neu in self.neurons.items():
                if i_from1 in present:
                    neu.set_present()
                else:
                    neu.set_absent()
            return
        if added is not None:
            self.neurons[added].set_present()
        if removed is not None:
            self.neurons[removed].set_absent()

    def change_highlighted_neuron(self, high: int=None, unhigh:int=None,
                                  **kwargs):
        """
        Highlights or unhighlights neuron buttons.
        :param high: neuron id (from 1), will be highlighted if given
        :param unhigh: neuron id (from 1), will be unhighlighted if given
        """
        if high is not None:
            self.neurons[high].highlight()

        if unhigh is not None:
            self.neurons[unhigh].unhighlight()


class NeuronBarItem:
    """
    This is one neuron in the neuron bar. It contains the colored button with the neuron id (from1), and the key button.
    """
    qss = """
             QPushButton{
             height: 10px;
             width: 20px;
             min-width: 50px;
             }
             QPushButton[color = "a"]{
                 background-color: red;
             }
             QPushButton[color = "p"]{
                 background-color: blue;
             }
             QPushButton[color = "hp"]{
                 background-color: green;
             }
             QPushButton[color = "ha"]{
                 background-color: orange;
             }
          """
          # also had this inside the brackets of the first QPushButton:
          # height: 10px;
          # width: 20px;
          # min-width: 20px;

    def __init__(self, i_from1, parent):
        super().__init__()
        self.i_from1 = i_from1
        self.parent = parent
        self.neuron_key_button = QPushButton(" ")
        self.neuron_key_button.setStyleSheet(self.qss)
        self.neuron_key_button.clicked.connect(self.parent._make_user_neuron_key(self.i_from1))
        self.neuron_button = QPushButton(str(self.i_from1))
        self.neuron_button.setStyleSheet(self.qss)  # TODO set correct color NOW??
        self.neuron_button.clicked.connect(self.parent._make_neuron_highlight(self.i_from1))
        self.present = False   # Todo: set at init??
        self.highlighted = False   # Todo: set at init??

    def install_in(self, holder_layout):
        holder_layout.addWidget(self.neuron_key_button, 0, self.i_from1 - 1)
        holder_layout.addWidget(self.neuron_button, 1, self.i_from1 - 1)

    def set_present(self):
        self.present = True
        if self.highlighted:
            self.neuron_button.setProperty("color", "hp")
        else:
            self.neuron_button.setProperty("color", "p")
        self.neuron_button.setStyle(self.neuron_button.style())   # for some reason this is needed to actually change the color

    def set_absent(self):
        self.present = False
        if self.highlighted:
            self.neuron_button.setProperty("color", "ha")
        else:
            self.neuron_button.setProperty("color", "a")
        self.neuron_button.setStyle(self.neuron_button.style())   # for some reason this is needed to actually change the color

    def highlight(self):
        self.highlighted = True
        self.neuron_button.setStyleSheet(self.qss)
        if self.present:
            self.neuron_button.setProperty("color", "hp")
        else:
            self.neuron_button.setProperty("color", "ha")
        self.neuron_button.setStyle(self.neuron_button.style())   # for some reason this is needed to actually change the color

    def unhighlight(self):
        self.highlighted = False
        if self.present:
            self.neuron_button.setProperty("color", "p")
        else:
            self.neuron_button.setProperty("color", "a")
        self.neuron_button.setStyle(self.neuron_button.style())   # for some reason this is needed to actually change the color

    def set_text(self, text):
        self.neuron_key_button.setText(text)


class DashboardItem(QPushButton):
    qss = """
                 QPushButton{
                    border-radius: 4px;
                    min-width: 25px;
                    max-width: 25px;
                 }
                 QPushButton[color = "a"]{
                     background-color: red;
                 }
                 QPushButton[color = "p"]{
                     background-color: blue;
                 }
              """

    def __init__(self, i_from1, callback):
        super().__init__()
        self.i_from1 = i_from1
        self.clicked.connect(callback)
        self.present = False  # Todo: set at init??

    def set_present(self):
        self.present = True
        self.setProperty("color", "p")
        self.setStyle(self.style())   # for some reason this is needed to actually change the color

    def set_absent(self):
        self.present = False
        self.setProperty("color", "a")
        self.setStyle(self.style())   # for some reason this is needed to actually change the color


class ViewTab(QScrollArea):
    """
    This is the tab that controls viewing parameters
    """
    def __init__(self, controller, controlled_plot, default_values: Dict[str, bool]):
        """
        :param controller: main controller to report to
        :param controlled_plot: the instance of ImageRendering that is controlled by the commands in this tab
        :param default_values: dictionary (string->bool) for whether some check buttons should be checked.
            Should include "just_show_first_channel", "autolevels" and "overlay_mask_by_default".
        """
        super(ViewTab, self).__init__()
        self.controller = controller
        self.controlled_plot = controlled_plot
        as_points = self.controller.point_data

        view_tab_grid = QGridLayout()
        row = 0

        view_checkboxes_lay = QGridLayout()
        first_channel_only_checkbox = QCheckBox("Show only first channel")
        first_channel_only_checkbox.setChecked(int(default_values["just_show_first_channel"]))#MB changed bool to int
        first_channel_only_checkbox.toggled.connect(self.controller.toggle_first_channel_only)
        view_checkboxes_lay.addWidget(first_channel_only_checkbox, 0, 0)

        second_channel_only_checkbox = QCheckBox("Show only second channel")
        if first_channel_only_checkbox:
            second_channel_only_checkbox.setChecked(0)#MB changed bool to int
        second_channel_only_checkbox.toggled.connect(self.controller.toggle_second_channel_only)
        view_checkboxes_lay.addWidget(second_channel_only_checkbox, 0, 1)

        autolevels_checkbox = QCheckBox("Autolevels")
        autolevels_checkbox.setChecked(int(self.controlled_plot.figure.autolevels))
        autolevels_checkbox.toggled.connect(self.controlled_plot.change_autolevels)
        view_checkboxes_lay.addWidget(autolevels_checkbox, 0, 2)
        view_tab_grid.addLayout(view_checkboxes_lay, row, 0)
        row += 1

        gamma_lay = QVBoxLayout()
        gamma_slider = QSlider(Qt.Horizontal)
        gamma_slider.setMinimum(1)
        gamma_slider.setMaximum(100)
        gamma_slider.setValue(40)
        gamma_slider.valueChanged.connect(lambda val: self.controlled_plot.change_gamma(val))
        gamma_lay.addWidget(QLabel("Gamma"))
        gamma_lay.addWidget(gamma_slider)
        view_tab_grid.addLayout(gamma_lay, row, 0)
        row += 1

        blend_slider_lay = QGridLayout()
        blend_slider_lay.addWidget(QLabel("Blender Red"), 0, 0)
        blend_slider_lay.addWidget(QLabel("Blender Green"), 0, 1)
        blend_slider_r = QSlider(Qt.Horizontal)
        blend_slider_r.setMinimum(0)
        blend_slider_r.setMaximum(100)
        blend_slider_r.setValue(100)
        blend_slider_r.valueChanged.connect(lambda val: self.controlled_plot.change_blend_r(val))
        blend_slider_lay.addWidget(blend_slider_r, 1, 0)
        blend_slider_g = QSlider(Qt.Horizontal)
        blend_slider_g.setMinimum(0)
        blend_slider_g.setMaximum(100)
        blend_slider_g.setValue(100)
        blend_slider_g.valueChanged.connect(lambda val: self.controlled_plot.change_blend_g(val))
        blend_slider_lay.addWidget(blend_slider_g, 1, 1)
        view_tab_grid.addLayout(blend_slider_lay, row, 0)
        row += 1

        thres_slider_lay = QGridLayout()
        thres_slider_lay.addWidget(QLabel("Threshold Low"), 0, 0)
        thres_slider_lay.addWidget(QLabel("Threshold High"), 0, 1)
        thres_slider_l = QSlider(Qt.Horizontal)
        thres_slider_l.setMinimum(0)
        thres_slider_l.setMaximum(100)
        thres_slider_l.setValue(0)
        thres_slider_l.valueChanged.connect(lambda val: self.controlled_plot.change_low_thresh(val))
        thres_slider_h = QSlider(Qt.Horizontal)
        thres_slider_h.setMinimum(0)
        thres_slider_h.setMaximum(100)
        thres_slider_h.setValue(100)
        thres_slider_h.valueChanged.connect(lambda val: self.controlled_plot.change_high_thresh(val))
        thres_slider_lay.addWidget(thres_slider_l, 1, 0)
        thres_slider_lay.addWidget(thres_slider_h, 1, 1)
        view_tab_grid.addLayout(thres_slider_lay, row, 0)
        row += 1

        if as_points:
            # view_tab_grid.addWidget(QLabel("------------ Point mode ------------"), row, 0)
            # row += 1

            first_lay = QGridLayout()
            curr_CheckBox = QCheckBox("Overlay Current Points")
            curr_CheckBox.setChecked(True)
            curr_CheckBox.toggled.connect(self.controller.toggle_pts_overlay)
            first_lay.addWidget(curr_CheckBox, 0, 0)
            NN_CheckBox = QCheckBox("Overlay NN Prediction")
            NN_CheckBox.setChecked(True)
            NN_CheckBox.toggled.connect(self.controller.toggle_NN_overlay)
            first_lay.addWidget(NN_CheckBox, 0, 1)

            view_tab_grid.addLayout(first_lay, row, 0)
            row += 1

            overlay_tracks_lay = QGridLayout()
            overlay_tracks_CheckBox = QCheckBox("Overlay Tracks")
            overlay_tracks_CheckBox.setChecked(True)
            overlay_tracks_CheckBox.toggled.connect(self.controller.toggle_track_overlay)
            overlay_tracks_lay.addWidget(overlay_tracks_CheckBox, 0, 0, 1, 2)
            ov_tr_past = QLineEdit("-5")
            ov_tr_past.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
            ov_tr_past.setValidator(QtGui.QIntValidator(-15, 0))
            ov_tr_past.textChanged.connect(lambda x: self.controller.change_track_past(x))
            ov_tr_future = QLineEdit("5")
            ov_tr_future.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
            ov_tr_future.setValidator(QtGui.QIntValidator(0, 15))
            ov_tr_future.textChanged.connect(lambda x: self.controller.change_track_future(x))
            overlay_tracks_lay.addWidget(QLabel("Past"), 1, 0)
            overlay_tracks_lay.addWidget(QLabel("Future"), 1, 1)
            overlay_tracks_lay.addWidget(ov_tr_past, 2, 0)
            overlay_tracks_lay.addWidget(ov_tr_future, 2, 1)
            view_tab_grid.addLayout(overlay_tracks_lay, row, 0)
            row += 1

            adjLayout = QHBoxLayout()
            getadj = QLineEdit("-1")
            getadj.setValidator(QtGui.QIntValidator(-10, 10))
            getadj.textChanged.connect(lambda x: self._adjacent_changed(x))
            self.getadjlab = QLabel("-1")
            adj_CheckBox = QCheckBox("Overlay Adjacent Points")
            adj_CheckBox.toggled.connect(self.controller.toggle_adjacent_overlay)
            adjLayout.addWidget(adj_CheckBox)
            adjLayout.addWidget(getadj)
            adjLayout.addWidget(self.getadjlab)
            view_tab_grid.addLayout(adjLayout, row, 0)
            row += 1

        else:
            # view_tab_grid.addWidget(QLabel("------------ Mask mode ------------"), row, 0)
            # row += 1

            # SJR: I copied the code from above and got rid of point-specific stuff
            mask_boxes = QGridLayout()
            mask_checkbox = QCheckBox("Overlay Mask")
            mask_checkbox.setChecked(int(default_values["overlay_mask_by_default"]))
            mask_checkbox.toggled.connect(self.controller.toggle_mask_overlay)
            mask_boxes.addWidget(mask_checkbox, 0, 0)

            # MB: the following 3 lines si for overlaying the NN mask
            NNmask_checkbox = QCheckBox("Only NN mask")
            NNmask_checkbox.setChecked(False)
            NNmask_checkbox.toggled.connect(self.controller.toggle_NN_mask_only)
            mask_boxes.addWidget(NNmask_checkbox, 0, 1)

            aligned_checkbox = QCheckBox("Aligned")
            aligned_checkbox.setChecked(False)
            aligned_checkbox.toggled.connect(self.controller.toggle_display_alignment)
            mask_boxes.addWidget(aligned_checkbox, 0, 2)
            cropped_checkbox = QCheckBox("Cropped")
            cropped_checkbox.setChecked(False)
            cropped_checkbox.toggled.connect(self.controller.toggle_display_cropped)
            mask_boxes.addWidget(cropped_checkbox, 0, 3)
            view_tab_grid.addLayout(mask_boxes, row, 0)
            row += 1

            # SJR: code below is for blurring button and slider
            blur_checkbox = QCheckBox("Blur image")
            blur_checkbox.toggled.connect(self.controlled_plot.change_blur_image)
            view_tab_grid.addWidget(blur_checkbox, row, 0)
            row += 1
            blur_slider_lay = QGridLayout()
            blur_slider_lay.addWidget(QLabel("Blur sigm parameter"), 0, 0)
            blur_slider_lay.addWidget(QLabel("Blur bg_factor parameter"), 0, 1)
            blur_slider_s = QSlider(Qt.Horizontal)
            blur_slider_s.setMinimum(1)
            blur_slider_s.setMaximum(10)
            blur_slider_s.setValue(1)
            blur_slider_s.valueChanged.connect(lambda val: self.controlled_plot.change_blur_s(val))
            blur_slider_lay.addWidget(blur_slider_s, 1, 0)
            blur_slider_b = QSlider(Qt.Horizontal)
            blur_slider_b.setMinimum(0)
            blur_slider_b.setMaximum(100)
            blur_slider_b.setValue(25)
            blur_slider_b.valueChanged.connect(lambda val: self.controlled_plot.change_blur_b(val))
            blur_slider_lay.addWidget(blur_slider_b, 1, 1)
            view_tab_grid.addLayout(blur_slider_lay, row, 0)
            row += 1
            # SJR: code above is for blurring button and slider

        self.setWidgetResizable(True)
        self.setContentsMargins(5, 5, 5, 5)
        lay = QWidget()
        lay.setLayout(view_tab_grid)
        self.setWidget(lay)

        self.setWidgetResizable(True)
        self.setContentsMargins(5, 5, 5, 5)

    def _adjacent_changed(self, value):
        ok = self.controller.change_adjacent(value)
        if ok:
            self.getadjlab.setText(str(int(value)))


class AnnotationTab(QWidget):
    """
    This is the tab with point and mask annotation tools
    """

    def __init__(self, controller, frame_num:int, mask_threshold_for_new_region):
        """
        :param controller: main controller to report to
        :param frame_num: number of frames in video
        """
        super().__init__()
        self.controller = controller
        self.controller.highlighted_neuron_registered_clients.append(self)
        self.controller.mask_thres_registered_clients.append(self)
        self.controller.autocenter_registered_clients.append(self)
        as_points = self.controller.point_data
        self.as_points = as_points

        main_layout = QGridLayout()

        row = 0

        if as_points:
            # points_lab = QLabel("------------ Point mode ------------")
            # main_layout.addWidget(points_lab, row, 0)
            # row += 1

            subrow = 0
            approve_lay = QGridLayout()
            self.approve_lab = QLabel("Approve")
            approve_lay.addWidget(self.approve_lab, subrow, 0)
            subrow += 1

            self.delete_clear = QPushButton("Clear This frame")
            self.delete_clear.setStyleSheet("background-color: red")
            self.delete_clear.clicked.connect(self.controller.clear_frame_NN)
            approve_lay.addWidget(self.delete_clear, subrow, 0)
            subrow += 1

            self.delete_select = QPushButton("Delete Within")
            self.delete_select.setStyleSheet("background-color: red")
            self.delete_select.clicked.connect(self._selective_delete)
            self.delete_select.setEnabled(False)
            approve_lay.addWidget(self.delete_select, subrow, 0)
            self.delete_select_from = QLineEdit("0")
            self.delete_select_from.setValidator(QtGui.QIntValidator(0, frame_num - 1))
            approve_lay.addWidget(self.delete_select_from, subrow, 1)
            self.delete_select_to = QLineEdit(str(frame_num - 1))
            self.delete_select_to.setValidator(QtGui.QIntValidator(0, frame_num - 1))
            approve_lay.addWidget(self.delete_select_to, subrow, 2)
            subrow += 1

            self.approve_select = QPushButton("Approve Within")
            self.approve_select.setStyleSheet("background-color: green")
            self.approve_select.clicked.connect(self._selective_approve)
            self.approve_select.setEnabled(False)
            approve_lay.addWidget(self.approve_select, subrow, 0)
            self.approve_select_from = QLineEdit("0")
            self.approve_select_from.setValidator(QtGui.QIntValidator(0, frame_num - 1))
            approve_lay.addWidget(self.approve_select_from, subrow, 1)
            self.approve_select_to = QLineEdit(str(frame_num - 1))
            self.approve_select_to.setValidator(QtGui.QIntValidator(0, frame_num - 1))
            approve_lay.addWidget(self.approve_select_to, subrow, 2)
            subrow += 1

            main_layout.addLayout(approve_lay, row, 0)
            row += 1

            rotate_lay = QGridLayout()
            main_layout.addLayout(rotate_lay, row, 0)
            row += 1

            fh_checkbox = QCheckBox("Follow Highlighted")
            fh_checkbox.setChecked(True)
            fh_checkbox.toggled.connect(self.controller.toggle_z_follow_highlighted)
            main_layout.addWidget(fh_checkbox, row, 0)
            row += 1

            AutoCenterMode = QGridLayout()
            if True:
                AutoCenterWidget1 = QHBoxLayout()
                en_AutoCenter = QLabel("Auto Center Size:")
                getSize_AutoCenter = QLineEdit("3")
                getSize_AutoCenter.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
                getSize_AutoCenter.setValidator(QtGui.QIntValidator(0, 10))
                getSize_AutoCenter.textChanged.connect(self._set_xy_autocenter)
                getSize_AutoCenterz = QLineEdit("2")
                getSize_AutoCenterz.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
                getSize_AutoCenterz.setValidator(QtGui.QIntValidator(0, 5))
                getSize_AutoCenterz.textChanged.connect(self._set_z_autocenter)
                self.autocenterlabxy = QLabel("3")
                self.autocenterlabz = QLabel("2")
                AutoCenterWidget1_btn = QRadioButton("Nearest maximum intensity")

                AutoCenterWidget1.addWidget(en_AutoCenter)
                AutoCenterWidget1.addWidget(QLabel("X,Y:"))
                AutoCenterWidget1.addWidget(self.autocenterlabxy)
                AutoCenterWidget1.addWidget(getSize_AutoCenter)

                AutoCenterWidget1.addWidget(QLabel("Z:"))
                AutoCenterWidget1.addWidget(self.autocenterlabz)
                AutoCenterWidget1.addWidget(getSize_AutoCenterz)

                AutoCenterWidget2 = QHBoxLayout()
                getthres_peaks = QLineEdit("4")
                getthres_peaks.setValidator(QtGui.QIntValidator(0, 255))
                getthres_peaks.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
                getthres_peaks.textChanged.connect(self._set_peak_thresh)
                self.peak_thres_lab = QLabel("4")
                getsep_peaks = QLineEdit("2")
                getsep_peaks.setValidator(QtGui.QIntValidator(0, 10))
                getsep_peaks.textChanged.connect(self._set_peak_sep)
                getsep_peaks.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
                self.peak_sep_lab = QLabel("2")

                AutoCenterWidget2.addWidget(QLabel("Intensity Threshold:"))
                AutoCenterWidget2.addWidget(self.peak_thres_lab)
                AutoCenterWidget2.addWidget(getthres_peaks)

                AutoCenterWidget2.addWidget(QLabel("Minimum Separation:"))
                AutoCenterWidget2.addWidget(self.peak_sep_lab)
                AutoCenterWidget2.addWidget(getsep_peaks)

                AutoCenterWidget2_btn = QRadioButton("Nearest Peak")
                AutoCenterWidget2_btn.setChecked(True)
                AutoCenterWidget2_btn.toggled.connect(self.controller.toggle_autocenter_peakmode)

                autocenter_enable_lay = QHBoxLayout()
                autocenter_enable_lay.addWidget(QLabel("Autocenter Enabled [A]:"))
                self.auto_en_lab = QLabel("  ")
                self.auto_en_lab.setStyleSheet("background-color: green; height: 5px; width: 5px;min-width: 5px;")
                autocenter_enable_lay.addWidget(self.auto_en_lab)

                rowi = 0
                AutoCenterMode.addLayout(autocenter_enable_lay, rowi, 0, 1, 2)
                rowi += 1
                AutoCenterMode.addWidget(QLabel("  "), rowi, 0)
                AutoCenterMode.addWidget(QLabel("Select Auto Center Mode"), rowi, 1)
                rowi += 1
                AutoCenterMode.addWidget(AutoCenterWidget2_btn, rowi, 1)
                rowi += 1
                AutoCenterMode.addLayout(AutoCenterWidget2, rowi, 1)
                rowi += 1
                AutoCenterMode.addWidget(AutoCenterWidget1_btn, rowi, 1)
                rowi += 1
                AutoCenterMode.addLayout(AutoCenterWidget1, rowi, 1)
                rowi += 1
            main_layout.addLayout(AutoCenterMode, row, 0)
            row += 1

        else:
            # mask_lab = QLabel("------------ Mask mode ------------")
            # main_layout.addWidget(mask_lab, row, 0)
            # row += 1

            # SJR: Annotate mask section in the annotate tab
            mask_annotation_Layout = QGridLayout()
            subrow=0
            mask_annotation_checkbox = QCheckBox("Mask annotation mode")
            mask_annotation_checkbox.toggled.connect(self.controller.toggle_mask_annotation_mode)
            mask_annotation_Layout.addWidget(mask_annotation_checkbox,subrow, 0)
            self.mask_annotation_thresh = QLineEdit(mask_threshold_for_new_region)
            self.mask_annotation_thresh.setValidator(QtGui.QIntValidator(1, 1000))
            self.mask_annotation_thresh.textChanged.connect(lambda x: self.controller.set_mask_annotation_threshold(x))
            mask_annotation_Layout.addWidget(self.mask_annotation_thresh,subrow, 1)
            mask_annotation_thresh_label = QLabel("Treshold for adding regions")
            mask_annotation_Layout.addWidget(mask_annotation_thresh_label,subrow, 2)
            subrow += 1
            box_mode_checkbox = QCheckBox("boxing mode")
            box_mode_checkbox.toggled.connect(self.controller.toggle_box_mode)
            mask_annotation_Layout.addWidget(box_mode_checkbox,subrow, 0)
            self.box_dimensions = QLineEdit("1,1,1-0")
            self.box_dimensions.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
            self.box_dimensions.textChanged.connect(lambda x: self.controller.set_box_dimensions(x))
            mask_annotation_Layout.addWidget(self.box_dimensions,subrow, 1)
            boxing_dim_label = QLabel("Box details (W,H,D-box_id)")
            mask_annotation_Layout.addWidget(boxing_dim_label,subrow, 2)

            main_layout.addLayout(mask_annotation_Layout, row, 0)
            row += 1


            mask_buttons = QGridLayout()
            subrow=0
            self.change_within_checkbox = QCheckBox("Change within")
            mask_buttons.addWidget(self.change_within_checkbox, subrow, 0)
            self.mask_change_from = QLineEdit("0")
            self.mask_change_from .setValidator(QtGui.QIntValidator(0, frame_num -1))
            mask_buttons.addWidget(self.mask_change_from , subrow, 1)
            mask_buttons.addWidget(QLabel("to:"), subrow, 2)
            self.mask_change_to = QLineEdit(str(frame_num))
            self.mask_change_to.setValidator(QtGui.QIntValidator(0, frame_num))
            mask_buttons.addWidget(self.mask_change_to, subrow, 3)

            subrow += 1

            self.renumber_mask_obj = QPushButton("Renumber")
            self.renumber_mask_obj.setStyleSheet("background-color: green")
            self.renumber_mask_obj.clicked.connect(self._selective_renumber)
            self.renumber_mask_obj.setEnabled(False)
            mask_buttons.addWidget(self.renumber_mask_obj, subrow, 0)
            self.delete_mask_obj = QPushButton("Delete")
            self.delete_mask_obj.setStyleSheet("background-color: red")
            self.delete_mask_obj.clicked.connect(self.controller.delete_mask_obj)
            self.delete_mask_obj.setEnabled(False)
            mask_buttons.addWidget(self.delete_mask_obj, subrow, 1)
            main_layout.addLayout(mask_buttons, row, 0)
            row += 1

            Permute_buttons = QGridLayout()
            self.cell_permutation_entry = QLineEdit("0")
            self.cell_permutation_entry.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
            Permute_buttons.addWidget(QLabel("Enter cell numbers separated with ,"), 0, 0)
            Permute_buttons.addWidget(self.cell_permutation_entry, 0, 1)
            Permute_btn = QPushButton("Permute")
            Permute_btn.setStyleSheet("background-color: yellow")
            Permute_btn.clicked.connect(self._Permutation_fun)
            Permute_buttons.addWidget(Permute_btn, 0, 2)

            main_layout.addLayout(Permute_buttons, row, 0)

        self.setLayout(main_layout)

    def _selective_delete(self):
        fro, to = int(self.delete_select_from.text()), int(self.delete_select_to.text())
        self.controller.clear_NN_selective(fro, to)

    def _selective_approve(self):
        fro, to = int(self.approve_select_from.text()), int(self.approve_select_to.text())
        self.controller.approve_selective(fro, to)

    def _selective_renumber(self):
        fro, to = int(self.mask_change_from.text()), int(self.mask_change_to.text())
        if self.change_within_checkbox.checkState():
            self.controller.renumber_All_mask_instances(fro,to)
        else:
            self.controller.renumber_mask_obj()

    def _Permutation_fun(self):
        cells = self.cell_permutation_entry.text()
        cells = cells.split(',')
        cellarray = [int(k) for k in cells]
        cellarraycycle = np.zeros(len(cellarray)+1)
        cellarraycycle[:-1]=cellarray
        cellarraycycle[len(cellarray)] = cellarray[0]
        self.controller.permute_masks(cellarraycycle)

    def _set_xy_autocenter(self, value):
        try:
            int(value)
        except:
            return
        self.autocenterlabxy.setText(value)
        self.controller.set_autocenter(int(value), z=False)

    def _set_z_autocenter(self, value):
        try:
            int(value)
        except:
            return
        self.autocenterlabz.setText(value)
        self.controller.set_autocenter(int(value), z=True)

    def _set_peak_thresh(self, value):
        try:
            value = int(value)
        except:
            return
        self.controller.set_peak_threshold(value)
        self.peak_thres_lab.setText(str(value))

    def _set_peak_sep(self, value):
        try:
            value = int(value)
        except:
            return
        self.controller.set_peak_sep(value)
        self.peak_sep_lab.setText(str(value))

    def change_mask_thresh(self, value):
        self.mask_annotation_thresh.blockSignals(True)
        self.mask_annotation_thresh.setText(str(value))
        self.mask_annotation_thresh.blockSignals(False)

    def change_highlighted_neuron(self, high: int=None, unhigh:int=None, high_key=None, **kwargs):
        """
        Callback when the highlighted neuron changes.
        :param high: neuron id (from 1)
        :param unhigh: neuron id (from 1)
        :param high_key: key assigned to highlighted neuron (for display on the "approve" label)
        """
        if high is None and unhigh is not None:   # just unghlight
            if self.as_points:
                self.approve_lab.setText("Approve")
                self.delete_select.setEnabled(False)
                self.approve_select.setEnabled(False)
            else:
                self.renumber_mask_obj.setEnabled(False)
                self.delete_mask_obj.setEnabled(False)

        elif high is not None:
            if high_key is None:
                key_name = " "
            else:
                key_name = " [" + high_key + "]"
            if self.as_points:
                self.approve_lab.setText("Approve: " + str(high) + key_name)
                self.delete_select.setEnabled(True)
                self.approve_select.setEnabled(True)
            else:
                self.renumber_mask_obj.setEnabled(True)
                self.delete_mask_obj.setEnabled(True)

    def change_autocenter_mode(self, on:bool):
        if not self.as_points:
            return
        if on:
            self.auto_en_lab.setStyleSheet("background-color: green; height: 15px; width: 5px;min-width: 5px;")
        else:
            self.auto_en_lab.setStyleSheet("background-color: red; height: 15px; width: 5px;min-width: 5px;")


class NNControlTab(QWidget):
    """
    This is the tab that deals with NNs, launching them, retrieving their results...
    """
    def __init__(self, controller, data_name:str):
        """
        :param controller: main controller to report to
        :param data_name: name of the dataset
        """
        super().__init__()
        self.controller = controller
        self.controller.NN_instances_registered_clients.append(self)
        self.controller.validation_set_registered_clients.append(self)
        self.data_name = data_name
        as_points = self.controller.point_data
        self.as_points = as_points

        main_layout = QGridLayout()

        row = 0

        if as_points:
            #CFP-HELP:in the link below I think I should give the controller to the TrackTab object, basically some communication channel from this widget to everything else.
            self.widget=TrackTab(None)
            dummylayout=QGridLayout()
            dummylayout.addWidget(self.widget)
            self.setLayout(dummylayout)
            
            #CFP-HELP:Here I need a QCombobox to select a helper(NN result or output of other algorithms) then give it to NN_pointdat.
            """
            lab = QLabel("Select NN points")
            main_layout.addWidget(lab, row, 0, 1, 2)
            row += 1

            self.NN_pt_select = QComboBox()
            self.NN_pt_select.addItem("None")
            self.NN_pt_select.currentTextChanged.connect(self._select_pt_instance)
            main_layout.addWidget(self.NN_pt_select, row, 0, 1, 2)
            row += 1

            NN_train = QPushButton("Train Point Detection Neural Network")
            NN_train.clicked.connect(lambda: self._run_script(mask=False))
            NN_train_fol = QPushButton("Output Train NNpts folder")
            NN_train_fol.clicked.connect(lambda: self._run_NN(mask=False, fol=True))
            main_layout.addWidget(NN_train, row, 0, 1, 2)
            row += 1
            main_layout.addWidget(NN_train_fol, row, 0, 1, 2)
            row += 1
            """
        else:
            # and this is to select NN from which to load masks
            lab = QLabel("Select NN masks")
            main_layout.addWidget(lab, row, 0, 1, 2)
            row += 1

            self.NN_mask_select = QComboBox()
            self.NN_mask_select.currentTextChanged.connect(self._select_mask_instance)
            main_layout.addWidget(self.NN_mask_select, row, 0, 1, 2)
            row += 1


            self.Exempt_Neurons = QLineEdit("0")
            self.Exempt_Neurons.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
            #self.PostProc_Mode = QLineEdit("1")
            #self.PostProc_Mode.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
            main_layout.addWidget(QLabel("Neurons exempt from postprocessing:"),row,0,1, 1)
            main_layout.addWidget(self.Exempt_Neurons,row,1, 1, 1)
            row += 1

            #tab for determining different post processing modes
            PostProcess_mask = QPushButton("Post-process masks")
            PostProcess_mask.setStyleSheet("background-color: yellow")
            PostProcess_mask.clicked.connect(self._Postprocess_NN_masks)
            main_layout.addWidget(PostProcess_mask,row, 1,1,1)
            self.PostProc_Mode = QLineEdit("1")
            self.PostProc_Mode.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
            main_layout.addWidget(self.PostProc_Mode,row,0, 1, 1)
            row += 1

            #MB added to save the selected predicted masks as the GT masks
            approve_mask = QPushButton("Approve masks")
            approve_mask.setStyleSheet("background-color: green")
            approve_mask.clicked.connect(self.controller.approve_NN_masks)
            main_layout.addWidget(approve_mask,row, 0,1, 2)
            row += 1

            # MB added: to get the validation frames ids:
            #CFP: made this conditional to masks
            val_frame_box = QtHelpers.CollapsibleBox("Validation frames id:")  # MB added
            lay = QVBoxLayout()
            self.val_set_display = QLabel()
            self.val_set_display.setText("unknown at init")
            lay.addWidget(self.val_set_display)
            val_frame_box.setContentLayout(lay)
            main_layout.addWidget(val_frame_box, row, 0, 1, 2)
            row += 1

            main_layout.addWidget(QLabel("--------"), row, 0, 1, 2)
            row += 1
        
            #CFP made this conditional
            NN_Check = QPushButton("Check NN progress")
            NN_Check.clicked.connect(self.controller.check_NN_run)
            main_layout.addWidget(NN_Check, row, 0, 1, 2)
            row += 1

            main_layout.addWidget(QLabel("--------"), row, 0, 1, 2)
            row += 1
            
            old_train_checkbox = QCheckBox("Use old train set")
            old_train_checkbox.toggled.connect(self.controller.toggle_old_trainset)
            main_layout.addWidget(old_train_checkbox, row, 0)
            deform_checkbox = QCheckBox("Add deformation")
            deform_checkbox.toggled.connect(self.controller.toggle_add_deformation)
            main_layout.addWidget(deform_checkbox,row , 1)
            row += 1
            main_layout.addWidget(QLabel("Number of target frames:"), row, 0)
            self.targset = QLineEdit("5")
            self.targset.setValidator(QtGui.QIntValidator(0, 200))
            main_layout.addWidget(self.targset, row, 1)
            row += 1

            subrow = 0
            options_checkboxes_lay = QGridLayout()

            options_checkboxes_lay.addWidget(QLabel("val:"), subrow, 0)
            self.valset = QLineEdit("1")
            self.valset.setValidator(QtGui.QIntValidator(0, 150))
            options_checkboxes_lay.addWidget(self.valset, subrow, 1)
            options_checkboxes_lay.addWidget(QLabel("train:"), subrow, 2)
            self.trainset = QLineEdit("6")
            self.trainset.setValidator(QtGui.QIntValidator(0, 150))
            options_checkboxes_lay.addWidget(self.trainset, subrow, 3)

            options_checkboxes_lay.addWidget(QLabel("epochs:"), subrow, 4)
            self.epochs = QLineEdit("100")
            self.epochs.setValidator(QtGui.QIntValidator(0, 1000))
            options_checkboxes_lay.addWidget(self.epochs, subrow, 5)

            subrow += 2
            main_layout.addLayout(options_checkboxes_lay, row, 0)
            row += 1

            NN_train = QPushButton("Train Mask Prediction Neural Network")
            NN_train.clicked.connect(lambda: self._run_NN(mask=True))
            NN_train_fol = QPushButton("Output Train NNmasks folder")
            NN_train_fol.clicked.connect(lambda: self._run_NN(mask=True, fol=True))
            main_layout.addWidget(NN_train, row, 0, 1, 2)
            row += 1
            main_layout.addWidget(NN_train_fol, row, 0, 1, 2)
            row += 1

            main_layout.addWidget(QLabel("Settings"), row, 0, 1, 2)
            row += 1

            self.NN_model_select = QComboBox()
            self.NN_instance_select = QComboBox()
            self.change_NN_instances()
            self.setup_NNmodels()   # populate the self.NN_model_select with the existing models
            self.populate_NNinstances()
            self.NN_model_select.currentTextChanged.connect(self.populate_NNinstances)
            main_layout.addWidget(self.NN_model_select, row, 0)
            main_layout.addWidget(self.NN_instance_select, row, 1)
            row += 1

        self.setLayout(main_layout)

    @property
    def NNinstances(self):
        """
        Gets the NNinstances dict directly from the controller.
        Warning, this means self should never modify self.NNinstances.
        """
        return self.controller.NNinstances

    def _select_pt_instance(self, txt):
        if txt == "":
            net, instance = "", None
        else:
            net, instance = txt.split(" ")
        self.controller.select_NN_instance_points(net, instance)

    def _select_mask_instance(self, txt):
        if txt == "":
            net, instance = "", None
        else:
            net, instance = txt.split(" ")
        self.controller.select_NN_instance_masks(net, instance)
        
    def _run_script(self):
        pass

    def _run_NN(self, mask=False, fol=False):
        modelname = self.NN_model_select.currentText()
        instancename = self.NN_instance_select.currentText()
        if instancename == "new":
            text, ok = QInputDialog.getText(self, 'Name your instance', 'Instance Name: ')   # TODO AD is self ok here or should it be another class of Qt thing?
            if not ok:
                return
            if any([el in text for el in ["\n"," ",".","/","\t"]]) or text=="new":
                errdial=QErrorMessage()
                errdial.showMessage('Invalid Instance Name')
                errdial.exec_()
            else:
                instancename=text
        msgBox = QMessageBox()
        msgBox.setText("Confirm Neural Network Run:\n"+modelname+" on "+self.data_name+" named "+instancename)
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        confirmation = msgBox.exec()
        if confirmation == QMessageBox.Ok:
            if mask:
                runres, msg = self.controller.run_NN_masks(modelname,instancename,fol,int(self.epochs.text()),int(self.trainset.text()),int(self.valset.text()),int(self.targset.text()))
            else:
                runres, msg = self.controller.run_NN_points(modelname,instancename,fol)
            if not runres:
                errdial=QErrorMessage()
                errdial.showMessage('Run Failed:\n'+msg)
                errdial.exec_()
            else:
                dial=QMessageBox()
                dial.setText('Run Success\n'+msg)
                dial.exec_()

    def setup_NNmodels(self):
        """
        Populates the list of NN models with models existing in self.controller.
        Only for initialization (list of NN models is fixed).
        """
        self.NN_model_select.clear()
        for modelname in self.controller.NNmodels:
            self.NN_model_select.addItem(modelname)

    def populate_NNinstances(self):
        """
        Populates the list of selectable NN instances with existing instances for current model and new.
        """
        self.NN_instance_select.clear()
        currname = self.NN_model_select.currentText()
        self.NN_instance_select.addItem("new")
        if currname in self.NNinstances:
            for instance in self.NNinstances[currname]:
                self.NN_instance_select.addItem(instance)

    def change_NN_instances(self):
        """
        Callback when new instances of NNs are created (or deleted?).
        """
        self.populate_NNinstances()

        if not self.as_points:
            self.NN_mask_select.clear()
            self.NN_mask_select.addItem("")
        else:
            self.NN_pt_select.clear()
            self.NN_pt_select.addItem("")
        for net,insts in self.NNinstances.items():
            for inst in insts:
                if not self.as_points:
                    self.NN_mask_select.addItem(net+" "+inst)
                elif self.controller.NN_inst_has_pointdat(net, inst):
                    self.NN_pt_select.addItem(net+" "+inst)

    def change_validation_set(self, validation_set):
        self.val_set_display.setText(str(validation_set))

    def _Postprocess_NN_masks(self):
        ExNeu = self.Exempt_Neurons.text()
        ExNeu = ExNeu.split(',')
        ExNeu = [int(n) for n in ExNeu ]

        Mode = int(self.PostProc_Mode.text())
        Modes = set([1,2,3,4,5])
        assert Mode in Modes, "Acceptable modes are 1, 2, 3, 4, and 5"
        if Mode ==1:
            self.controller.post_process_NN_masks(ExNeu)
        if Mode ==2:
            self.controller.post_process_NN_masks2(ExNeu)
        if Mode ==3:
            self.controller.post_process_NN_masks3(ExNeu)
        if Mode ==4:
            self.controller.post_process_NN_masks4(ExNeu)
        if Mode ==5:
            self.controller.post_process_NN_masks5(ExNeu)

class SelectionTab(QWidget):
    """
    This is the tab that allows to select subsets of data
    """
    def __init__(self, controller):
        """
        :param controller: main controller to report to
        """
        super().__init__()
        self.controller = controller

        main_layout = QVBoxLayout()

        all_correct_btn = QPushButton("Flag all selected frames as ground truth")
        all_correct_btn.clicked.connect(self.controller.flag_all_selected_as_gt)
        main_layout.addWidget(all_correct_btn)

        use_as_ref_btn = QPushButton("Use this frame as reference")  # (for registration)
        use_as_ref_btn.clicked.connect(self.controller.use_current_as_ref)
        main_layout.addWidget(use_as_ref_btn)

        selection_lay = QGridLayout()
        # select proportion of frames
        self.frac_entry = QLineEdit("100")
        self.frac_entry.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
        self.frac_entry.setValidator(QtGui.QIntValidator(0, 100))

        selection_lay.addWidget(self.frac_entry, 0, 0)
        selection_lay.addWidget(QLabel("% of"), 0, 1)

        #MB added to selet individual frames manually
        self.fr_num_entry = QLineEdit("0")
        self.fr_num_entry.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
        selection_lay.addWidget(QLabel("Enter frame numbers separated with ,"), 1, 0)
        selection_lay.addWidget(self.fr_num_entry, 1, 1)

        # select population of frames from which to select this proportion
        buttons_layout = QVBoxLayout()
        self.population_buttons = QButtonGroup()  # widget
        all_radiobtn = QRadioButton("all")
        all_radiobtn.setChecked(True)
        self.population_buttons.addButton(all_radiobtn)
        buttons_layout.addWidget(all_radiobtn)
        segmented_radiobtn = QRadioButton("segmented")
        segmented_radiobtn.setChecked(False)
        self.population_buttons.addButton(segmented_radiobtn)
        buttons_layout.addWidget(segmented_radiobtn)
        non_segmented_radiobtn = QRadioButton("non segmented")
        non_segmented_radiobtn.setChecked(False)
        self.population_buttons.addButton(non_segmented_radiobtn)
        buttons_layout.addWidget(non_segmented_radiobtn)
        ground_truth_radiobtn = QRadioButton("ground truth")
        ground_truth_radiobtn.setChecked(False)
        self.population_buttons.addButton(ground_truth_radiobtn)
        buttons_layout.addWidget(ground_truth_radiobtn)
        non_ground_truth_radiobtn = QRadioButton("non ground truth")
        non_ground_truth_radiobtn.setChecked(False)
        self.population_buttons.addButton(non_ground_truth_radiobtn)
        buttons_layout.addWidget(non_ground_truth_radiobtn)
        single_radiobtn = QRadioButton("manual selection")
        single_radiobtn.setChecked(False)
        self.population_buttons.addButton(single_radiobtn)
        buttons_layout.addWidget(single_radiobtn)


        selection_lay.addLayout(buttons_layout, 0, 2)


        select_btn = QPushButton("Select")
        select_btn.clicked.connect(self._frame_fraction_fun)
        selection_lay.addWidget(select_btn, 0, 3)

        main_layout.addLayout(selection_lay)
        self.setLayout(main_layout)

    def _frame_fraction_fun(self):
        frac = self.frac_entry.text()
        frames = self.fr_num_entry.text()
        frames = frames.split(',')

        Tot_fr = set()
        for i in range(len(frames)):
            interval_str = frames[i].split('-')
            if len(interval_str)>1:
                interval_array = range(int(interval_str[0]),int(interval_str[1])+1)
                Tot_fr = Tot_fr.union(set(interval_array))
            else:
                Tot_fr = Tot_fr.union(set(interval_str))

        Tot_fr_final = [int(fr) for fr in Tot_fr]
        self.controller.select_frames(float(frac)/100, Tot_fr_final, self.population_buttons.checkedButton().text())


class MaskProcessingTab(QWidget):
    """
    This is the tab that controls all processes specific to data with masks (though not just masks themselves): segmentation, clustering...
    """

    def __init__(self, controller):
        """
        :param controller: main controller to report to
        """
        super().__init__()
        self.controller = controller
        self.seg_params = self.controller.get_seg_params()
        self.cluster_params = self.controller.get_cluster_params()
        # Warning, these two are the real dict from the dataset, can be modified but it's useless to try reassigning the variables
        # todo AD: could also be refactored in terms of just calling the controller and registering to the controller
        #  for changes of params (but it's less practical)

        main_layout = QGridLayout()

        # segmentation parameters
        seg_param_box = QtHelpers.CollapsibleBox("Segmentation parameters")
        lay = QVBoxLayout()
        seg_param_tree = ParameterTree()
        params = [{'name': k, 'value': v, **Parameters.pyqt_param_keywords(k)} for k, v in
                  self.seg_params.items()]
        seg_params = Parameter.create(name='params', type='group', children=params)
        seg_param_tree.setParameters(seg_params, showTop=False)
        lay.addWidget(seg_param_tree)

        def change_seg_pars(param, changes):  # Todo: as method
            for param, change, data in changes:
                self.seg_params.update({param.name(): data})

        seg_params.sigTreeStateChanged.connect(change_seg_pars)

        seg_param_box.setContentLayout(lay)
        main_layout.addWidget(seg_param_box)

        # Segmentation buttons
        init_seg_btn = QPushButton("Test segmentation on current frame")
        init_seg_btn.clicked.connect(self.controller.test_segmentation_params)
        main_layout.addWidget(init_seg_btn)
        seg_lay = QHBoxLayout()
        coarse_seg_btn = QCheckBox("Coarse segmentation")
        coarse_seg_btn.setChecked(False)
        coarse_seg_btn.toggled.connect(self.controller.toggle_coarse_seg_mode)
        seg_btn = QPushButton("Segmentation")
        seg_btn.clicked.connect(self.controller.segment)
        seg_lay.addWidget(seg_btn)
        seg_lay.addWidget(coarse_seg_btn)
        dum = QWidget()
        dum.setLayout(seg_lay)
        main_layout.addWidget(dum)

        ftr_lay = QHBoxLayout()
        ftr_from_seg_check = QCheckBox("from segmentations")
        ftr_from_seg_check.setChecked(False)
        ftr_from_seg_check.toggled.connect(self.controller.toggle_use_seg_for_feature)
        ftr_btn = QPushButton("Extract features")
        ftr_btn.clicked.connect(self.controller.extract_features)
        ftr_lay.addWidget(ftr_btn)
        ftr_lay.addWidget(ftr_from_seg_check)
        dum2 = QWidget()
        dum2.setLayout(ftr_lay)
        main_layout.addWidget(dum2)

        # clustering parameters
        cluster_param_box = QtHelpers.CollapsibleBox("Clustering parameters")
        lay = QVBoxLayout()
        cluster_param_tree = ParameterTree()

        params = [{'name': k, 'value': v, **Parameters.pyqt_param_keywords(k)} for k, v in
                  self.cluster_params.items()]

        cluster_params = Parameter.create(name='params', type='group', children=params)
        cluster_param_tree.setParameters(cluster_params, showTop=False)
        lay.addWidget(cluster_param_tree)

        def change_cluster_pars(param, changes):  # Todo: as method
            for param, change, data in changes:
                self.cluster_params.update({param.name(): eval(data) if isinstance(data, str) else data})

        cluster_params.sigTreeStateChanged.connect(change_cluster_pars)

        cluster_param_box.setContentLayout(lay)
        main_layout.addWidget(cluster_param_box)

        # remaining processing buttons
        cluster_btn = QPushButton("Cluster")
        cluster_btn.clicked.connect(self.controller.cluster)
        main_layout.addWidget(cluster_btn)
        clf_btn = QPushButton("Classify")
        clf_btn.clicked.connect(self.controller.classify)
        main_layout.addWidget(clf_btn)
        reg_btn = QPushButton("Auto add to ground truth")
        reg_btn.clicked.connect(self.controller.auto_add_gt_by_registration)
        main_layout.addWidget(reg_btn)
        rotation_btn = QPushButton("Compute rotation")
        rotation_btn.clicked.connect(self.controller.compute_rotation)
        main_layout.addWidget(rotation_btn)

        coarot_btn = QPushButton("Coarse segmentation + rotation")
        coarot_btn.clicked.connect(self.controller.segment)
        coarot_btn.clicked.connect(self.controller.compute_rotation)
        main_layout.addWidget(coarot_btn)

        crop_btn = QPushButton("Crop")
        crop_btn.clicked.connect(self.controller.define_crop_region)
        main_layout.addWidget(crop_btn)

        self.setLayout(main_layout)


class SavingTab(QWidget):
    """
    This is the tab for saving etc
    """

    def __init__(self, controller):
        """
        :param controller: main controller to report to
        """
        super().__init__()
        self.controller = controller

        main_layout = QGridLayout()

        row = 0
        en_AutoSave = QCheckBox("Enable Auto Save")
        en_AutoSave.setChecked(False)
        en_AutoSave.toggled.connect(self.controller.toggle_autosave)
        main_layout.addWidget(en_AutoSave, row, 0)
        row += 1

        push_button = QPushButton("ReCalculate Calcium Intensity")
        push_button.clicked.connect(self.controller.update_ci)
        push_button.setStyleSheet("background-color: blue")
        main_layout.addWidget(push_button, row, 0)
        row += 1

        push_button = QPushButton("Save and Repack")
        push_button.clicked.connect(self.controller.save_and_repack)
        main_layout.addWidget(push_button, row, 0)
        row += 1

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.controller.save_status)
        save_button.setStyleSheet("background-color: green")
        main_layout.addWidget(save_button, row, 0)
        row += 1

        self.setLayout(main_layout)

class PreProcessTab(QWidget):
        """
        MB: This is the tab for preprocessing and saving an a separate file
        """
        def __init__(self, controller, frame_num:int, mask_threshold_for_new_region):
            """
            :param controller: main controller to report to
            :param frame_num: number of frames in video
            """
            super().__init__()
            self.controller = controller

            preproc_tab_grid = QGridLayout()
            row = 0

            Import_section = QLabel("------------ Import Section -----------")
            preproc_tab_grid.addWidget(Import_section, row, 0)
            row += 1

            subrow = 0
            load_mask_lay = QGridLayout()
            self.import_address = QLineEdit("0")
            self.import_address.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
            load_mask_lay.addWidget(QLabel("Address of the imported masks:"), subrow, 0)
            load_mask_lay.addWidget(self.import_address, subrow, 1)
            subrow += 1

            reverse_transform_checkbox = QCheckBox("Reverse transform")
            reverse_transform_checkbox.toggled.connect(self.controller.toggle_reverse_transform)
            load_mask_lay.addWidget(reverse_transform_checkbox, subrow, 0)
            import_btn = QPushButton("import")
            import_btn.clicked.connect(self.import_file)
            load_mask_lay.addWidget(import_btn, subrow, 1)
            import_green_btn = QPushButton("import as green mask")
            import_green_btn.clicked.connect(self.import_file_green)
            import_green_btn.setStyleSheet("background-color: green")
            load_mask_lay.addWidget(import_green_btn, subrow+1, 0)

            preproc_tab_grid.addLayout(load_mask_lay,row,0)
            row += 1

            Export_section = QLabel("------------ Export Section --------------")
            preproc_tab_grid.addWidget(Export_section, row, 0)
            row += 1

            subrow = 0
            save_checkboxes_lay = QGridLayout()
            save_rotation_crop_checkbox = QCheckBox("Rotate and crop")
            save_rotation_crop_checkbox.toggled.connect(self.controller.toggle_save_crop_rotate)
            save_checkboxes_lay.addWidget(save_rotation_crop_checkbox, subrow, 0)

            auto_delete_checkbox = QCheckBox("auto delete")
            auto_delete_checkbox.toggled.connect(self.controller.toggle_auto_delete)
            save_checkboxes_lay.addWidget(auto_delete_checkbox,subrow , 1)
            subrow += 2

            save_1channel_checkbox = QCheckBox("save red channel")
            #bg_subtraction_checkbox.setChecked(int(default_values["just_show_first_channel"]))#MB changed bool to int
            save_1channel_checkbox.toggled.connect(self.controller.toggle_save_1st_channel)
            save_checkboxes_lay.addWidget(save_1channel_checkbox,subrow , 0)

            save_green_checkbox = QCheckBox("save green channel")
            save_green_checkbox.toggled.connect(self.controller.toggle_save_green_channel)
            save_checkboxes_lay.addWidget(save_green_checkbox,subrow , 1)


            subrow += 2

            Blur_checkbox = QCheckBox("Blur")
            Blur_checkbox.toggled.connect(self.controller.toggle_save_blurred)
            save_checkboxes_lay.addWidget(Blur_checkbox, subrow, 0)

            save_checkboxes_lay.addWidget(QLabel("background factor:"), subrow, 1)
            self.bg_blur = QLineEdit("40")
            self.bg_blur.setValidator(QtGui.QIntValidator(0, 100))
            save_checkboxes_lay.addWidget(self.bg_blur, subrow, 2)
            save_checkboxes_lay.addWidget(QLabel("sigma:"), subrow, 3)
            self.sd_blur = QLineEdit("6")
            self.sd_blur.setValidator(QtGui.QIntValidator(0, 10))
            save_checkboxes_lay.addWidget(self.sd_blur, subrow, 4)
            subrow += 1

            bg_subtraction_checkbox = QCheckBox("Subtract bg")
            bg_subtraction_checkbox.toggled.connect(self.controller.toggle_save_subtracted_bg)
            save_checkboxes_lay.addWidget(bg_subtraction_checkbox, subrow, 0)
            save_checkboxes_lay.addWidget(QLabel("background:"), subrow, 1)
            self.bg_subt = QLineEdit("1")
            self.bg_subt.setValidator(QtGui.QIntValidator(0, 10))
            save_checkboxes_lay.addWidget(self.bg_subt, subrow, 2)
            subrow += 2

            resize_img_checkbox = QCheckBox("Resize image")
            resize_img_checkbox.toggled.connect(self.controller.toggle_save_resized_img)
            save_checkboxes_lay.addWidget(resize_img_checkbox, subrow, 0)
            save_checkboxes_lay.addWidget(QLabel("Width:"), subrow, 1)
            self.resized_width = QLineEdit("16")
            self.resized_width.setValidator(QtGui.QIntValidator(0, 512))
            save_checkboxes_lay.addWidget(self.resized_width, subrow, 2)

            save_checkboxes_lay.addWidget(QLabel("height:"), subrow, 3)
            self.resized_height = QLineEdit("32")
            self.resized_height.setValidator(QtGui.QIntValidator(0, 512))
            save_checkboxes_lay.addWidget(self.resized_height, subrow, 4)

            preproc_tab_grid.addLayout(save_checkboxes_lay, row, 0)
            row += 1



            subrow = 0
            approve_lay = QGridLayout()


            approve_lay.addWidget(QLabel("Choose frames from:"), subrow, 0)
            self.frame_from = QLineEdit("0")
            self.frame_from.setValidator(QtGui.QIntValidator(0, frame_num - 1))
            approve_lay.addWidget(self.frame_from, subrow, 1)
            approve_lay.addWidget(QLabel("to:"), subrow, 2)
            self.frame_to = QLineEdit(str(frame_num))
            self.frame_to.setValidator(QtGui.QIntValidator(0, frame_num))
            approve_lay.addWidget(self.frame_to, subrow, 3)
            subrow += 1

            approve_lay.addWidget(QLabel("Delete frames:"), subrow, 0)

            self.delete_fr = QLineEdit(str(frame_num)+","+str(frame_num+1))
            self.delete_fr.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")
            approve_lay.addWidget(self.delete_fr, subrow, 1)
            approve_lay.addWidget(QLabel("and intervals:"), subrow, 2)
            self.delete_inter = QLineEdit(str(frame_num)+"-"+str(frame_num+1))
            self.delete_inter.setStyleSheet("height: 15px; width: 5px;min-width: 5px;")

            approve_lay.addWidget(self.delete_inter, subrow, 3)
            subrow += 1

            approve_lay.addWidget(QLabel("Choose z from:"), subrow, 0)
            self.z_from = QLineEdit("0")
            self.z_from.setValidator(QtGui.QIntValidator(0, frame_num - 1))
            approve_lay.addWidget(self.z_from, subrow, 1)
            approve_lay.addWidget(QLabel("to:"), subrow, 2)
            self.z_to = QLineEdit("32")
            self.z_to.setValidator(QtGui.QIntValidator(0, 35))
            approve_lay.addWidget(self.z_to, subrow, 3)
            subrow += 1

            approve_lay.addWidget(QLabel("Choose x from:"), subrow, 0)
            self.x_from = QLineEdit("0")

            approve_lay.addWidget(self.x_from, subrow, 1)
            approve_lay.addWidget(QLabel("to:"), subrow, 2)
            self.x_to = QLineEdit("0")
            #self.x_to.setValidator(QtGui.QIntValidator(0, 1025))
            approve_lay.addWidget(self.x_to, subrow, 3)
            subrow += 1

            approve_lay.addWidget(QLabel("Choose y from:"), subrow, 0)
            self.y_from = QLineEdit("0")
            #self.y_from.setValidator(QtGui.QIntValidator(0, frame_num - 1))
            approve_lay.addWidget(self.y_from, subrow, 1)
            approve_lay.addWidget(QLabel("to:"), subrow, 2)
            self.y_to = QLineEdit("0")
            #self.y_to.setValidator(QtGui.QIntValidator(0, 1025))
            approve_lay.addWidget(self.y_to, subrow, 3)
            subrow += 1

            save_separate_btn = QPushButton("Save in a separate file")
            save_separate_btn.clicked.connect(self._Preprocess_and_save)
            approve_lay.addWidget(save_separate_btn)

            preproc_tab_grid.addLayout(approve_lay, row, 0)
            row += 1
            self.setLayout(preproc_tab_grid)

        def _Preprocess_and_save(self):
            Z_int = np.zeros(2)
            Z_int[0]=int(self.z_from.text())
            Z_int[1]=int(self.z_to.text())

            X_int = np.zeros(2)
            X_int[0]=int(self.x_from.text())
            X_int[1]=int(self.x_to.text())

            Y_int = np.zeros(2)
            Y_int[0]=int(self.y_from.text())
            Y_int[1]=int(self.y_to.text())

            frame_range = range(int(self.frame_from.text()),int(self.frame_to.text()))
            del_single_fr = self.delete_fr.text().split(',')
            del_single_fr_array = [int(fr) for fr in del_single_fr]
            del_interval_fr = self.delete_inter.text().split(',')
            Tot_del_fr = set()
            for i in range(len(del_interval_fr)):
                interval_str = del_interval_fr[i].split('-')
                interval_array = range(int(interval_str[0]),int(interval_str[1])+1)
                print(interval_array)
                Tot_del_fr = Tot_del_fr.union(set(interval_array))
            Tot_del_fr = Tot_del_fr.union(set(del_single_fr_array))
            Tot_del_fr_final = [int(fr) for fr in Tot_del_fr]
            bg_blur=int(self.bg_blur.text())
            sd_blur=int(self.sd_blur.text())
            bg_subt=int(self.bg_subt.text())
            resized_width = int(self.resized_width.text())
            resized_height = int(self.resized_height.text())
            self.controller.Preprocess_and_save(frame_range,Tot_del_fr,Z_int,X_int,Y_int,bg_blur,sd_blur,bg_subt,resized_width,resized_height)

        def import_file(self):
            FileAddress = self.import_address.text()
            self.controller.import_mask_from_external_file(FileAddress)

        def import_file_green(self):
            FileAddress = self.import_address.text()
            self.controller.import_mask_from_external_file(FileAddress,green=True)


class LabeledSlider(QWidget):
    def __init__(self, minimum, maximum, interval=1, orientation=Qt.Horizontal,
            labels=None, parent=None):
        super(LabeledSlider, self).__init__(parent=parent)

        levels=list(range(minimum, maximum+interval, interval))
        if levels[-1]>maximum:
            levels=levels[:-1]
        #levels[-1]=maximum
        if labels is not None:
            if not isinstance(labels, (tuple, list)):
                raise Exception("<labels> is a list or tuple.")
            if len(labels) != len(levels):
                raise Exception("Size of <labels> doesn't match levels.")
            self.levels=list(zip(levels,labels))
        else:
            self.levels=list(zip(levels,map(str,levels)))

        if orientation==Qt.Horizontal:
            self.layout=QVBoxLayout(self)
        elif orientation==Qt.Vertical:
            self.layout=QHBoxLayout(self)
        else:
            raise Exception("<orientation> wrong.")

        # gives some space to print labels
        self.left_margin=10
        self.top_margin=10
        self.right_margin=10
        self.bottom_margin=10

        self.layout.setContentsMargins(self.left_margin,self.top_margin,
                self.right_margin,self.bottom_margin)

        self.sl=QSlider(orientation, self)
        self.sl.setMinimum(minimum)
        self.sl.setMaximum(maximum)
        self.sl.setValue(minimum)
        if orientation==Qt.Horizontal:
            self.sl.setTickPosition(QSlider.TicksBelow)
            self.sl.setMinimumWidth(300) # just to make it easier to read
        else:
            self.sl.setTickPosition(QSlider.TicksLeft)
            self.sl.setMinimumHeight(300) # just to make it easier to read
        self.sl.setTickInterval(interval)
        self.sl.setSingleStep(1)

        self.layout.addWidget(self.sl)

    def paintEvent(self, e):

        super(LabeledSlider,self).paintEvent(e)

        style=self.sl.style()
        painter=QPainter(self)
        st_slider=QStyleOptionSlider()
        st_slider.initFrom(self.sl)
        st_slider.orientation=self.sl.orientation()

        length=style.pixelMetric(QStyle.PM_SliderLength, st_slider, self.sl)
        available=style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self.sl)

        for v, v_str in self.levels:

            # get the size of the label
            rect=painter.drawText(QRect(), Qt.TextDontPrint, v_str)

            if self.sl.orientation()==Qt.Horizontal:
                # I assume the offset is half the length of slider, therefore
                # + length//2
                x_loc=QStyle.sliderPositionFromValue(self.sl.minimum(),
                        self.sl.maximum(), v, available)+length//2

                # left bound of the text = center - half of text width + L_margin
                left=x_loc-rect.width()//2+self.left_margin
                bottom=self.rect().bottom()

                # enlarge margins if clipping
                if v==self.sl.minimum():
                    if left<=0:
                        self.left_margin=rect.width()//2-x_loc
                    if self.bottom_margin<=rect.height():
                        self.bottom_margin=rect.height()

                    self.layout.setContentsMargins(self.left_margin,
                            self.top_margin, self.right_margin,
                            self.bottom_margin)

                if v==self.sl.maximum() and rect.width()//2>=self.right_margin:
                    self.right_margin=rect.width()//2
                    self.layout.setContentsMargins(self.left_margin,
                            self.top_margin, self.right_margin,
                            self.bottom_margin)

            else:
                y_loc=QStyle.sliderPositionFromValue(self.sl.minimum(),
                        self.sl.maximum(), v, available, upsideDown=True)

                bottom=y_loc+length//2+rect.height()//2+self.top_margin-3
                # there is a 3 px offset that I can't attribute to any metric

                left=self.left_margin-rect.width()
                if left<=0:
                    self.left_margin=rect.width()+2
                    self.layout.setContentsMargins(self.left_margin,
                            self.top_margin, self.right_margin,
                            self.bottom_margin)

            pos=QPoint(left, bottom)
            painter.drawText(pos, v_str)

        return


class TimeSlider(LabeledSlider):
    def __init__(self,controller,T,n_labels):
        self.controller=controller
        self.controller.frame_registered_clients.append(self)
        if n_labels>T:
            n_labels=T
        super(TimeSlider,self).__init__(minimum=0, maximum=T-1, interval=int(T/n_labels))
        self.sl.valueChanged.connect(lambda :self.controller.go_to_frame(self.sl.value()))

    def change_t(self,t):
        """Callback when the controller changes the time, this adjusts the slider display."""
        # blocking the signal is very important because otherwise setValue will emit valueChanged, which will trigger
        # controller.go_to_frame (which calls change_t) and cause an infinite loop
        #t=30#self.gui.time
        if True:#self.sl.value()!=t:
            self.sl.blockSignals(True)
            self.sl.setValue(t)
            self.sl.blockSignals(False)


class GoTo(QWidget):
    """
    This is the field and button to manually enter the frame number and move to it.
    """

    def __init__(self, controller, nb_frames:int):
        """
        :param controller: main controller to report to
        :param nb_frames: number of frames in video
        """
        super().__init__()
        self.controller = controller
        self.grid=QGridLayout()

        self.grid.addWidget(QLabel("Go To Frame: "), 0, 0)

        self.text_field = QLineEdit("0")
        self.text_field.setValidator(QtGui.QIntValidator(0, nb_frames - 1))
        self.grid.addWidget(self.text_field, 0, 1)

        gobut = QPushButton("Go")
        gobut.setStyleSheet("background-color : rgb(93,177,130); border-radius: 4px; min-height: 20px; min-width: 50px")
        gobut.clicked.connect(self.signal_goto)
        self.grid.addWidget(gobut, 0, 2)
        self.setLayout(self.grid)

    def signal_goto(self):
        self.controller.go_to_frame(int(self.text_field.text()))


class DashboardTab(QWidget):
    def __init__(self, controller, dashboard_chunk_size):
        super().__init__()
        self.controller = controller
        self.T = self.controller.frame_num
        self.controller.frame_registered_clients.append(self)
        self.controller.neuron_keys_registered_clients.append(self)
        self.controller.present_neurons_registered_clients.append(self)
        self.controller.present_neurons_all_times_registered_clients.append(self)

        self.scrollarea = QScrollArea()
        self.scrollwidget = QWidget()
        self.chunksize = dashboard_chunk_size
        self.n_cols = 0
        self.chunknumber = 0
        self.current_i = 0   # the line number of the current time t, i.e. t%self.chunksize
        self.grid = QGridLayout()
        self.time_label_buttons = []
        self.button_columns = {}
        for i in range(self.chunksize):
            label_button = QPushButton(str(i) if i < self.T else "")
            label_button.clicked.connect(self._make_button_press_function_t(i))
            label_button.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px;")
            label_button.setFixedWidth(30)
            self.time_label_buttons.append(label_button)
            self.grid.addWidget(label_button, i, 0)

        current_btn = self.current_label_button
        current_btn.setStyleSheet("background-color : rgb(42,99,246); border-radius: 4px;")
        self.scrollwidget.setLayout(self.grid)

        self.scrollarea.setWidget(self.scrollwidget)
        self.scrollarea.setMinimumWidth(
            self.scrollarea.sizeHint().width() + self.scrollarea.verticalScrollBar().sizeHint().width())
        self.scrollarea.horizontalScrollBar().setEnabled(False)
        self.scrollarea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        maingrid = QVBoxLayout()
        topscrollarea = QScrollArea()
        topscrollarea.setFixedHeight(40)
        topscrollwidget = QWidget()
        self.topgrid = QGridLayout()
        button = QPushButton("")
        button.setFixedWidth(30)
        button.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px;")
        self.topgrid.addWidget(button, 0, 0)
        self.keys = {}
        self.assigned_colors = np.array([])   # TODO: consistent with highlighting etc
        self.assigned_colors = np.array(
            [[int(val) for val in col.split(",")] for col in ("41,98,24;235,85,40;249,216,73;117,251,76;117,251,253;0,0,245").split(";")])

        topscrollwidget.setLayout(self.topgrid)
        topscrollarea.setWidget(topscrollwidget)
        topscrollarea.setMinimumWidth(
            topscrollarea.sizeHint().width() + topscrollarea.verticalScrollBar().sizeHint().width())
        topscrollarea.horizontalScrollBar().setEnabled(False)
        topscrollarea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        maingrid.addWidget(topscrollarea)
        maingrid.addWidget(self.scrollarea)
        self.setLayout(maingrid)

    @property
    def current_label_button(self):
        return self.time_label_buttons[self.current_i]

    def _make_button_press_function_t(self, i):
        def button_press_function():
            self.controller.go_to_frame(np.clip(self.chunksize * self.chunknumber + i, 0, self.T-1))

        return button_press_function

    def _make_button_press_function_h(self, idx_from1):
        def button_press_function():
            self.controller.highlight_neuron(idx_from1)

        return button_press_function

    def _make_button_press_function_th(self, i, idx_from1):
        def button_press_function():
            self.controller.go_to_frame(np.clip(self.chunksize * self.chunknumber + i, 0, self.T-1))
            self.controller.highlight_neuron(idx_from1)

        return button_press_function

    def change_t(self, t):
        # unselect previous current button
        current_btn = self.current_label_button
        current_btn.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px;")
        current_btn.setStyle(current_btn.style())   # for some reason this is needed to actually change the color

        # change time
        chunknumber = t // self.chunksize
        if self.chunknumber != chunknumber:
            for i in range(self.chunksize):
                t = self.chunksize * chunknumber + i
                self.time_label_buttons[i].setText(str(t) if t < self.T else "")
            self.chunknumber = chunknumber
        self.current_i = t % self.chunksize

        # select new current button
        current_btn = self.current_label_button
        self.scrollarea.ensureWidgetVisible(current_btn)
        current_btn.setStyleSheet("background-color : rgb(42,99,246); border-radius: 4px;")
        current_btn.setStyle(current_btn.style())   # for some reason this is needed to actually change the color

    def change_neuron_keys(self, key_changes):
        """
        :param key_changes: list of (neuron_idx_from1, key), with key=None for unassigning
        """
        for idx_from1, key in key_changes:
            if key is None:   # remove the button column
                for btn in self.button_columns[idx_from1]:
                    # btn.widget().setParent(None)
                    btn.setParent(None)
                self.n_cols -= 1
            else:   # add a column of buttons
                j = self.n_cols
                col = []
                button = QPushButton(key)
                button.clicked.connect(self._make_button_press_function_h(idx_from1))
                button.setFixedWidth(25)
                button.setStyleSheet("background-color : rgb(" + str(self.assigned_colors[j, 0]) + "," + str(
                    self.assigned_colors[j, 1]) + "," + str(self.assigned_colors[j, 2]) + "); border-radius: 4px;")
                col.append(button)
                self.topgrid.addWidget(button, 0, j + 1)

                # find at which times the neuron is present
                times_present = set(self.controller.times_of_presence(idx_from1))
                for i in range(self.chunksize):
                    callback = self._make_button_press_function_th(i, idx_from1)
                    button = DashboardItem(idx_from1, callback)

                    t = self.chunknumber * self.chunksize + i
                    if t in times_present:
                        button.set_present()
                    else:
                        button.set_absent()

                    col.append(button)
                    self.grid.addWidget(button, i, j + 1)
                self.button_columns[idx_from1] = col
                self.n_cols += 1

    def change_present_neurons(self, present=None, added=None, removed=None):
        """
        Changes which of the neurons are present in current frame, as their corresponding buttons should be colored
        in blue instead of red.
        :param present: which neuron indices (from 1) are present, if given
        :param added: single neuron index (from 1) that was added, if given
        :param removed: single neuron index (from 1) that was removed, if given
        """
        if present is not None:
            for i_from1, col in self.button_columns.items():
                if i_from1 in present:
                    col[self.current_i+1].set_present()
                else:
                    col[self.current_i+1].set_absent()
            return
        if added is not None and added in self.button_columns:
            self.button_columns[added][self.current_i+1].set_present()
        if removed is not None and removed in self.button_columns:
            self.button_columns[removed][self.current_i+1].set_absent()

    def change_present_neurons_all_times(self):
        pass   # TODO



class TrackTab(QWidget):
    def __init__(self,gui):
        super().__init__()
        #CFP-HELP: Probably, this should be the controller? 
        self.gui=gui

        self.methods=tracking_methods.methods
        self.grid=QGridLayout()
        
        row=0
        self.grid.addWidget(QLabel("Select Method"),row,0)
        row+=1

        self.combobox=QComboBox()
        self.combobox.addItem("")
        for key in self.methods.keys():
            self.combobox.addItem(key)
        self.combobox.setCurrentIndex(0)
        self.combobox.currentIndexChanged.connect(lambda x: self.run_button.setEnabled(False) if x==0 else self.run_button.setEnabled(True))
        self.grid.addWidget(self.combobox,row,0)
        row+=1

        self.param_edit=QLineEdit()
        self.grid.addWidget(self.param_edit,row,0)
        row+=1

        self.run_button=QPushButton("Run")
        self.run_button.setStyleSheet("background-color : rgb(93,177,130); border-radius: 4px; min-height: 20px")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.make_run_function())
        self.grid.addWidget(self.run_button,row,0)
        row+=1

        self.setLayout(self.grid)

    def make_run_function(self):
        def run_function():
            name=str(self.combobox.currentText())
            params=self.param_edit.text()
            self.run(name,params)
        return run_function

    def run(self,method_name,params):
        msgbox=QMessageBox()
        msgbox.setText("Confirm Run")
        msgbox.setInformativeText("Run "+method_name+" with "+params+"?")
        msgbox.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        res=msgbox.exec()
        if res==QMessageBox.No:
            return
            
        #CFP-HELP: I want to save all changes and close the dataset completely, as well as freeze the gui until the script runs, can you do this?
        #self.gui.respond("save")
        #self.gui.respond("timer_stop")
        #self.gui.dataset.close()
        #
        print("CFP:save")
        print("CFP:timer_stop")
        print("CFP:close")

        progress=QProgressDialog("","cancel",-1,101)
        labeltext="Running "+method_name+((" with "+params) if params!="" else "") +":\n"
        progress.setLabelText(labeltext)
        progress.showNormal()
        progress.resize(500,100)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(-1)
        QApplication.processEvents()

        command_pipe_main,command_pipe_sub=Pipe()
        process = Process(target=self.methods[method_name], args=(command_pipe_sub,self.gui.dataset.file_path,params))
        process.start()
        command_pipe_main.send("run")
        while True:
            if progress.wasCanceled():
                command_pipe_main.send("cancel")
                break
            command_pipe_main.send("report")
            res=command_pipe_main.recv()
            if res=="Done":
                command_pipe_main.send("close")
                break
            else:
                if type(res)==list:
                    stepname,progress_value=res
                    progress.setLabelText(labeltext+stepname)
                    progress.setValue(progress_value)
            QApplication.processEvents()
        process.join()

        progress.setValue(101)
        
        #CFP-HELP: Now that the script ended, I want to reopen the dataset and link it to the GUI, is this possible?
        print("CFP:open")
        #CFP-HELP:Finally, the GUI should be alerted that the NN points have changed(have new results)
        print("CFP:renew_helpers")
        print("CFP:timer_start")
        #self.gui.dataset.open()
        #self.gui.respond("renew_helpers")
        #self.gui.respond("timer_start")
        
        
        
        