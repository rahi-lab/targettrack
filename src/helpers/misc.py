import numpy as np


class ColorAssignment:
    """
    This class assigns a new color to each neuron that is newly assigned to a key.
    """
    def __init__(self, controller):
        self.controller = controller
        self.color_list = [[int(val) for val in col.split(",")]
                           for col in self.controller.settings["keys_colors"].split(";")]
        self.no_color = [255, 255, 255]
        self.neuron_color_index = {}   # i = self.neuron_color_index[neuron_id_from1]; color is self.color_list[i]
        self.color_count = np.zeros(len(self.color_list))   # this contains the number of times each color is used
        self.controller.neuron_keys_registered_clients.append(self)

    def color_for_neuron(self, idx_from1):
        if idx_from1 in self.neuron_color_index:
            return self.color_list[self.neuron_color_index[idx_from1]]
        else:
            return self.no_color

    def change_neuron_keys(self, key_changes):
        """
        :param key_changes: list of (neuron_idx_from1, key), with key=None for unassigning
        """
        for neuron_idx_from1, key in key_changes:
            if key is None:
                if neuron_idx_from1 in self.neuron_color_index:
                    col_idx = self.neuron_color_index[neuron_idx_from1]
                    del self.neuron_color_index[neuron_idx_from1]
                    self.color_count[col_idx] -= 1
            else:
                col_idx = np.argmin(self.color_count)
                self.neuron_color_index[neuron_idx_from1] = col_idx
                self.color_count[col_idx] += 1
        # todo: could also check if max(self.color_count) - min(self.color_count) >= 2 then reassign colors to not have
        #  duplicate colors when other colors are unused
