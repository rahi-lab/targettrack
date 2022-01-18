from nd2reader import ND2Reader
# import .file_loading.tiff_loader as tif
import csv

import logging
import os


class WormReader:
    """
    General class for reading in worm data
    works for tif and nd2 format videos
    
    Usage reades = WormReader(filename)
          reader.get_3d_img(c = c, t = t)

    TODO make this work more like a pims data reader for ease of use More like the nd2reader package
    """

    # Constants for our wormfiles
    RED = 1
    GREEN = 0

    def __init__(self, filename):
        """
        Create reader for wormvids of format tif or nd2
        """

        self.logger = logging.getLogger("WormReader")

        self.movie_file = filename
        
        # Prepare file readers for the original images
        if os.path.splitext(self.movie_file)[1] == '.nd2':
            self.reader = ND2Reader(self.movie_file)
            self.prep_nd2_reader(c=self.RED)
            self.frames = self.nd2_measure_frames()

        # elif os.path.splitext(self.movie_file)[1] == '.tif':
        #     self.logger.info("Loading tif file {}".format(self.movie_file))
        #     self.reader = None
        #     tif.prep_reader(self.movie_file)
        #     self.frames = tif.reader.frames

        self.frame_shape = self.get_3d_img().shape

    def nd2_measure_frames(self, reader=None):
        """
        work around for nd2Reader bug. the reader doesn't get the correct time length for cut videos (metadata reading error)
        So this function simply loops through time frames until error to get the correct time length
        forgive me god for this function

        :param reader: ND2Reader attached to the file of interest. If not given (normal use case), self.reader is used.
        :returns: time frame list (as any iterable)
        """

        if reader is None:
            reader = self.reader
        else:
            self.logger.warning("Using given reader instead of self.reader.")

        max_n_frames = len(self.reader.metadata["frames"])

        mi = 0
        ma = max_n_frames
        prev_t = -1   # anything that is not the starting point
        while True:
            t = mi + (ma - mi) // 2
            try:
                reader[t]
                mi = t
            except:
                ma = t
            if prev_t == t:
                break
            prev_t = t
        return range(0, t+1)

    def get_3d_img(self, c=0, t=0):
        """
        Loads 3d image as numpy array from the file at position x,y,z.
        Just use as get_3d_img(c, t)
        """
        t = int(t)
        if isinstance(c, str):
            if c.lower() == "red":
                c = WormReader.RED
            elif c.lower() == "green":
                c = WormReader.GREEN
            else:
                raise ValueError("Color c must be either 0 or 1, or 'green' or 'red', respectively")
        else:
            c = int(c)
            if not c in [0,1]:
                raise ValueError("Color c must be either 0 or 1, or 'green' or 'red', respectively")

        if os.path.splitext(self.movie_file)[1] == ".nd2":
            self.prep_nd2_reader(c=c)   # TODO: not necessary to do this every time??
            img3d = self.reader[t]

        # elif os.path.splitext(self.movie_file)[1] == ".tif":
        #     img3d = tif.read_frame(self.movie_file, c, t)

        else:
            raise ValueError("Can only read .nd2 and .tif files")

        return img3d

    def prep_nd2_reader(self, c=RED):
        """
        Prepare the parameters of the ND2Reader object to be consistent with our data
        bundled_axes "yxz"
        default_cord for the channel of interest
        So looping over the reader goes stright through time
        """

        self.reader.default_coords['c'] = c
        self.reader.bundle_axes         = 'yxz'
        self.reader.iter_axes           = 't'

    def get_position(self, output_file = None):

        raw_metadata = self.reader._parser._raw_metadata

        x_data = raw_metadata.x_data.tolist()
        y_data = raw_metadata.y_data.tolist()
        z_data = raw_metadata.z_data.tolist()
        timesteps = self.reader.get_timesteps()

        if output_file is None:
            output_file = self.movie_file.split('/')[1]
            output_file = output_file.replace('.nd2', '_position.csv')
            output_file = 'results_real/'+ output_file

        self.logger.info("Output file is written at " + output_file )

        with open(output_file, 'w', newline='') as file:
            writer = csv.writer( file)
            for index in range(len(x_data)):
                row = [timesteps[index],x_data[index],y_data[index],z_data[index]]
                writer.writerow(row)

    def __del__(self):
        if self.reader is not None:
            self.reader.close()


if __name__ == "__main__":
    path = "worm_videos/test_t8.nd2"
    # path = "/Users/ariane/Desktop/EPFL/LPBS/debug_fm/20200327_144323_SJR5.3_wo_AIY_w1_s3_for_test.nd2"
    wormreader = WormReader(path)