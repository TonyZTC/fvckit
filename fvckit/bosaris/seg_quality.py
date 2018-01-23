# -*- coding: utf-8 -*-

# This package is a translation of a part of the BOSARIS toolkit.
# The authors thank Niko Brummer and Agnitio for allowing them to
# translate this code and provide the community with efficient structures
# and tools.
#
# The BOSARIS Toolkit is a collection of functions and classes in Matlab
# that can be used to calibrate, fuse and plot scores from speaker recognition
# (or other fields in which scores are used to test the hypothesis that two
# samples are from the same source) trials involving a model and a test segment.
# The toolkit was written at the BOSARIS2010 workshop which took place at the
# University of Technology in Brno, Czech Republic from 5 July to 6 August 2010.
# See the User Guide (available on the toolkit website)1 for a discussion of the
# theory behind the toolkit and descriptions of some of the algorithms used.
#
# The BOSARIS toolkit in MATLAB can be downloaded from `the website
# <https://sites.google.com/site/bosaristoolkit/>`_.

"""
This is the 'seg_quality' module
"""
import numpy
import sys
import h5py
import logging
import copy
from fvckit import STAT_TYPE
from fvckit.bosaris.idmap import IdMap
from fvckit.fvckit_wrappers import check_path_existance

__author__ = "Andreas Nautsch, Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__credits__ = ["Niko Brummer", "Edward de Villiers"]


def diff(list1, list2):
    c = [item for item in list1 if item not in list2]
    c.sort()
    return c


def ismember(list1, list2):
    c = [item in list2 for item in list1]
    return c


class SegQuality:
    """A class for representing segment quality information.

        :attr ids: a list of segments in a ndarray
        :attr values: 1D ndarray of float which represent quality values
    """
    def __init__(self, seq_quality_file_name='', ids=None, values=None):
        """Initialize SegQuality object
        :param seq_quality_file_name:
        :param ids:
        :param values:
        """
        self.ids = numpy.empty(0, dtype="|O")
        self.values = numpy.array([], dtype=STAT_TYPE)

        if seq_quality_file_name == '':
            self.ids = copy.deepcopy(ids)
            self.values = copy.deepcopy(values)
        else:
            tmp = self.read(seq_quality_file_name)
            self.ids = tmp.ids
            self.values = tmp.values
        assert(self.validate())

    @check_path_existance
    def write(self, output_file_name):
        assert self.validate(), "Error: wrong SegQuality format"

        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("ids", data=self.ids.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("values", data=self.values.astype(STAT_TYPE),
                         maxshape=(None, None),
                         compression="gzip",
                         fletcher32=True)

    @check_path_existance
    def write_txt(self, output_file_name):
        """Save a SegQuality object to a text file.

        :param output_file_name: name of the output text file
        """
        qual_line_format = '{}' + ''.join([' {}'] * self.values.shape[1]) + '\n'
        fid = open(output_file_name, 'w')
        fid.write('{}\n'.format(self.values.shape[0]))
        for idx, id in enumerate(self.ids):
            fid.write(qual_line_format.format(id, self.values[idx]))
        fid.close()

    def align_with_ids(self, idmap):
        assert self.validate(), "Error: wrong SegQuality format"
        assert(isinstance(idmap, IdMap))
        assert idmap.validate(), "Error: wrong SegQuality format"

        # ids by samples, i.e. idmap.rightids
        unique_refids = numpy.unique(idmap.rightids)

        in_num_ids = self.ids.shape[0]
        num_qual = self.values.shape[1]
        num_ids = unique_refids.shape[0]

        aligned_qual = SegQuality()
        aligned_qual.ids = unique_refids

        ridx = numpy.array(ismember(self.ids, unique_refids))

        aligned_qual.values = numpy.zeros(num_qual, num_ids)
        aligned_qual.values = self.values[:, ridx]

        lost = in_num_ids - num_ids
        if lost > 0:
            logging.warning('Number of segments reduced from {} to {}.'.format(in_num_ids, num_ids))

        if aligned_qual.ids.shape[0] < num_ids:
            logging.warning('{} of {} ids don''t have quality values'.format(
                num_ids - aligned_qual.ids.shape[0], num_ids))

        assert(aligned_qual.validate())

    def validate(self):
        assert(numpy.isfinite(self.values.all()))

        ok = isinstance(self.ids, numpy.ndarray)
        ok &= isinstance(self.values, numpy.ndarray)
        ok &= self.ids.ndim == 1
        ok &= self.values.ndim == 2
        ok &= self.ids.shape[0] == self.values.shape[1]

        return ok

    @staticmethod
    def read(input_file_fame):
        """Reads a SegQuality object from an hdf5 file.

        :param input_file_fame: name of the file to read from
        """
        with h5py.File(input_file_fame, "r") as f:
            qual = SegQuality()
            qual.ids = f.get("ids").value

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                qual.ids = qual.ids.astype('U100', copy=False)

            qual.values = f.get("values").value

            assert qual.validate(), "Error: wrong SegQuality format"
            return qual

    @staticmethod
    def read_txt(input_file_name):
        """Creates a SegQuality object from information stored in a text file.

            :param input_file_name: name of the file to read from
        """
        qual = SegQuality()
        ids = []
        qual_values = numpy.array([], dtype=STAT_TYPE)

        with open(input_file_name) as fid:
            num_qual = int(fid.readline(1))
            for line in fid:
                tmp = line.split()
                ids.append(tmp[0])
                qual_values = numpy.append(qual_values, tmp[1:])

            qual.ids = numpy.array(ids, dtype='object')
            qual.values = qual_values.transpose()

            assert qual.validate(), "Error: wrong SegQuality format"
            return qual

