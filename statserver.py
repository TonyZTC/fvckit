# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.


"""
Copyright 2014-2016 Anthony Larcher

:mod:`statserver` provides methods to manage zero and first statistics.
"""
import h5py
import numpy
import os
import sys
import ctypes
import warnings
import copy
import pickle
import gzip
import logging
import scipy
import multiprocessing

from sidekit.bosaris import IdMap
from sidekit.mixture import Mixture
from sidekit.features_server import FeaturesServer
from sidekit.sidekit_wrappers import process_parallel_lists, deprecated, check_path_existance
import sidekit.frontend
from sidekit import STAT_TYPE


ct = ctypes.c_double
if STAT_TYPE == numpy.float32:
    ct = ctypes.c_float


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def compute_llk(stat, V, Sigma, U=None, D=None):
    # Compute Likelihood
    (N, d) = stat.stat1.shape
    centered_data = stat.stat1 - stat.get_mean_stat1()
    
    if Sigma.ndim == 2:
        Sigma_tot = numpy.dot(V, V.T) + Sigma
    else:
        Sigma_tot = numpy.dot(V, V.T) + numpy.diag(Sigma)
    if U is not None:
        Sigma_tot += numpy.dot(U, U.T)
    
    E, junk = scipy.linalg.eigh(Sigma_tot)
    log_det = numpy.sum(numpy.log(E))
    
    return (-0.5 * (N * d * numpy.log(2 * numpy.pi) + N * log_det +
                    numpy.sum(numpy.sum(numpy.dot(centered_data, scipy.linalg.inv(Sigma_tot)) * centered_data, axis=1))))


def sum_log_probabilities(lp):
    """Sum log probabilities in a secure manner to avoid extreme values

    :param lp: ndarray of log-probabilities to sum
    """
    ppMax = numpy.max(lp, axis=1)
    loglk = ppMax \
        + numpy.log(numpy.sum(numpy.exp((lp.transpose() - ppMax).transpose()), axis=1))
    ind = ~numpy.isfinite(ppMax)
    if sum(ind) != 0:
        loglk[ind] = ppMax[ind]
    pp = numpy.exp((lp.transpose() - loglk).transpose())
    return pp, loglk


@process_parallel_lists
def fa_model_loop(batch_start, mini_batch_indices, r, Phi_white, Phi, Sigma, stat0, stat1, E_h, E_hh, numThread):
    """
    :param batch_start: index to start at in the list
    :param mini_batch_indices: indices of the elements in the list (should start at zero)
    :param r: rank of the matrix
    :param Phi_white: whitened version of the factor matrix
    :param Phi: non-whitened version of the factor matrix
    :param Sigma: covariance matrix
    :param stat0: matrix of zero order statistics
    :param stat1: matrix of first order statistics
    :param E_h: accumulator
    :param E_hh: accumulator
    :param numThread: number of parallel process to run
    """
    if Sigma.ndim == 2:
        A = Phi.T.dot(scipy.linalg.solve(Sigma,Phi)).astype(dtype=STAT_TYPE)
    
    tmp = numpy.zeros((Phi.shape[1], Phi.shape[1]), dtype=STAT_TYPE)

    for idx in mini_batch_indices:
        if Sigma.ndim == 1:
            invLambda = scipy.linalg.inv(numpy.eye(r) + (Phi_white.T * stat0[idx + batch_start, :]).dot(Phi_white))
        else: 
            invLambda = scipy.linalg.inv(stat0[idx + batch_start, 0] * A + numpy.eye(A.shape[0]))

        Aux = Phi_white.T.dot(stat1[idx + batch_start, :])
        numpy.dot(Aux, invLambda, out=E_h[idx])
        E_hh[idx] = invLambda + numpy.outer(E_h[idx], E_h[idx], tmp)
   

@process_parallel_lists
def fa_distribution_loop(distrib_indices, _A, stat0, batch_start, batch_stop, E_hh, numThread):
    """
    :param distrib_indices: indices of the distributions to iterate on
    :param _A: accumulator
    :param stat0: matrix of zero order statistics
    :param batch_start: index of the first session to process
    :param batch_stop: index of the last session to process
    :param E_hh: accumulator
    :param numThread: number of parallel process to run
    """
    tmp = numpy.zeros((E_hh.shape[1], E_hh.shape[1]), dtype=STAT_TYPE)
    for c in distrib_indices:
        _A[c] += numpy.einsum('ijk,i->jk', E_hh, stat0[batch_start:batch_stop, c], out=tmp)
        # The line abov is equivalent to the two lines below:
        #tmp = (E_hh.T * stat0[batch_start:batch_stop, c]).T
        #_A[c] += numpy.sum(tmp, axis=0)

def load_existing_statistics_hdf5(statserver, statserverFileName):
    """Load required statistics into the StatServer by reading from a file
        in hdf5 format.

    :param statserver: sidekit.StatServer to fill
    :param statserverFileName: name of the file to read from
    """
    assert os.path.isfile(statserverFileName), "statserverFileName does not exist"

    # Load the StatServer
    ss = StatServer(statserverFileName)

    # Check dimension consistency with current Stat_Server
    ok = True
    if statserver.stat0.shape[0] > 0:
        ok &= (ss.stat0.shape[0] == statserver.stat0.shape[1])
        ok &= (ss.stat1.shape[0] == statserver.stat1.shape[1])
    else:
        statserver.stat0 = numpy.zeros((statserver. modelset.shape[0], ss.stat0.shape[1]), dtype=STAT_TYPE)
        statserver.stat1 = numpy.zeros((statserver. modelset.shape[0], ss.stat1.shape[1]), dtype=STAT_TYPE)

    if ok:
        # For each segment, load statistics if they exist
        # Get the lists of existing segments
        segIdx = [i for i in range(statserver. segset.shape[0]) if statserver.segset[i] in ss.segset]
        statIdx = [numpy.where(ss.segset == seg)[0][0] for seg in statserver.segset if seg in ss.segset]

        # Copy statistics
        statserver.stat0[segIdx, :] = ss.stat0[statIdx, :]
        statserver.stat1[segIdx, :] = ss.stat1[statIdx, :]
    else:
        raise Exception('Mismatched statistic dimensions')


class StatServer:
    """A class for statistic storage and processing

    :attr modelset: list of model IDs for each session as an array of strings
    :attr segset: the list of session IDs as an array of strings
    :attr start: index of the first frame of the segment
    :attr stop: index of the last frame of the segment
    :attr stat0: a ndarray of float64. Each line contains 0-order statistics 
        from the corresponding session
    :attr stat1: a ndarray of float64. Each line contains 1-order statistics 
        from the corresponding session
    
    """

    def __init__(self, statserverFileName='', ubm=None, statserverFileFormat='hdf5'):
        """Initialize an empty StatServer or load a StatServer from an existing
        file.

        :param statserverFileName: name of the file to read from. If filename 
                is an empty string, the StatServer is initialized empty. 
                If filename is an IdMap object, the StatServer is initialized 
                to match the structure of the IdMap.
        :param statServerFileFormat: format of the file(s) to read from. 
            Can be:
                - pickle 
                - hdf5 (default)
        """
        assert ((statserverFileFormat.lower() == 'pickle') |
                (statserverFileFormat.lower() == 'hdf5') |
                (statserverFileFormat.lower() == 'alize')),\
            'statserverFileFormat must be pickle or hdf5'

        self.modelset = numpy.empty(0, dtype="|O")
        self.segset = numpy.empty(0, dtype="|O")
        self.start = numpy.empty(0, dtype="|O")
        self.stop = numpy.empty(0, dtype="|O")
        self.stat0 = numpy.array([], dtype=STAT_TYPE)
        self.stat1 = numpy.array([], dtype=STAT_TYPE)

        if statserverFileName == '':
            pass
        # initialize
        elif isinstance(statserverFileName, IdMap):
            self.modelset = statserverFileName.leftids
            self.segset = statserverFileName.rightids
            self.start = statserverFileName.start
            self.stop = statserverFileName.stop

            if ubm is not None:            
                # Initialize stat0 and stat1 given the size of the UBM
                self.stat0 = numpy.zeros((self. segset.shape[0], ubm.distrib_nb()), dtype=STAT_TYPE)
                self.stat1 = numpy.zeros((self. segset.shape[0], ubm.sv_size()), dtype=STAT_TYPE)
            
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning) 
                    tmp_stat0 = multiprocessing.Array(ct, self.stat0.size)
                    self.stat0 = numpy.ctypeslib.as_array(tmp_stat0.get_obj())
                    self.stat0 = self.stat0.reshape(self.segset.shape[0], ubm.distrib_nb())
            
                    tmp_stat1 = multiprocessing.Array(ct, self.stat1.size)
                    self.stat1 = numpy.ctypeslib.as_array(tmp_stat1.get_obj())
                    self.stat1 = self.stat1.reshape(self.segset.shape[0], ubm.sv_size())

        # initialize by reading an existing StatServer
        elif statserverFileFormat == 'pickle':
            self.read_pickle(statserverFileName)
        elif statserverFileFormat == 'hdf5':
            self.read_hdf5(statserverFileName)

    def validate(self, warn=False):
        """Validate the structure and content of the StatServer. 
        Check consistency between the different attributes of 
        the StatServer:
        - dimension of the modelset
        - dimension of the segset
        - length of the modelset and segset
        - consistency of stat0 and stat1
        
        :param warn: bollean optional, if True, display possible warning
        """
        ok = self.modelset.ndim == 1 \
            and (self.modelset.shape == self.segset.shape == self.start.shape == self.stop.shape) \
            and (self.stat0.shape[0] == self.stat1.shape[0] == self.modelset.shape[0]) \
            and (not bool(self.stat1.shape[1] % self.stat0.shape[1]))

        if warn and (self.segset.shape != numpy.unique(self.segset).shape):
                logging.warning('Duplicated segments in StatServer')
        return ok

    def merge(*arg):
        """
        Merge a variable number of StatServers into one.
        If a pair segmentID is duplicated, keep ony one
        of them and raises a WARNING
        """
        line_number = 0
        for idx, ss in enumerate(arg):
            assert(isinstance(ss, sidekit.StatServer) and ss.validate()), "Arguments must be proper StatServers"
            
            # Check consistency of StatServers (dimension of the stat0 and stat1)
            if idx == 0:
                dim_stat0 = ss.stat0.shape[1]
                dim_stat1 = ss.stat1.shape[1]            
            else:
                assert(dim_stat0 == ss.stat0.shape[1] and 
                       dim_stat1 == ss.stat1.shape[1]), "Stat dimensions are not consistent"
    
            line_number += ss.modelset.shape[0]
    
        # Get a list of unique modelID-segmentID    
        ID_list = []
        for ss in arg:
            ID_list += list(ss.segset)
        ID_set = set(ID_list)
        if line_number != len(ID_set):
            print("WARNING: duplicated segmentID in input StatServers")
        
        # Initialize the new StatServer with unique set of segmentID
        new_stat_server = sidekit.StatServer()
        new_stat_server.modelset = numpy.empty(len(ID_set), dtype='object')
        new_stat_server.segset = numpy.array(list(ID_set))
        new_stat_server.start = numpy.empty(len(ID_set), 'object')
        new_stat_server.stop = numpy.empty(len(ID_set), dtype='object')
        new_stat_server.stat0 = numpy.zeros((len(ID_set), dim_stat0), dtype=STAT_TYPE)
        new_stat_server.stat1 = numpy.zeros((len(ID_set), dim_stat1), dtype=STAT_TYPE)
        
        for ss in arg:
            for idx, segment in enumerate(ss.segset):
                new_idx = numpy.argwhere(new_stat_server.segset == segment)
                new_stat_server.modelset[new_idx] = ss.modelset[idx]
                new_stat_server.start[new_idx] = ss.start[idx]
                new_stat_server.stop[new_idx] = ss.stop[idx]
                new_stat_server.stat0[new_idx, :] = ss.stat0[idx, :]
                new_stat_server.stat1[new_idx, :] = ss.stat1[idx, :]
                
        assert(new_stat_server.validate()), "Problem in StatServer Merging"
        return new_stat_server
    
    def read(self, inputFileName):
        """Read information from a file and constructs a StatServer object. The
        type of file is deduced from the extension. The extension must be
        '.hdf5' or '.h5' for a HDF5 file and '.p' for pickle.
        In order to use different extension, use specific functions.

        :param inputFileName: name of the file o read from
        """
        extension = os.path.splitext(inputFileName)[1][1:].lower()
        if extension == 'p':
            self.read_pickle(inputFileName)
        elif extension in ['hdf5', 'h5']:
            self.read_hdf5(inputFileName)
        else:
            raise Exception('Error: unknown extension')

    def read_hdf5(self, statserverFileName, prefix=''):
        """Read StatServer in hdf5 format
        
        :param statserverFileName: name of the file to read from
        """
        with h5py.File(statserverFileName, "r") as f:
            self.modelset = f.get(prefix+"modelset").value
            self.segset = f.get(prefix+"segset").value

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                self.modelset = self.modelset.astype('U', copy=False)
                self.segset = self.segset.astype('U', copy=False)

            tmpstart = f.get(prefix+"start").value
            tmpstop = f.get(prefix+"stop").value
            self.start = numpy.empty(f[prefix+"start"].shape, '|O')
            self.stop = numpy.empty(f[prefix+"stop"].shape, '|O')
            self.start[tmpstart != -1] = tmpstart[tmpstart != -1]
            self.stop[tmpstop != -1] = tmpstop[tmpstop != -1]

            self.stat0 = f.get(prefix+"stat0").value.astype(dtype=STAT_TYPE)
            self.stat1 = f.get(prefix+"stat1").value.astype(dtype=STAT_TYPE)

            assert self.validate(), "Error: wrong StatServer format"

    def read_pickle(self, inputFileName):
        """Read StatServer in PICKLE format.
        
        :param inputFileName: name of the file to read from
        """
        with gzip.open(inputFileName, "rb") as f:
            ss = pickle.load(f)
            self.modelset = ss.modelset
            self.segset = ss.segset
            self.stat0 = ss.stat0.astype(dtype=STAT_TYPE)
            self.stat1 = ss.stat1.astype(dtype=STAT_TYPE)
            self.start = ss.start
            self.stop = ss.stop

    @deprecated
    def save(self, outputFileName):
        self.write(outputFileName)

    @check_path_existance
    def write(self, outputFileName):
        """Save the StatServer object to file. The format of the file 
        to create is set accordingly to the extension of the filename.
        This extension can be '.p' for pickle format or '.hdf5' and '.h5' 
        for HDF5 format.

        :param outputFileName: name of the file to write to
        """
        extension = os.path.splitext(outputFileName)[1][1:].lower()
        if extension == 'p':
            self.write_pickle(outputFileName)
        elif extension in ['hdf5', 'h5']:
            self.write_hdf5(outputFileName)
        elif extension == 'txt':
            self.write_txt(outputFileName)
        else:
            raise Exception('Wrong output format, must be pickle or hdf5')

    @deprecated
    def save_hdf5(self, outpuFileName, prefix= ''):
        self.write_hdf5(outpuFileName, prefix)

    @check_path_existance
    def write_hdf5(self, outpuFileName, prefix= ''):
        """Write the StatServer to disk in hdf5 format.
        
        :param outpuFileName: name of the file to write in.
        """
        assert self.validate(), "Error: wrong StatServer format"
        with h5py.File(outpuFileName, "w") as f:

            f.create_dataset(prefix+"modelset", data=self.modelset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset(prefix+"segset", data=self.segset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset(prefix+"stat0", data=self.stat0,
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset(prefix+"stat1", data=self.stat1,
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)

            start = copy.deepcopy(self.start)
            start[numpy.isnan(self.start.astype('float'))] = -1
            start = start.astype('int8', copy=False)

            stop = copy.deepcopy(self.stop)
            stop[numpy.isnan(self.stop.astype('float'))] = -1
            stop = stop.astype( 'int8', copy=False)

            f.create_dataset(prefix+"start", data=start,
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset(prefix+"stop", data=stop,
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)

    @deprecated
    def save_pickle(self, outputFileName):
        self. write_pickle(outputFileName)

    @check_path_existance
    def write_pickle(self, outputFileName):
        """Save StatServer in PICKLE format.
        In Python > 3.3, statistics are converted into float32 to save space
        
        :param outputFileName: name of the file to write to
        """
        with gzip.open(outputFileName, "wb") as f:
            self.stat0.astype('float32', copy=False)
            self.stat1.astype('float32', copy=False)
            pickle.dump(self, f)

    def load_existing_statistics_pickle(self, statserverFileName):
        """Load required statistics into the StatServer by reading from a file
            in pickle format.
        
        :param statserverFileName: name of the file to read from
        """
        assert os.path.isfile(statserverFileName), "statserverFileName does not exist"
        
        # Load the StatServer
        ss = StatServer(statserverFileName)

        # Check dimension consistency with current Stat_Server
        if self.stat0.shape[0] > 0:
            ok = ss.stat0.shape[0] == self.stat0.shape[1] and \
                ss.stat1.shape[0] == self.stat1.shape[1] and \
                ss.start.shape[0] == self.start.shape[1] and \
                ss.stop.shape[0] == self.stop.shape[1]
                
        else:
            self.stat0 = numpy.zeros((self.modelset.shape[0], ss.stat0.shape[1]))
            self.stat1 = numpy.zeros((self.modelset.shape[0], ss.stat1.shape[1]))
            self.start = numpy.empty(self.modelset.shape[0], '|O')
            self.stop = numpy.empty(self.modelset.shape[0], '|O')

        if ok:
            # For each segment, load statistics if they exist
            # Get the lists of existing segments
            segIdx = [i for i in range(self. segset.shape[0]) if self.segset[i] in ss.segset]
            statIdx = [numpy.where(ss.segset == seg)[0][0] for seg in self.segset if seg in ss.segset]

            # Copy statistics
            self.stat0[segIdx, :] = ss.stat0[statIdx, :]
            self.stat1[segIdx, :] = ss.stat1[statIdx, :]
            self.start[segIdx] = ss.start[statIdx]
            self.stop[segIdx] = ss.stop[statIdx]
        else:
            raise Exception('Mismatched statistic dimensions in StatServer')

    def get_model_stat0(self, modID):
        """Return zero-order statistics of a given model
        
        :param modID: ID of the model which stat0 will be returned
          
        :return: a matrix of zero-order statistics as a ndarray
        """
        S = self.stat0[self. modelset == modID, ]
        return S

    def get_model_stat1(self, modID):
        """Return first-order statistics of a given model
        
        :param modID: string, ID of the model which stat1 will be returned
          
        :return: a matrix of first-order statistics as a ndarray
        """
        S = self.stat1[self.modelset == modID, ]
        return S

    def get_model_stat0_by_index(self, modIDX):
        """Return zero-order statistics of model number modIDX
        
        :param modIDX: integer, index of the unique model which stat0 will be
            returned
        
        :return: a matrix of zero-order statistics as a ndarray
        """
        selectSeg = (self.modelset == numpy.unique(self.modelset)[modIDX])
        S = self.stat0[selectSeg, ]
        return S

    def get_model_stat1_by_index(self, modIDX):
        """Return first-order statistics of model number modIDX
        
        :param modIDX: integer, index of the unique model which stat1 will be
              returned
        
        :return: a matrix of first-order statistics as a ndarray
        """
        selectSeg = (self.modelset == numpy.unique(self.modelset)[modIDX])
        S = self.stat1[selectSeg, ]
        return S

    def get_segment_stat0(self, segID):
        """Return zero-order statistics of segment which ID is segID
        
        :param segID: string, ID of the segment which stat0 will be 
              returned
        
        :return: a matrix of zero-order statistics as a ndarray
        """
        S = self.stat0[self.segset == segID, ]
        return S

    def get_segment_stat1(self, segID):
        """Return first-order statistics of segment which ID is segID
        
        :param segID: string, ID of the segment which stat1 will be 
              returned
        
        :return: a matrix of first-order statistics as a ndarray
        """
        S = self.stat1[self.segset == segID, ]
        return S

    def get_segment_stat0_by_index(self, segIDX):
        """Return zero-order statistics of segment number segIDX
        
        :param segIDX: integer, index of the unique segment which stat0 will be 
              returned
        
        :return: a matrix of zero-order statistics as a ndarray
        """
        return self.stat0[segIDX, ]

    def get_segment_stat1_by_index(self, segIDX):
        """Return first-order statistics of segment number segIDX
        
        :param segIDX: integer, index of the unique segment which stat1 will be 
              returned
        
        :return: a matrix of first-order statistics as a ndarray
        """
        return self.stat1[segIDX, ]

    def get_model_segments(self, modID):
        """Return the list of segments belonging to model modID
        
        :param modID: string, ID of the model which belonging segments will be 
              returned
        
        :return: a list of segments belonging to the model
        """
        S = self.segset[self.modelset == modID, ]
        return S

    def get_model_segments_by_index(self, modIDX):
        """Return the list of segments belonging to model number modIDX
        
        :param modIDX: index of the model which list of segments will be 
            returned
        
        :return: a list of segments belonging to the model
        """
        selectSeg = (self.modelset == numpy.unique(self.modelset)[modIDX])
        S = self.segset[selectSeg, ]
        return S

    def align_segments(self, segmentList):
        """Align segments of the current StatServer to match a list of segment 
            provided as input parameter. The size of the StatServer might be 
            reduced to match the input list of segments.
        
        :param segmentList: ndarray of strings, list of segments to match
        """
        indx = numpy.array([numpy.argwhere(self.segset == v)[0][0] for v in segmentList])
        self.segset = self.segset[indx]
        self.modelset = self.modelset[indx]
        self.start = self.start[indx]
        self.stop = self.stop[indx]
        self.stat0 = self.stat0[indx, :]
        self.stat1 = self.stat1[indx, :]
        
    def align_models(self, modelList):
        """Align models of the current StatServer to match a list of models 
            provided as input parameter. The size of the StatServer might be 
            reduced to match the input list of models.
        
        :param modelList: ndarray of strings, list of models to match
        """
        indx = numpy.array([numpy.argwhere(self.modelset == v)[0][0] for v in modelList])
        self.segset = self.segset[indx]
        self.modelset = self.modelset[indx]
        self.start = self.start[indx]
        self.stop = self.stop[indx]
        self.stat0 = self.stat0[indx, :]
        self.stat1 = self.stat1[indx, :]

    @process_parallel_lists
    def accumulate_stat(self, ubm, feature_server, seg_indices=[], numThread=1):
        """Compute statistics for a list of sessions which indices 
            are given in segIndices.
        
        :param ubm: a Mixture object used to compute the statistics
        :param feature_server: featureServer object
        :param seg_indices: list of indices of segments to process
              if segIndices is an empty list, process all segments.
        :param numThread: number of parallel process to run
        """
        assert isinstance(ubm, Mixture), 'First parameter has to be a Mixture'
        assert isinstance(feature_server, FeaturesServer), 'Second parameter has to be a FeaturesServer'
        if (list(seg_indices) == []) | (self.stat0.shape[0] != self.segset.shape[0]) | (self.stat1.shape[0] != self.segset.shape[0]): 
            self.stat0 = numpy.zeros((self.segset.shape[0], ubm.distrib_nb()), dtype=STAT_TYPE)
            self.stat1 = numpy.zeros((self.segset.shape[0], ubm.sv_size()), dtype=STAT_TYPE)
            seg_indices = range(self.segset.shape[0])
        feature_server.keep_all_features = True

        for idx, show in enumerate(seg_indices):
            logging.debug('Compute statistics for %s', self.segset[idx])

            # If using a FeaturesExtractor, get the channel number by checking the extension of the show
            channel = 0
            if feature_server.features_extractor is not None \
                    and show.endswith(feature_server.double_channel_extension[1]):
                channel = 1

            cep, vad = feature_server.load(show, channel=channel)
            stop = vad.shape[0] if self.stop[idx] is None else min(self.stop[idx], vad.shape[0])
            data = cep[self.start[idx]:stop, :]
            data = data[vad[self.start[idx]:stop], :]

            # Verify that frame dimension is equal to gmm dimension
            if not ubm.dim() == data.shape[1]:
                raise Exception('dimension of ubm and features differ: {:d} / {:d}'.format(ubm.dim(), data.shape[1]))
            else:
                lp = ubm.compute_log_posterior_probabilities(data)
                pp, foo = sum_log_probabilities(lp)
                # Compute 0th-order statistics
                self.stat0[idx, :] = pp.sum(0)
                # Compute 1st-order statistics
                self.stat1[idx, :] = numpy.reshape(numpy.transpose(
                        numpy.dot(data.transpose(), pp)), ubm.sv_size())

    def get_mean_stat1(self):
        """Return the mean of first order statistics
        
        return: the mean array of the first order statistics.
        """
        mu = numpy.mean(self.stat1, axis=0)
        return mu

    def norm_stat1(self):
        """Divide all first-order statistics by their euclidian norm."""
        self.stat1 = (self.stat1.transpose() / numpy.linalg.norm(self.stat1, axis=1)).transpose()

    def rotate_stat1(self, R):
        """Rotate first-order statistics by a right-product.
        
        :param R: ndarray, matrix to use for right product on the first order 
            statistics.
        """
        self.stat1 = numpy.dot(self.stat1, R)

    def center_stat1(self, mu):
        """Center first orde statistics.
        
        :param mu: array to center on.
        """
        dim = self.stat1.shape[1] / self.stat0.shape[1]
        index_map = numpy.repeat(numpy.arange(self.stat0.shape[1]), dim)
        self.stat1 = self.stat1 - (self.stat0[:, index_map] * mu)

    def subtract_weighted_stat1(self, sts):
        """Subtract the stat1 from from the sts StatServer to the stat1 of 
        the current StatServer after multiplying by the zero-order statistics
        from the current statserver
        
        :param sts: a StatServer
        
        :return: a new StatServer
        """
        newSts = copy.deepcopy(self)         
        
        # check the compatibility of the two statservers
        #   exact match of the sessions and dimensions of the stat0 and stat1
        if all(numpy.sort(sts.modelset) == numpy.sort(self.modelset))and \
                all(numpy.sort(sts.segset) == numpy.sort(self.segset)) and \
                (sts.stat0.shape == self.stat0.shape) and \
                (sts.stat1.shape == self.stat1.shape):
    
            # align sts according to self.segset
            idx = self.segset.argsort()
            idx_sts = sts.segset.argsort()
            newSts.stat1[idx, :] = sts.stat1[idx_sts, :]
            
            # Subtract the stat1
            dim = self.stat1.shape[1] / self.stat0.shape[1]
            index_map = numpy.repeat(numpy.arange(self.stat0.shape[1]), dim)
            newSts.stat1 = self.stat1 - (self.stat0[:, index_map] * newSts.stat1)
            
        else:
            raise Exception('Statserver are not compatible')
        
        return newSts

    def whiten_stat1(self, mu, Sigma, isSqrInvSigma=False):
        """Whiten first-order statistics
        
        :param mu: array, mean vector to be subtracted from the statistics
        :param Sigma: narray, co-variance matrix or covariance super-vector
        :param isSqrInvSigma: boolean, True if the input Sigma matrix is the inverse of the square root of a covariance
         matrix
        """
        if Sigma.ndim == 2:
            # Compute the inverse square root of the co-variance matrix Sigma
            sqrInvSigma = Sigma
            
            if not isSqrInvSigma:
                eigenValues, eigenVectors = scipy.linalg.eigh(Sigma)
                ind = eigenValues.real.argsort()[::-1]  
                eigenValues = eigenValues.real[ind]
                eigenVectors = eigenVectors.real[:, ind]
            
                sqrInv_Eval_sigma = 1 / numpy.sqrt(eigenValues.real)
                sqrInvSigma = numpy.dot(eigenVectors, numpy.diag(sqrInv_Eval_sigma))
            else:
                pass

            # Whitening of the first-order statistics
            self.center_stat1(mu)
            self.rotate_stat1(sqrInvSigma)
        elif Sigma.ndim == 1:
            self.center_stat1(mu)
            self.stat1 = self.stat1 / numpy.sqrt(Sigma)
        else:
            raise Exception('Wrong dimension of Sigma, must be 1 or 2')
            
    def whiten_cholesky_stat1(self, mu, Sigma):
        """Whiten first-order statistics by using Cholesky decomposition of 
        Sigma
        
        :param mu: array, mean vector to be subtracted from the statistics
        :param Sigma: narray, co-variance matrix or covariance super-vector
        """
        if Sigma.ndim == 2:
            # Compute the inverse square root of the co-variance matrix Sigma
            invSigma = scipy.linalg.inv(Sigma)
            chol_invcov = scipy.linalg.cholesky(invSigma).T

            # Whitening of the first-order statistics
            self.center_stat1(mu)
            self.stat1 = self.stat1.dot(chol_invcov)            

        elif Sigma.ndim == 1:
            self.center_stat1(mu)
            self.stat1 = self.stat1 / numpy.sqrt(Sigma)
        else:
            raise Exception('Wrong dimension of Sigma, must be 1 or 2')

    def get_total_covariance_stat1(self):
        """Compute and return the total covariance matrix of the first-order 
            statistics.
        
        :return: the total co-variance matrix of the first-order statistics 
                as a ndarray.
        """
        C = self.stat1 - self.stat1.mean(axis=0)
        Sigma = numpy.dot(C.transpose(), C) / self.stat1.shape[0]
        return Sigma

    def get_within_covariance_stat1(self):
        """Compute and return the within-class covariance matrix of the 
            first-order statistics.
        
        :return: the within-class co-variance matrix of the first-order statistics 
              as a ndarray.
        """
        vectSize = self.stat1.shape[1]
        uniqueSpeaker = numpy.unique(self.modelset)
        W = numpy.zeros((vectSize, vectSize))

        for speakerID in uniqueSpeaker:
            spkCtrVec = self.get_model_stat1(speakerID) \
                        - numpy.mean(self.get_model_stat1(speakerID), axis=0)
            W += numpy.dot(spkCtrVec.transpose(), spkCtrVec)
        W /= self.stat1.shape[0]
        return W

    def get_between_covariance_stat1(self):
        """Compute and return the between-class covariance matrix of the 
            first-order statistics.
        
        :return: the between-class co-variance matrix of the first-order 
            statistics as a ndarray.
        """
        vectSize = self.stat1.shape[1]
        uniqueSpeaker = numpy.unique(self.modelset)
        B = numpy.zeros((vectSize, vectSize))

        # Compute overall mean first-order statistics
        mu = self.get_mean_stat1()

        # Compute and accumulate mean first-order statistics for each class
        for speakerID in uniqueSpeaker:
            spkSessions = self.get_model_stat1(speakerID)
            tmp = numpy.mean(spkSessions, axis=0) - mu
            B += (spkSessions.shape[0] * numpy.outer(tmp, tmp))
        B /= self.stat1.shape[0]
        return B

    def get_lda_matrix_stat1(self, rank):
        """Compute and return the Linear Discriminant Analysis matrix 
            on the first-order statistics. Columns of the LDA matrix are ordered
            according to the corresponding eigenvalues in descending order.
        
        :param rank: integer, rank of the LDA matrix to return
        
        :return: the LDA matrix of rank "rank" as a ndarray
        """
        vectSize = self.stat1.shape[1]
        uniqueSpeaker = numpy.unique(self.modelset)

        mu = self.get_mean_stat1()

        classMeans = numpy.zeros((uniqueSpeaker.shape[0], vectSize))
        Sw = numpy.zeros((vectSize, vectSize))

        spkIdx = 0
        for speakerID in uniqueSpeaker:
            spkSessions = self.get_model_stat1(speakerID) \
                        - numpy.mean(self.get_model_stat1(speakerID), axis=0)
            Sw += numpy.dot(spkSessions.transpose(), spkSessions) / spkSessions.shape[0]
            classMeans[spkIdx, :] = numpy.mean(self.get_model_stat1(speakerID), axis=0)
            spkIdx += 1

        # Compute Between-class scatter matrix
        classMeans = classMeans - mu
        Sb = numpy.dot(classMeans.transpose(), classMeans)

        # Compute the Eigenvectors & eigenvalues of the discrimination matrix
        DiscriminationMatrix = numpy.dot(Sb, scipy.linalg.inv(Sw)).transpose()
        eigenValues, eigenVectors = scipy.linalg.eigh(DiscriminationMatrix)
        eigenValues = eigenValues.real
        eigenVectors = eigenVectors.real

        # Rearrange the eigenvectors according to decreasing eigenvalues
        # get indexes of the rank top eigen values
        idx = eigenValues.real.argsort()[-rank:][::-1]
        L = eigenVectors[:, idx]
        return L

    def get_mahalanobis_matrix_stat1(self):
        """Compute and return Mahalanobis matrix of first-order statistics.
        
        :return: the mahalanobis matrix computed on the first-order 
            statistics as a ndarray
        """
        W = self.get_within_covariance_stat1()
        M = scipy.linalg.inv(W)
        return M

    def get_wccn_choleski_stat1(self):
        """Compute and return the lower Cholesky decomposition matrix of the
            Within Class Co-variance Normalization matrix on the first-order
            statistics.
        
        :return: the lower Choleski decomposition of the WCCN matrix 
            as a ndarray
        """
        vectSize = self.stat1.shape[1]
        uniqueSpeaker = numpy.unique(self.modelset)
        WCCN = numpy.zeros((vectSize, vectSize))

        for speakerID in uniqueSpeaker:
            spkCtrVec = self.get_model_stat1(speakerID) \
                      - numpy.mean(self.get_model_stat1(speakerID), axis=0)
            WCCN += numpy.dot(spkCtrVec.transpose(), spkCtrVec)
            # WCCN = WCCN + np.dot(spkCtrVec.transpose(),
            #     spkCtrVec) / spkCtrVec.shape[0]

        WCCN /= self.stat1.shape[0]
        # WCCN = WCCN / self.uniqueSpeaker.shape[0]

        # Choleski decomposition of the WCCN matrix
        invW = scipy.linalg.inv(WCCN)
        W = scipy.linalg.cholesky(invW).T
        return W

    def get_nap_matrix_stat1(self, coRank):
        """Compute return the Nuisance Attribute Projection matrix
            from first-order statistics.
        
        :param coRank: co-rank of the Nuisance Attribute Projection matrix
        
        :return: the NAP matrix of rank "coRank"
        """
        vectSize = self.stat1.shape[1]
        W = numpy.dot(self.stat1, self.stat1.transpose()) / vectSize
        eigenValues, eigenVectors = scipy.linalg.eigh(W)

        # Rearrange the eigenvectors according to decreasing eigenvalues
        # get indexes of the rank top eigen values
        idx = eigenValues.real.argsort()[-coRank:][::-1]
        N = numpy.dot(self.stat1.transpose(), eigenVectors[:, idx])
        N = numpy.dot(N, numpy.diag(1 / numpy.sqrt(vectSize * eigenValues.real[idx])))
        return N

    def adapt_mean_MAP(self, ubm, r=16, norm=False):
        """Maximum A Posteriori adaptation of the mean super-vector of ubm,
            train one model per segment.
        
        :param ubm: a Mixture object to adapt
        :param r: float, the relevant factor for MAP adaptation
        :param norm: boolean, normalize by using the UBM co-variance. 
            Default is False
          
        :return: a StatServer with 1 as stat0 and the MAP adapted super-vectors 
              as stat1
        """
        gsv_statserver = StatServer()
        gsv_statserver.modelset = self.modelset
        gsv_statserver.segset = self.segset
        gsv_statserver.start = self.start
        gsv_statserver.stop = self.stop
        gsv_statserver.stat0 = numpy.ones((self.segset. shape[0], 1), dtype=STAT_TYPE)

        index_map = numpy.repeat(numpy.arange(ubm.distrib_nb()), ubm.dim())

        # Adapt mean vectors
        alpha = self.stat0 / (self.stat0 + r)   # Adaptation coefficient
        M = self.stat1 / self.stat0[:, index_map]
        M[numpy.isnan(M)] = 0  # Replace NaN due to divide by zeros
        M = alpha[:, index_map] * M + (1 - alpha[:, index_map]) * numpy.tile(ubm.get_mean_super_vector(), (M.shape[0], 1))

        if norm:
            if ubm.invcov.ndim == 2:
                # Normalization corresponds to KL divergence
                w = numpy.repeat(ubm.w, ubm.dim())
                KLD = numpy.sqrt(w * ubm.get_invcov_super_vector())

            M = M * KLD

        gsv_statserver.stat1 = M.astype(dtype=STAT_TYPE)
        gsv_statserver.validate()
        return gsv_statserver

    def adapt_mean_MAP_multisession(self, ubm, r=16, norm=False):
        """Maximum A Posteriori adaptation of the mean super-vector of ubm,
            train one model per model in the modelset by summing the statistics
            of the multiple segments.
        
        :param ubm: a Mixture object to adapt 
        :param r: float, the relevant factor for MAP adaptation
        :param norm: boolean, normalize by using the UBM co-variance. 
            Default is False
          
        :return: a StatServer with 1 as stat0 and the MAP adapted super-vectors 
              as stat1
        """
        gsv_statserver = StatServer()
        gsv_statserver.modelset = numpy.unique(self.modelset)
        gsv_statserver.segset = numpy.unique(self.modelset)
        gsv_statserver.start = numpy.empty(gsv_statserver.modelset.shape, dtype="|O")
        gsv_statserver.stop = numpy.empty(gsv_statserver.modelset.shape, dtype="|O")
        gsv_statserver.stat0 = numpy.ones((numpy.unique(self.modelset).shape[0], 1), dtype=STAT_TYPE)

        index_map = numpy.repeat(numpy.arange(ubm.distrib_nb()), ubm.dim())

        # Sum the statistics per model
        modelStat = self.sum_stat_per_model()[0]
        
        # Adapt mean vectors
        alpha = modelStat.stat0 / (modelStat.stat0 + r)
        M = modelStat.stat1 / modelStat.stat0[:, index_map]
        M[numpy.isnan(M)] = 0  # Replace NaN due to divide by zeros
        M = alpha[:, index_map] * M + (1 - alpha[:, index_map]) * numpy.tile(ubm.get_mean_super_vector(), (M.shape[0], 1))

        if norm:
            if ubm.invcov.ndim == 2:
                # Normalization corresponds to KL divergence
                w = numpy.repeat(ubm.w, ubm.dim())
                KLD = numpy.sqrt(w * ubm.get_invcov_super_vector())

            M = M * KLD

        gsv_statserver.stat1 = M.astype(dtype=STAT_TYPE)
        gsv_statserver.validate()
        return gsv_statserver

    def precompute_svm_kernel_stat1(self):
        """Pre-compute the Kernel for SVM training and testing,
            the output parameter is a matrix that only contains the impostor
            part of the Kernel. This one has to be completed by the
            target-dependent part during training and testing.
        
        :return: the impostor part of the SVM Graam matrix as a ndarray
        """
        K = numpy.dot(self.stat1, self.stat1.transpose())
        return K

    def ivector_extraction_weight(self, ubm, W, Tnorm, delta=numpy.array([])):
        """Compute i-vectors using the ubm weight approximation.
            For more information, refers to:
            
            Glembeck, O.; Burget, L.; Matejka, P.; Karafiat, M. & Kenny, P. 
            "Simplification and optimization of I-Vector extraction," 
            in IEEE International Conference on Acoustics, Speech, and Signal 
            Processing, ICASSP, 2011, 4516-4519
        
        :param ubm: a Mixture used as UBM for i-vector estimation
        :param W: fix matrix pre-computed using the weights from 
            the UBM and the total variability matrix
        :param Tnorm: total variability matrix pre-normalized using 
                the co-variance of the UBM
        :param delta: men vector if re-estimated using minimum divergence 
                criteria
        
        :return: a StatServer which zero-order statistics are 1 
                and first-order statistics are approximated i-vectors.
        """
        # check consistency of dimensions for delta, Tnorm, W, ubm
        assert ubm.get_invcov_super_vector().shape[0] == Tnorm.shape[0], \
            'UBM and TV matrix dimension are not consistent'
        if delta.shape == (0, ):
            delta = numpy.zeros(ubm.get_invcov_super_vector().shape)
        assert ubm.get_invcov_super_vector().shape[0] == delta.shape[0],\
            'Minimum divergence mean and TV matrix dimension not consistent'
        assert W.shape[0] == Tnorm.shape[1], 'W and TV matrix dimension are not consistent'
        ivector_size = Tnorm.shape[1]
    
        # Sum stat0
        sumStat0 = self.stat0.sum(axis=1)
    
        # Center and normalize first-order statistics 
        # for the case of diagonal covariance UBM
        self.whiten_stat1(delta, 1./ubm.get_invcov_super_vector())
    
        X = numpy.dot(self.stat1, Tnorm)
    
        enroll_iv = StatServer()
        enroll_iv.modelset = self.modelset
        enroll_iv.segset = self.segset
        enroll_iv.stat0 = numpy.ones((enroll_iv.segset.shape[0], 1))
        enroll_iv.stat1 = numpy.zeros((enroll_iv.segset.shape[0], ivector_size))
        for iv in range(self.stat0.shape[0]):  # loop on i-vector
            logging.debug('Estimate i-vector [ %d / %d ]', iv + 1, self.stat0.shape[0])
            # Compute precision matrix
            L = numpy.eye(ivector_size) + sumStat0[iv] * W
            # Estimate i-vector
            enroll_iv.stat1[iv, :] = scipy.linalg.solve(L, X[iv, :])

        return enroll_iv

    def ivector_extraction_eigen_decomposition(self, ubm, Q, D_bar_c, 
                                               Tnorm, delta=numpy.array([])):
        """Compute i-vectors using the eigen decomposition approximation.
            For more information, refers to[Glembeck09]_
        
        :param ubm: a Mixture used as UBM for i-vector estimation
        :param Q: Q matrix as described in [Glembeck11]
        :param D_bar_c: matrices as described in [Glembeck11]
        :param Tnorm: total variability matrix pre-normalized using 
                the co-variance of the UBM
        :param delta: men vector if re-estimated using minimum divergence 
                criteria
        
        :return: a StatServer which zero-order statistics are 1 
                and first-order statistics are approximated i-vectors.
        """
        # check consistency of dimensions for delta, Tnorm, Q, D_bar_c, ubm
        assert ubm.get_invcov_super_vector().shape[0] == Tnorm.shape[0], \
            'UBM and TV matrix dimension not consistent'
        if delta.shape == (0, ):
            delta = numpy.zeros(ubm.get_invcov_super_vector().shape)
        assert ubm.get_invcov_super_vector().shape[0] == delta.shape[0], \
            'Minimum divergence mean and TV matrix dimension not consistent'
        assert D_bar_c.shape[1] == Tnorm.shape[1], \
            'D_bar_c and TV matrix dimension are not consistent'
        assert D_bar_c.shape[0] == ubm.w.shape[0], \
            'D_bar_c and UBM dimension are not consistent'
    
        ivector_size = Tnorm.shape[1]

        # Center and normalize first-order statistics 
        # for the case of diagonal covariance UBM
        self.whiten_stat1(delta, 1./ubm.get_invcov_super_vector())
    
        X = numpy.dot(self.stat1, Tnorm)
    
        enroll_iv = StatServer()
        enroll_iv.modelset = self.modelset
        enroll_iv.segset = self.segset
        enroll_iv.stat0 = numpy.ones((enroll_iv.segset.shape[0], 1), dtype=STAT_TYPE)
        enroll_iv.stat1 = numpy.zeros((enroll_iv.segset.shape[0], ivector_size), dtype=STAT_TYPE)
        for iv in range(self.stat0.shape[0]):  # loop on i-vector
            logging.debug('Estimate i-vector [ %d / %d ]', iv + 1, self.stat0.shape[0])

            # Compute precision matrix
            diag_L = 1 + numpy.sum(numpy.dot(numpy.diag(self.stat0[iv, :]), D_bar_c), axis=0)

            # Estimate i-vector
            enroll_iv.stat1[iv, :] = \
                reduce(numpy.dot, [X[iv, :], Q, numpy.diag(1/diag_L), Q.transpose()])

        return enroll_iv

    def estimate_spectral_norm_stat1(self, it=1, mode='efr'):
        """Compute meta-parameters for Spectral Normalization as described
            in [Bousquet11]_
            
            Can be used to perform Eigen Factor Radial or Spherical Nuisance
            Normalization. Default behavior is equivalent to Length Norm as 
            described in [Garcia-Romero11]_
            
            Statistics are transformed while the meta-parameters are 
            estimated.
        
        :param it: integer, number of iterations to perform
        :param mode: string, can be 
                - efr for Eigen Factor Radial
                - sphNorm, for Spherical Nuisance Normalization
                  
        :return: a tupple of two lists:
                - a list of mean vectors
                - a list of co-variance matrices as ndarrays
        """
        spectral_norm_mean = []
        spectral_norm_cov = []
        tmp_iv = copy.deepcopy(self)
        
        for i in range(it):
            # estimate mena and covariance matrix
            spectral_norm_mean.append(tmp_iv.get_mean_stat1())
            
            if mode == 'efr':
                spectral_norm_cov.append(tmp_iv.get_total_covariance_stat1())
            elif mode == 'sphNorm':
                spectral_norm_cov.append(tmp_iv.get_within_covariance_stat1())

            # Center and whiten the statistics
            tmp_iv.whiten_stat1(spectral_norm_mean[i], spectral_norm_cov[i])
            tmp_iv.norm_stat1()
        return spectral_norm_mean, spectral_norm_cov

    def spectral_norm_stat1(self, spectral_norm_mean, spectral_norm_cov, isSqrInvSigma=False):
        """Apply Spectral Sormalization to all first order statistics.
            See more details in [Bousquet11]_
            
            The number of iterations performed is equal to the length of the
            input lists.
        
        :param spectral_norm_mean: a list of mean vectors
        :param spectral_norm_cov: a list of co-variance matrices as ndarrays
        :param isSqrInvSigma: boolean, True if
        """
        assert len(spectral_norm_mean) == len(spectral_norm_cov), \
            'Number of mean vectors and covariance matrices is different'

        for mu, Cov in zip(spectral_norm_mean, spectral_norm_cov):
            self.whiten_stat1(mu, Cov, isSqrInvSigma)
            self.norm_stat1()

    def __repr__(self):
        ch = '-' * 30 + '\n'
        ch += 'modelset: ' + self.modelset.__repr__() + '\n'
        ch += 'segset: ' + self.segset.__repr__() + '\n'
        ch += 'seg start:' + self.start.__repr__() + '\n'
        ch += 'seg stop:' + self.stop.__repr__() + '\n'
        ch += 'stat0:' + self.stat0.__repr__() + '\n'
        ch += 'stat1:' + self.stat1.__repr__() + '\n'
        ch += '-' * 30 + '\n'
        return ch

    def sum_stat_per_model(self):
        """Sum the zero- and first-order statistics per model and store them 
        in a new StatServer.        
        
        :return: a StatServer with the statistics summed per model
        """
        sts_per_model = sidekit.StatServer()
        sts_per_model.modelset = numpy.unique(self.modelset)
        sts_per_model.segset = sts_per_model.modelset
        sts_per_model.stat0 = numpy.zeros((sts_per_model.modelset.shape[0], self.stat0.shape[1]), dtype=STAT_TYPE)
        sts_per_model.stat1 = numpy.zeros((sts_per_model.modelset.shape[0], self.stat1.shape[1]), dtype=STAT_TYPE)
        sts_per_model.start = numpy.empty(sts_per_model.segset.shape, '|O')
        sts_per_model.stop = numpy.empty(sts_per_model.segset.shape, '|O')
        
        session_per_model = numpy.zeros(numpy.unique(self.modelset).shape[0])

        for idx, model in enumerate(sts_per_model.modelset):
            sts_per_model.stat0[idx, :] = self.get_model_stat0(model).sum(axis=0)
            sts_per_model.stat1[idx, :] = self.get_model_stat1(model).sum(axis=0)
            session_per_model[idx] += self.get_model_stat1(model).shape[0]
        return sts_per_model, session_per_model

    def mean_stat_per_model(self):
        """Average the zero- and first-order statistics per model and store them
        in a new StatServer.

        :return: a StatServer with the statistics averaged per model
        """
        sts_per_model = sidekit.StatServer()
        sts_per_model.modelset = numpy.unique(self.modelset)
        sts_per_model.segset = sts_per_model.modelset
        sts_per_model.stat0 = numpy.zeros((sts_per_model.modelset.shape[0], self.stat0.shape[1]), dtype=STAT_TYPE)
        sts_per_model.stat1 = numpy.zeros((sts_per_model.modelset.shape[0], self.stat1.shape[1]), dtype=STAT_TYPE)
        sts_per_model.start = numpy.empty(sts_per_model.segset.shape, '|O')
        sts_per_model.stop = numpy.empty(sts_per_model.segset.shape, '|O')

        for idx, model in enumerate(sts_per_model.modelset):
            sts_per_model.stat0[idx, :] = self.get_model_stat0(model).mean(axis=0)
            sts_per_model.stat1[idx, :] = self.get_model_stat1(model).mean(axis=0)
        return sts_per_model

    def _expectation(self, Phi, mean, Sigma, session_per_model, batch_size=100, numThread=1):
        """
        dans cette version, on considère que les stats NE sont PAS blanchis avant
        """
        r = Phi.shape[-1]
        d = int(self.stat1.shape[1] / self.stat0.shape[1])
        C = self.stat0.shape[1]

        """Whiten the statistics and multiply the covariance matrix by the 
        square root of the inverse of the residual covariance"""
        self.whiten_stat1(mean, Sigma)
        Phi_white = copy.deepcopy(Phi)
        if Sigma.ndim == 2:
            eigenValues, eigenVectors = scipy.linalg.eigh(Sigma)
            ind = eigenValues.real.argsort()[::-1]  
            eigenValues = eigenValues.real[ind]
            eigenVectors = eigenVectors.real[:, ind]
            sqrInv_Eval_sigma = 1 / numpy.sqrt(eigenValues.real)
            sqrInvSigma = numpy.dot(eigenVectors, numpy.diag(sqrInv_Eval_sigma))
            Phi_white = sqrInvSigma.T.dot(Phi)
        elif Sigma.ndim == 1:
            sqrInvSigma = 1/numpy.sqrt(Sigma)
            Phi_white = Phi * sqrInvSigma[:, None]
            
        # Replicate self.stat0
        index_map = numpy.repeat(numpy.arange(C), d)
        _stat0 = self.stat0[:, index_map]

        # Create accumulators for the list of models to process
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            _A = numpy.zeros((C, r, r), dtype=STAT_TYPE)
            tmp_A = multiprocessing.Array(ct, _A.size)
            _A = numpy.ctypeslib.as_array(tmp_A.get_obj())
            _A = _A.reshape(C, r, r)

        _C = numpy.zeros((r, d * C), dtype=STAT_TYPE)
        
        _R = numpy.zeros((r, r), dtype=STAT_TYPE)
        _r = numpy.zeros(r, dtype=STAT_TYPE)

        # Process in batches in order to reduce the memory requirement
        batch_nb = int(numpy.floor(self.segset.shape[0]/float(batch_size) + 0.999))
        
        for batch in range(batch_nb):
            batch_start = batch * batch_size
            batch_stop = min((batch + 1) * batch_size, self.segset.shape[0])
            batch_len = batch_stop - batch_start

            # Allocate the memory to save time
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                E_h = numpy.zeros((batch_len, r), dtype=STAT_TYPE)
                tmp_E_h = multiprocessing.Array(ct, E_h.size)
                E_h = numpy.ctypeslib.as_array(tmp_E_h.get_obj())
                E_h = E_h.reshape(batch_len, r)

                E_hh = numpy.zeros((batch_len, r, r), dtype=STAT_TYPE)
                tmp_E_hh = multiprocessing.Array(ct, E_hh.size)
                E_hh = numpy.ctypeslib.as_array(tmp_E_hh.get_obj())
                E_hh = E_hh.reshape(batch_len, r, r)

            # loop on model id's
            fa_model_loop(batch_start=batch_start, mini_batch_indices=numpy.arange(batch_len),
                          r=r, Phi_white=Phi_white, Phi=Phi, Sigma=Sigma,
                          stat0=_stat0, stat1=self.stat1,
                          E_h=E_h, E_hh=E_hh, numThread=numThread)
            
            # Accumulate for minimum divergence step
            _r += numpy.sum(E_h * session_per_model[batch_start:batch_stop, None], axis=0)
            _R += numpy.sum(E_hh, axis=0)

            if sqrInvSigma.ndim == 2:
                _C += E_h.T.dot(self.stat1[batch_start:batch_stop, :]).dot(scipy.linalg.inv(sqrInvSigma))
            elif sqrInvSigma.ndim == 1:
                _C += E_h.T.dot(self.stat1[batch_start:batch_stop, :]) / sqrInvSigma
            
            # Parallelized loop on the model id's
            fa_distribution_loop(distrib_indices=numpy.arange(C), _A=_A,
                                 stat0=self.stat0, batch_start=batch_start,
                                 batch_stop=batch_stop, E_hh=E_hh,
                                 numThread=numThread)

        _r /= session_per_model.sum()
        _R /= session_per_model.shape[0]
        
        return _A, _C, _R  

    def _maximization(self, Phi, _A, _C, _R=None, Sigma_obs=None, session_number=None):
        """
        """
        r = Phi.shape[1]
        d = self.stat1.shape[1] // self.stat0.shape[1]
        C = self.stat0.shape[1]
        
        for c in range(C):
            distrib_idx = range(c * d, (c+1) * d)
            Phi[distrib_idx, :] = scipy.linalg.solve(_A[c], _C[:, distrib_idx]).T

        # Update the residual covariance if needed 
        # (only for full co-variance case of PLDA
        Sigma = None
        if Sigma_obs is not None:
            Sigma = Sigma_obs - Phi.dot(_C)/session_number

        # MINIMUM DIVERGENCE STEP
        if _R is not None:
            print('applyminDiv reestimation')
            ch = scipy.linalg.cholesky(_R)
            Phi = Phi.dot(ch)

        return Phi, Sigma

    def estimate_between_class(self, itNb, V, mean, Sigma_obs, batch_size=100, Ux=None, Dz=None,
                               minDiv=True, numThread=1, re_estimate_residual=False, save_partial=False):
        """Estimate the factor loading matrix for the between class covariance

        :param itNb: number of iterations to estimate the between class covariance matrix
        :param V: initial between class covariance matrix
        :param mean: global mean vector
        :param Sigma_obs: covariance matrix of the input data
        :param batch_size: size of the batches to process one by one to reduce the memory usage
        :param Ux: statserver of supervectors
        :param Dz: statserver of supervectors
        :param minDiv: boolean, if True run the minimum divergence step after maximization
        :param numThread: number of parallel process to run
        :param re_estimate_residual: boolean, if True the residual covariance matrix is re-estimated (for PLDA)
        
        :return: the within class factor loading matrix
        """
        
        # Initialize the covariance
        Sigma = Sigma_obs
        import time 
        # Estimate F by iterating the EM algorithm
        for it in range(itNb):
            logging.info('Estimate between class covariance, it %d / %d',
                         it + 1, itNb)
            start = time.time()
            # Dans la fonction estimate_between_class
            model_shifted_stat = copy.deepcopy(self)
        
            # subtract channel effect, Ux, if already estimated 
            if Ux is not None:
                model_shifted_stat = model_shifted_stat.subtract_weighted_stat1(Ux)
            
            # Sum statistics per speaker
            model_shifted_stat, session_per_model = model_shifted_stat.sum_stat_per_model()
            # subtract residual, Dz, if already estimated
            if Dz is not None:
                model_shifted_stat = model_shifted_stat.subtract(Dz)                     
                    
            # E-step
            print("E_step")
            _A, _C, _R = model_shifted_stat._expectation(V, mean, Sigma, session_per_model, batch_size, numThread)
                
            if not minDiv:
                _R = None
            
            # M-step
            print("M_step")
            if re_estimate_residual:
                V, Sigma = model_shifted_stat._maximization(V, _A, _C, _R, Sigma_obs, session_per_model.sum())
            else:
                V = model_shifted_stat._maximization(V, _A, _C, _R)[0]

            if Sigma.ndim == 2:
                logging.info('Likelihood after iteration %d / %f', it + 1, compute_llk(self, V, Sigma))
            
            del model_shifted_stat
            print("time for iteration = {}".format(time.time() - start))

            if save_partial:
                sidekit.sidekit_io.write_fa_hdf5((mean, V, None, None, Sigma), save_partial + "_{}_between_class.h5".format(it))

        return V, Sigma

    def estimate_within_class(self, itNb, U, mean, Sigma_obs, batch_size=100,
                              Vy=None, Dz=None, 
                              minDiv=True, numThread=1, save_partial=False):
        """Estimate the factor loading matrix for the within class covariance

        :param itNb: number of iterations to estimate the within class covariance matrix
        :param U: initial within class covariance matrix
        :param mean: mean of the input data
        :param Sigma_obs: co-variance matrix of the input data
        :param batch_size: number of sessions to process per batch to optimize memory usage
        :param Vy: statserver of supervectors
        :param Dz: statserver of supervectors
        :param minDiv: boolean, if True run the minimum divergence step after maximization
        :param numThread: number of parallel process to run
        
        :return: the within class factor loading matrix
        """
        session_shifted_stat = copy.deepcopy(self)
        
        session_per_model = numpy.ones(session_shifted_stat.modelset.shape[0])
        # Estimate F by iterating the EM algorithm
        for it in range(itNb):
            logging.info('Estimate between class covariance, it %d / %d',
                         it + 1, itNb)

            session_shifted_stat = self
            # subtract channel effect, Ux,  if already estimated 
            # and sum per speaker
            if Vy is not None:
                session_shifted_stat = session_shifted_stat.subtract_weighted_stat1(Vy)
                # session_shifted_stat = self.subtract_weighted_stat1(Vy)

            # subtract residual, Dz, if already estimated
            if Dz is not None:
                session_shifted_stat = session_shifted_stat.subtract_weighted_stat1(Dz)
        
            # E step
            A, C, R = session_shifted_stat._expectation(U, mean, Sigma_obs,
                                                        session_per_model,
                                                        batch_size, numThread)

            # M step
            if not minDiv:
                R = None
            U = session_shifted_stat._maximization(U, A, C, R)[0]

            if save_partial:
                sidekit.sidekit_io.write_fa_hdf5((None, None, U, None, None), save_partial + "_{}_within_class.h5")

        return U
        
    def estimate_map(self, itNb, D, mean, Sigma, Vy=None, Ux=None, numThread=1, save_partial=False):
        """
        
        :param itNb: number of iterations to estimate the MAP covariance matrix
        :param D: Maximum a Posteriori marix to estimate
        :param mean: mean of the input parameters
        :param Sigma: residual covariance matrix
        :param Vy: statserver of supervectors
        :param Ux: statserver of supervectors
        :param numThread: number of parallel process to run
        
        :return: the MAP covariance matrix into a vector as it is diagonal
        """
        model_shifted_stat = copy.deepcopy(self)
        
        logging.info('Estimate MAP matrix')
        # subtract speaker and channel if already estimated
        model_shifted_stat.center_stat1(mean)
        if Vy is not None:
            model_shifted_stat = model_shifted_stat.subtract_weighted_stat1(Vy)
        if Ux is not None:
            model_shifted_stat = model_shifted_stat.subtract_weighted_stat1(Ux)

        # Sum statistics per speaker
        model_shifted_stat = model_shifted_stat.sum_stat_per_model()[0]
        
        r = D.shape[-1]
        d = model_shifted_stat.stat1.shape[1] / model_shifted_stat.stat0.shape[1]
        C = model_shifted_stat.stat0.shape[1]

        # Replicate self.stat0
        index_map = numpy.repeat(numpy.arange(C), d)
        _stat0 = model_shifted_stat.stat0[:, index_map]
        
        # Estimate D by iterating the EM algorithm
        for it in range(itNb):
            logging.info('Estimate MAP covariance, it %d / %d', it + 1, itNb)

            # E step
            E_h = numpy.zeros(model_shifted_stat.stat1.shape, dtype=STAT_TYPE)
            _A = numpy.zeros(D.shape, dtype=STAT_TYPE)
            _C = numpy.zeros(D.shape, dtype=STAT_TYPE)
            for idx in range(model_shifted_stat.modelset.shape[0]):
                Lambda = numpy.ones(D.shape) + (_stat0[idx, :] * D**2 / Sigma)
                E_h[idx] = model_shifted_stat.stat1[idx] * D / (Lambda * Sigma)
                _A = _A + (1/Lambda + E_h[idx]**2) * _stat0[idx, :]
                _C = _C + E_h[idx] * model_shifted_stat.stat1[idx]

            # M step
            D = _C / _A

            if save_partial:
                sidekit.sidekit_io.write_fa_hdf5((None, None, None, D, None), save_partial + "_{}_map.h5")
            
        return D
               
    def estimate_hidden(self, mean, Sigma, V=None, U=None, D=None, numThread=1):
        """
        Assume that the statistics have been whitened and the matrix U
        and V have been multiplied by the squarre root 
        of the inverse of the covariance
        :param mean: global mean of the data to subtract
        :param Sigma: residual covariance matrix of the Factor Analysis model
        :param V: between class covariance matrix
        :param U: within class covariance matrix
        :param D: MAP covariance matrix
        :param numThread: number of parallel process to run
        """
        if V is None:
            V = numpy.zeros((self.stat1.shape[1], 0), dtype=STAT_TYPE)
        if U is None:
            U = numpy.zeros((self.stat1.shape[1], 0), dtype=STAT_TYPE)
        print("elf.stat1.shape = {}, V.shape = {}, U.shape = {}".format(self.stat1.shape, V.shape, U.shape))
        W = numpy.hstack((V, U))
        
        # Estimate yx    
        r = W.shape[1]
        d = int(self.stat1.shape[1] / self.stat0.shape[1])
        C = self.stat0.shape[1]
        session_nb = self.modelset.shape[0]

        self.whiten_stat1(mean, Sigma)
        W_white = copy.deepcopy(W)
        if Sigma.ndim == 2:
            eigenValues, eigenVectors = scipy.linalg.eigh(Sigma)
            ind = eigenValues.real.argsort()[::-1]  
            eigenValues = eigenValues.real[ind]
            eigenVectors = eigenVectors.real[:, ind]
            sqrInv_Eval_sigma = 1 / numpy.sqrt(eigenValues.real)
            sqrInvSigma = numpy.dot(eigenVectors, numpy.diag(sqrInv_Eval_sigma))
            W_white = sqrInvSigma.T.dot(W)
        elif Sigma.ndim == 1:
            sqrInvSigma = 1/numpy.sqrt(Sigma)
            W_white = W * sqrInvSigma[:, None]

        # Replicate self.stat0
        index_map = numpy.repeat(numpy.arange(C), d)
        _stat0 = self.stat0[:, index_map]
    
        # Create accumulators for the list of models to process
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            _A = numpy.zeros((C, r, r), dtype='float')
            tmp_A = multiprocessing.Array(ct, _A.size)
            _A = numpy.ctypeslib.as_array(tmp_A.get_obj())
            _A = _A.reshape(C, r, r)

            _C = numpy.zeros((r, d * C), dtype='float')
               
            # Alocate the memory to save time
            E_h = numpy.zeros((session_nb, r), dtype=STAT_TYPE)
            tmp_E_h = multiprocessing.Array(ct, E_h.size)
            E_h = numpy.ctypeslib.as_array(tmp_E_h.get_obj())
            E_h = E_h.reshape(session_nb, r)

            E_hh = numpy.zeros((session_nb, r, r), dtype=STAT_TYPE)
            tmp_E_hh = multiprocessing.Array(ct, E_hh.size)
            E_hh = numpy.ctypeslib.as_array(tmp_E_hh.get_obj())
            E_hh = E_hh.reshape(session_nb, r, r)

        # Parallelized loop on the model id's
        # mbi = numpy.array_split(numpy.arange(self.segset.shape[0]), numThread)
        fa_model_loop(batch_start=0, mini_batch_indices=numpy.arange(self.segset.shape[0]),
                      r=r, Phi_white=W_white, Phi=W, Sigma=Sigma,
                      stat0=_stat0, stat1=self.stat1,
                      E_h=E_h, E_hh=E_hh, numThread=numThread)

        y = sidekit.StatServer()
        y.modelset = copy.deepcopy(self.modelset)
        y.segset = copy.deepcopy(self.segset)
        y.start = copy.deepcopy(self.start)
        y.stop = copy.deepcopy(self.stop)
        y.stat0 = numpy.ones((self.modelset.shape[0], 1))
        y.stat1 = E_h[:, :V.shape[1]]

        x = sidekit.StatServer()
        x.modelset = copy.deepcopy(self.modelset)
        x.segset = copy.deepcopy(self.segset)
        x.start = copy.deepcopy(self.start)
        x.stop = copy.deepcopy(self.stop)
        x.stat0 = numpy.ones((self.modelset.shape[0], 1))
        x.stat1 = E_h[:, V.shape[1]:]
        
        z = sidekit.StatServer()
        if D is not None:
            
            # subtract Vy + Ux from the first-order statistics
            VUyx = copy.deepcopy(self)
            VUyx.stat1 = E_h.dot(W.T)
            self = self.subtract_weighted_stat1(VUyx)
            
            # estimate z
            z.modelset = copy.deepcopy(self.modelset)
            z.segset = copy.deepcopy(self.segset)
            z.stat0 = numpy.ones((self.modelset.shape[0], 1), dtype=STAT_TYPE)
            z.stat1 = numpy.ones((self.modelset.shape[0], D.shape[0]), dtype=STAT_TYPE)
            
            for idx in range(self.modelset.shape[0]):
                Lambda = numpy.ones(D.shape, dtype=STAT_TYPE) + (_stat0[idx, :] * D**2)
                z.stat1[idx] = self.stat1[idx] * D / Lambda            
         
        return y, x, z

    #@profile
    def factor_analysis(self, rank_F, rank_G=0, rank_H=None, re_estimate_residual=False,
                        itNb=(10, 10, 10), minDiv=True, ubm=None,
                        batch_size=100, numThread=1, save_partial=False):
        """        
        :param rank_F: rank of the between class variability matrix
        :param rank_G: rank of the within class variability matrix
        :param rank_H: boolean, if True, estimate the residual covariance 
            matrix. Default is False
        :param re_estimate_residual: boolean, if True, the residual covariance matrix is re-estimated (use for PLDA)
        :param itNb: tupple of three integers; number of iterations to run
            for F, G, H estimation
        :param minDiv: boolean, if True, re-estimate the covariance matrices 
            according to the minimum divergence criteria
        :param batch_size: number of sessions to process in one batch or memory optimization
        :param numThread: number of thread to run in parallel
        :param ubm: origin of the space; should be None for PLDA and be a 
            Mixture object for JFA or TV
        :param save_partial: name of the file to save intermediate models,
               if True, save before each split of the distributions
        
        :return: three matrices, the between class factor loading matrix,
            the within class factor loading matrix the diagonal MAP matrix 
            (as a vector) and the residual covariance matrix
        """

        """ not true anymore, stats are not whiten"""
        # Whiten the statistics around the UBM.mean or, 
        # if there is no UBM, around the effective mean

        vect_size = self.stat1.shape[1]
        print("vect_size = {}".format(vect_size))
        if ubm is None:
            mean = self.stat1.mean(axis=0)
            Sigma_obs = self.get_total_covariance_stat1()
            invSigma_obs = scipy.linalg.inv(Sigma_obs)
            evals, evecs = scipy.linalg.eigh(Sigma_obs)
            idx = numpy.argsort(evals)[::-1]
            evecs = evecs[:,idx]
            F_init = evecs[:, :rank_F]

        else:
            mean = ubm.get_mean_super_vector()
            invSigma_obs = ubm.get_invcov_super_vector()   
            Sigma_obs = 1./invSigma_obs 
            F_init = numpy.random.randn(vect_size, rank_F).astype(dtype=STAT_TYPE)

        G_init = numpy.random.randn(vect_size, rank_G)
        # rank_H = 0
        if rank_H is not None:  # H is empty or full-rank
            rank_H = vect_size
        else:
            rank_H = 0
        H_init = numpy.random.randn(rank_H).astype(dtype=STAT_TYPE) * Sigma_obs.mean()

        # Estimate the between class variability matrix
        if rank_F == 0:
            F = F_init
        else:
            # Modify the StatServer for the Total Variability estimation
            # each session is considered a class.
            if rank_G == rank_H == 0 and not re_estimate_residual:
                modelset_backup = copy.deepcopy(self.modelset)
                self.modelset = self.segset            
            
            F, Sigma = self.estimate_between_class(itNb[0],
                                                   F_init,
                                                   mean,
                                                   Sigma_obs,
                                                   batch_size,
                                                   None,
                                                   None,
                                                   minDiv,
                                                   numThread,
                                                   re_estimate_residual,
                                                   save_partial)

            if rank_G == rank_H == 0 and not re_estimate_residual:
                            self.modelset = modelset_backup

        # Estimate the within class variability matrix
        if rank_G == 0:
            G = G_init
        else:
            # Estimate Vy per model (not per session)
            Gtmp = numpy.random.randn(vect_size, 0)
            model_shifted_stat = self.sum_stat_per_model()[0]
            y, x, z = model_shifted_stat.estimate_hidden(mean, Sigma_obs, 
                                                         F, Gtmp, None, 
                                                         numThread)
                        
            """ Here we compute Vy for each  session so we duplicate first 
            the Y computed per model for each session corresponding to 
            this model and then multiply by V.
            We subtract then a weighted version of Vy from the statistics."""
            duplicate_y = numpy.zeros((self.modelset.shape[0], rank_F), dtype=STAT_TYPE)
            for idx, mod in enumerate(y.modelset):
                duplicate_y[self.modelset == mod] = y.stat1[idx]
            Vy = copy.deepcopy(self)
            Vy.stat1 = duplicate_y.dot(F.T)

            # Estimate G
            G = self.estimate_within_class(itNb[1],
                                           G_init,
                                           mean,
                                           Sigma_obs,
                                           batch_size,
                                           Vy,
                                           None,
                                           minDiv,
                                           numThread,
                                           save_partial)

        # Estimate the MAP covariance matrix
        if rank_H == 0:
            H = H_init
        else:
            # Estimate Vy per model (not per session)
            empty = numpy.random.randn(vect_size, 0)
            tmp_stat = self.sum_stat_per_model()[0]
            y, x, z = tmp_stat.estimate_hidden(mean, Sigma_obs, F, empty, None, numThread)
                        
            """ Here we compute Vy for each  session so we duplicate first 
            the Y computed per model for each session corresponding to 
            this model and then multiply by V.
            We subtract then a weighted version of Vy from the statistics."""
            duplicate_y = numpy.zeros((self.modelset.shape[0], rank_F), dtype=STAT_TYPE)
            for idx, mod in enumerate(y.modelset):
                duplicate_y[self.modelset == mod] = y.stat1[idx]
            Vy = copy.deepcopy(self)
            Vy.stat1 = duplicate_y.dot(F.T)
            
            # Estimate Ux per session
            tmp_stat = copy.deepcopy(self)
            tmp_stat = tmp_stat.subtract_weighted_stat1(Vy)
            y, x, z = tmp_stat.estimate_hidden(mean, Sigma_obs, empty, G, None, numThread)
            
            Ux = copy.deepcopy(self)
            Ux.stat1 = x.stat1.dot(G.T)
            
            # Estimate H
            H = self.estimate_map(itNb[2], H_init, 
                                  mean,
                                  Sigma_obs,
                                  Vy,
                                  Ux,
                                  save_partial)

        return mean, F, G, H, Sigma
