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
Copyright 2014-2016 Sylvain Meignier and Anthony Larcher

    :mod:`features_server` provides methods to manage features

"""
import h5py
import logging
import numpy
import os

from sidekit import PARAM_TYPE
from sidekit.frontend.features import mfcc
from sidekit.frontend.io import read_audio, read_label, write_hdf5
from sidekit.frontend.vad import vad_snr, vad_energy, vad_percentil
from sidekit.sidekit_wrappers import process_parallel_lists
from sidekit.bosaris.idmap import IdMap


__license__ = "LGPL"
__author__ = "Anthony Larcher & Sylvain Meignier"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


class FeaturesExtractor(object):
    """
    Charge un fichier audio (SPH, WAVE, RAW PCM)
    Extrait 1 unique canal
    Retourne un tuple contenant:
        (VAD, FB, CEPS, BNF)
         selon les options choisies
    """

    def __init__(self,
                 audio_filename_structure=None,
                 feature_filename_structure=None,
                 sampling_frequency=None,
                 lower_frequency=None,
                 higher_frequency=None,
                 filter_bank=None,
                 filter_bank_size=None,
                 window_size=None,
                 shift=None,
                 ceps_number=None,
                 vad=None,
                 snr=None,
                 pre_emphasis=None,
                 save_param=None,
                 keep_all_features=None,
                 single_channel_extension=None,
                 double_channel_extension=None):
        """

        :param audio_filename_structure:
        :param feature_filename_structure:
        :param sampling_frequency: optional, if processing RAW PCM
        :param lower_frequency:
        :param higher_frequency:
        :param filter_bank: type of fiter scale to use, can be lin or log (for linear of log-scale)
        :param filter_bank_size: number of filters bands
        :param window_size:
        :param shift:
        :param ceps_number:
        :param vad:
        :param snr:
        :param pre_emphasis:
        :param save_param: list of strings that indicate which parameters to save. The strings can be:
        "cep" for cepstral coefficients, "fb" for filter-banks, "energy" for the log-energy, "bnf"
        for bottle-neck features and "vad" for the frame selection labels. In the resuulting files, parameters are
         always concatenated in the following order: (energy,fb, cep, bnf, vad_label)
        :param keep_all_features:
        :param single_channel_extension:
        :param double_channel_extension:
        """

        # Set the default values
        self.audio_filename_structure = None
        self.feature_filename_structure = '{}'
        self.sampling_frequency = 8000
        self.lower_frequency = None
        self.higher_frequency = None
        self.filter_bank = None
        self.filter_bank_size = None
        self.window_size = None
        self.shift = None
        self.ceps_number = None
        self.vad = None
        self.snr = None
        self.pre_emphasis = 0.97
        self.save_param = ["energy", "cep", "fb", "bnf", "vad"]
        self.keep_all_features = None
        self.single_channel_extension = ''
        self.double_channel_extension = ('_a', '_b')

        if audio_filename_structure is not None:
            self.audio_filename_structure = audio_filename_structure
        if feature_filename_structure is not None:
            self.feature_filename_structure = feature_filename_structure
        if sampling_frequency is not None:
            self.sampling_frequency = sampling_frequency
        if lower_frequency is not None:
            self.lower_frequency = lower_frequency
        if higher_frequency is not None:
            self.higher_frequency = higher_frequency
        if filter_bank is not None:
            self.filter_bank = filter_bank
        if filter_bank_size is not None:
            self.filter_bank_size = filter_bank_size
        if window_size is not None:
            self.window_size = window_size
        if shift is not None:
            self.shift = shift
        if ceps_number is not None:
            self.ceps_number = ceps_number
        if vad is not None:
            self.vad = vad
        if snr is not None:
            self.snr = snr
        if pre_emphasis is not None:
            self.pre_emphasis = pre_emphasis
        if save_param is not None:
            self.save_param = save_param
        if keep_all_features is not None:
            self.keep_all_features = keep_all_features
        if single_channel_extension is not None:
            self.single_channel_extension = single_channel_extension
        if double_channel_extension is not None:
            self.double_channel_extension = double_channel_extension

        self.window_sample = None
        if not (self.window_size is None or self.sampling_frequency is None):
            self.window_sample = int(self.window_size * self.sampling_frequency)

        self.shift_sample = None
        if not (self.shift is None or self.sampling_frequency is None):
            self.shift_sample = int(self.shift * self.sampling_frequency)

        self.show = 'empty'

    def __repr__(self):
        ch = '\t show: {} keep_all_features: {}\n'.format(
            self.show, self.keep_all_features)
        ch += '\t audio_filename_structure: {}  \n'.format(self.audio_filename_structure)
        ch += '\t feature_filename_structure: {}  \n'.format(self.feature_filename_structure)
        ch += '\t pre-emphasis: {} \n'.format(self.pre_emphasis)
        ch += '\t lower_frequency: {}  higher_frequency: {} \n'.format(
            self.lower_frequency, self.higher_frequency)
        ch += '\t sampling_frequency: {} \n'.format(self.sampling_frequency)
        ch += '\t filter bank: {} filters of type {}\n'.format(
            self.filter_bank_size, self.filter_bank)
        ch += '\t ceps_number: {} \n\t window_size: {} shift: {} \n'.format(
            self.ceps_number, self.window_size, self.shift)
        ch += '\t vad: {}  snr: {} \n'.format(self.vad, self.snr)
        ch += '\t single channel extension: {} \n'.format(self.single_channel_extension)
        ch += '\t double channel extension: {} \n'.format(self.double_channel_extension)
        return ch

    def extract(self, show, channel, input_audio_filename=None, output_feature_filename=None, backing_store=False):
        """

        :param show:
        :param channel:
        :param input_audio_filename:
        :param output_feature_filename:
        :param backing_store:
        :return:
        """
        # Create the filename to load
        """
        Si le nom du fichier d'entrée est totalement indépendant du show
        -> si audio_filename_structure ne contient pas "{}"
        on peut mettre à jour: self.audio_filename_structure pour entrer directement le nom du fichier audio
        """
        if input_audio_filename is not None:
            self.audio_filename_structure = input_audio_filename
        """
        On met à jour l'audio_filename (que le show en fasse partie ou non)
        """
        audio_filename = self.audio_filename_structure.format(show)

        """
        Si le nom du fichier de sortie est totalement indépendant du show
        -> si feature_filename_structure ne contient pas "{}"
        on peut mettre à jour: self.audio_filename_structure pour entrer directement le nom du fichier de feature
        """
        if output_feature_filename is not None:
            self.feature_filename_structure = output_feature_filename
        """
        On met à jour le feature_filename (que le show en fasse partie ou non)
        """
        feature_filename = self.feature_filename_structure.format(show)
        # Open audio file, get the signal and possibly the sampling frequency
        signal, sample_rate = read_audio(audio_filename, self.sampling_frequency)
        if signal.ndim == 1:
            signal = signal[:, numpy.newaxis]

        # Process the target channel to return Filter-Banks, Cepstral coefficients and BNF if required
        length, chan = signal.shape

        # If the size of the signal is not enough for one frame, return zero features
        if length < self.window_sample:
            cep = numpy.empty((0, self.ceps_number), dtype=PARAM_TYPE)
            energy = numpy.empty((0, 1), dtype=PARAM_TYPE)
            fb = numpy.empty((0, self.filter_bank_size), dtype=PARAM_TYPE)
            label = numpy.empty((0, 1), dtype='int8')

        else:
            # Random noise is added to the input signal to avoid zero frames.
            numpy.random.seed(0)
            signal[:, channel] += 0.0001 * numpy.random.randn(signal.shape[0])

            dec = self.shift_sample * 250 * 25000 + self.window_sample
            dec2 = self.window_sample - self.shift_sample
            start = 0
            end = min(dec, length)

            # Process the signal by batch to avoid problems for very long signals
            while start < (length - dec2):
                logging.info('process part : %f %f %f',
                             start / self.sampling_frequency,
                             end / self.sampling_frequency,
                             length / self.sampling_frequency)

                # Extract cepstral coefficients, energy and filter banks
                cep, energy, _, fb = mfcc(signal[start:end, channel],
                                          fs=self.sampling_frequency,
                                          lowfreq=self.lower_frequency,
                                          maxfreq=self.higher_frequency,
                                          nlinfilt=self.filter_bank_size if self.filter_bank == "lin" else 0,
                                          nlogfilt=self.filter_bank_size if self.filter_bank == "log" else 0,
                                          nwin=self.window_size,
                                          nceps=self.ceps_number,
                                          get_spec=False,
                                          get_mspec=True,
                                          prefac=self.pre_emphasis)
                
                # Perform feature selection
                label, threshold = self._vad(cep, energy, fb, signal[start:end, channel])
                if len(label) < len(energy):
                    label = numpy.hstack((label, numpy.zeros(len(energy)-len(label), dtype='bool')))

                start = end - dec2
                end = min(end + dec, length)
                if cep.shape[0] > 0:
                    logging.info('!! size of signal cep: %f len %d type size %d', cep[-1].nbytes/1024/1024,
                                 len(cep[-1]),
                                 cep[-1].nbytes/len(cep[-1]))

        # Create the HDF5 file
        # Create the directory if it dosn't exist
        dir_name = os.path.dirname(feature_filename)  # get the path
        if not os.path.exists(dir_name) and (dir_name is not ''):
            os.makedirs(dir_name) 

        h5f = h5py.File(feature_filename, 'a', backing_store=backing_store, driver='core')
        if "cep" not in self.save_param:
            cep = None
        if "energy" not in self.save_param:
            energy = None
        if "fb" not in self.save_param:
            fb = None
        if "bnf" not in self.save_param:
            bnf = None
        if "vad" not in self.save_param:
            label = None
        logging.info(label)
       
        write_hdf5(show, h5f, cep, energy, fb, bnf, label)

        return h5f

    def save(self, show, channel=0, input_audio_filename=None, output_feature_filename=None):
        """

        :param show:
        :param channel:
        :param input_audio_filename:
        :param output_feature_filename:
        :return:
        """
        # Load the cepstral coefficients, energy, filter-banks, bnf and vad labels
        h5f = self.extract(show, channel, input_audio_filename, output_feature_filename, backing_store=True)
        logging.info(h5f.filename)

        # Write the hdf5 file to disk
        h5f.close()

    @staticmethod
    def _save(show, feature_filename_structure, save_param, cep, energy, fb, bnf, label):
        """

        :param show:
        :param feature_filename_structure:
        :param save_param:
        :param cep:
        :param energy:
        :param fb:
        :param bnf:
        :param label:
        :return:
        """
        feature_filename = feature_filename_structure.format(show)
        logging.info('output finename: '+feature_filename)
        dir_name = os.path.dirname(feature_filename)  # get the path
        if not os.path.exists(dir_name) and (dir_name is not ''):
            os.makedirs(dir_name)

        h5f = h5py.File(feature_filename, 'a', backing_store=True, driver='core')
        if "cep" not in save_param:
            cep = None
        if "energy" not in save_param:
            energy = None
        if "fb" not in save_param:
            fb = None
        if "bnf" not in save_param:
            bnf = None
        if "vad" not in save_param:
            label = None

        write_hdf5(show, h5f, cep, energy, fb, None, label)
        h5f.close()

    def save_multispeakers(self,
                           idmap,
                           channel=0,
                           input_audio_filename=None,
                           output_feature_filename=None,
                           keep_all=True):
        """
        :param idmap:
        :param channel:
        :param input_audio_filename:
        :param output_feature_filename:
        :param keep_all:
        :return:
        """
        param_vad = self.vad
        save_param = copy.deepcopy(self.save_param)
        self.save_param = ["energy", "cep", "fb", "bnf", "vad"]

        self.vad = None
        if output_feature_filename is None:
            output_feature_filename = self.feature_filename_structure

        tmp_dict = dict()
        nb = 0
        for show, id, start, stop in zip(idmap.rightids, idmap.leftids, idmap.start, idmap.stop):
            if show not in tmp_dict:
                tmp_dict[show] = dict()
            if id not in tmp_dict[show]:
                tmp_dict[show][id] = numpy.arange(start, stop-1)
                nb += 1
            else:
                tmp_dict[show][id] = numpy.concatenate((tmp_dict[show][id], numpy.arange(start, stop-1)), axis=0)

        output_show = list()
        output_id = list()
        output_start = list()
        output_stop = list()
        for show in tmp_dict:
            # temp_file_name = tempfile.NamedTemporaryFile().name
            # logging.info('tmp file name: '+temp_file_name)
            self.vad = None
            h5f = self.extract(show, channel, input_audio_filename, backing_store=False)
            energy = h5f.get(show + '/energy').value
            label = h5f.get(show + '/vad').value
            fb = h5f.get(show + '/fb').value
            cep = h5f.get(show + '/cep').value
            h5f.close()
            self.vad = param_vad

            for id in tmp_dict[show]:
                idx = tmp_dict[show][id]
                _, threshold_id = self._vad(None, energy[idx], None, None)
                logging.info('show: ' + show + ' cluster: ' + id + ' thr:' + str(threshold_id))
                label_id = energy > threshold_id
                label[idx] = label_id[idx]

                if not keep_all:
                    output_show.append(show+'/'+id)
                    output_id.append(id)
                    output_start.append(0)
                    output_stop.append(idx.shape[0])
                    logging.info('keep_all id: '+show+ ' show: '+show+'/'+id+' start: 0 stop: '+str(idx.shape[0]))
                    self._save(show+'/'+id,
                               output_feature_filename,
                               save_param, cep[idx],
                               energy[idx],
                               fb[idx],
                               None,
                               label[idx])

            if keep_all:
                self._save(show, output_feature_filename, save_param, cep, energy, fb, None, label)

        self.vad = param_vad
        self.save_param = save_param

        if keep_all:
            return copy.deepcopy(idmap)
        out_idmap = IdMap()
        out_idmap.set(numpy.array(output_id),
                      numpy.array(output_show),
                      start=numpy.array(output_start, dtype='int32'),
                      stop=numpy.array(output_stop, dtype='int32'))
        return out_idmap

    def _vad(self, cep, log_energy, fb, x, label_file_name=None):
        """
        Apply Voice Activity Detection.
        :param cep:
        :param log_energy:
        :param fb:
        :param x:
        :return:
        """
        threshold = -numpy.inf
        label = None
        if self.vad is None:
            logging.info('no vad')
            label = numpy.array([True] * log_energy.shape[0])
        elif self.vad == 'snr':
            logging.info('vad : snr')
            window_sample = int(self.window_size * self.sampling_frequency)
            label = vad_snr(x, self.snr, fs=self.sampling_frequency,
                            shift=self.shift, nwin=window_sample)
        elif self.vad == 'energy':
            logging.info('vad : energy')
            label = vad_energy(log_energy, distribNb=3,
                               nbTrainIt=8, flooring=0.0001,
                               ceiling=1.5, alpha=0.1)
        elif self.vad == 'percentil':
            label, threshold = vad_percentil(log_energy, 10)
            logging.info('percentil '+str(threshold))
        elif self.vad == 'dnn':
            pass  # TO DO
        elif self.vad == 'lbl':  # load existing labels as reference
            logging.info('vad : lbl')
            label = read_label(label_file_name)
        else:
            logging.warning('Wrong VAD type')
        return label, threshold

    @process_parallel_lists
    def save_list(self,
                  show_list,
                  channel_list,
                  audio_file_list=None,
                  feature_file_list=None,
                  num_thread=1):
        """
        :param show_list:
        :param channel_list:
        :param audio_file_list:
        :param feature_file_list:
        :param numThread: number of parallel process to run
        :return:
        """
        logging.info(self)

        # get the length of the longest list
        max_length = max([len(l) for l in [show_list, channel_list, audio_file_list, feature_file_list]
                          if l is not None])

        if show_list is None:
            show_list = numpy.empty(max_length, dtype='|O')
        if audio_file_list is None:
            audio_file_list = numpy.empty(max_length, dtype='|O')
        if feature_file_list is None:
            feature_file_list = numpy.empty(max_length, dtype='|O')

        for show, channel, audio_file, feature_file in zip(show_list, channel_list, audio_file_list, feature_file_list):
            self.save(show, channel, audio_file, feature_file)
