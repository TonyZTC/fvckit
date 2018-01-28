# -*- coding: utf-8 -*-
#
# This file is part of FVCKIT.
#
# FVCKIT is a python package for forensic voice comparison.
# It was forked from SIDEKIT (version 1.2.3), a python package for 
# speaker verification.
# Home page: https://github.com/entn-at/fvckit
#
# FVCKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# FVCKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FVCKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2017 Anthony Larcher and Sylvain Meignier and Andreas Nautsch (SIDEKIT)
Copyright 2018      Ewald Enzinger (FVCKIT)
"""

from ctypes import *
from ctypes.util import find_library
import logging
import numpy
import os
import sys
import importlib


# Read environment variable if it exists
FVCKIT_CONFIG={"mpi":False}

if 'FVCKIT' in os.environ:
    for cfg in os.environ['FVCKIT'].split(","):
        k, val = cfg.split("=")
        if k == "mpi":
            if val == "true":
               FVCKIT_CONFIG["mpi"] = True 

PARALLEL_MODULE = 'multiprocessing'  # can be , threading, multiprocessing MPI is planned in the future
PARAM_TYPE = numpy.float32
STAT_TYPE = numpy.float64  # can be numpy.float32 to speed up the computation but can lead to numerical issuess

# Import bosaris-like classes
from fvckit.bosaris import IdMap
from fvckit.bosaris import Ndx
from fvckit.bosaris import Key
from fvckit.bosaris import Scores
from fvckit.bosaris import SegQuality
from fvckit.bosaris import DetPlot
from fvckit.bosaris import effective_prior
from fvckit.bosaris import logit_effective_prior
from fvckit.bosaris import fast_minDCF

# Import classes
from fvckit.features_extractor import FeaturesExtractor
from fvckit.features_server import FeaturesServer
from fvckit.mixture import Mixture
from fvckit.statserver import StatServer
from fvckit.factor_analyser import FactorAnalyser

from fvckit.frontend.io import write_pcm
from fvckit.frontend.io import read_pcm
from fvckit.frontend.io import pcmu2lin
from fvckit.frontend.io import read_sph
from fvckit.frontend.io import write_label
from fvckit.frontend.io import read_label
from fvckit.frontend.io import read_spro4
from fvckit.frontend.io import read_audio
from fvckit.frontend.io import write_spro4
from fvckit.frontend.io import read_htk
from fvckit.frontend.io import write_htk

from fvckit.frontend.vad import vad_energy
from fvckit.frontend.vad import vad_snr
from fvckit.frontend.vad import label_fusion
from fvckit.frontend.vad import speech_enhancement


from fvckit.frontend.normfeat import cms
from fvckit.frontend.normfeat import cmvn
from fvckit.frontend.normfeat import stg
from fvckit.frontend.normfeat import rasta_filt


from fvckit.frontend.features import compute_delta
from fvckit.frontend.features import framing
from fvckit.frontend.features import pre_emphasis
from fvckit.frontend.features import trfbank
from fvckit.frontend.features import mel_filter_bank
from fvckit.frontend.features import mfcc
from fvckit.frontend.features import pca_dct
from fvckit.frontend.features import shifted_delta_cepstral

from fvckit.iv_scoring import cosine_scoring
from fvckit.iv_scoring import mahalanobis_scoring
from fvckit.iv_scoring import two_covariance_scoring
from fvckit.iv_scoring import PLDA_scoring

from fvckit.gmm_scoring import gmm_scoring 

from fvckit.jfa_scoring import jfa_scoring


from fvckit.sv_utils import clean_stat_server

if FVCKIT_CONFIG["mpi"]:
    found_mpi4py = importlib.find_loader('mpi4py') is not None
    if found_mpi4py:
        from fvckit.fvckit_mpi import EM_split, total_variability, extract_ivector
        print("Import MPI")
        

__author__ = "Anthony Larcher and Sylvain Meignier (SIDEKIT), Ewald Enzinger (FVCKIT)"
__copyright__ = "Copyright 2014-2017 Anthony Larcher and Sylvain Meignier (SIDEKIT), 2018 Ewald Enzinger (FVCKIT)"
__license__ = "LGPL"
__maintainer__ = "Ewald Enzinger"
__email__ = "ewald.enzinger@entn.at"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__version__="1.2.4"

# __all__ = ["io",
#            "vad",
#            "normfeat",
#            "features"
#            ]
