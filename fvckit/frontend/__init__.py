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
Copyright 2014-2017 Sylvain Meignier and Anthony Larcher (SIDEKIT)
          2018      Ewald Enzinger (FVCKIT)

:mod:`frontend` provides methods to process an audio signal in order to extract
useful parameters for speaker verification.
"""

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
from fvckit.frontend.normfeat import cep_sliding_norm


from fvckit.frontend.features import compute_delta
from fvckit.frontend.features import framing
from fvckit.frontend.features import pre_emphasis
from fvckit.frontend.features import trfbank
from fvckit.frontend.features import mel_filter_bank
from fvckit.frontend.features import mfcc
from fvckit.frontend.features import pca_dct
from fvckit.frontend.features import shifted_delta_cepstral

__author__ = "Anthony Larcher and Sylvain Meignier (SIDEKIT), Ewald Enzinger (FVCKIT)"
__copyright__ = "Copyright 2014-2017 Anthony Larcher and Sylvain Meignier (SIDEKIT), 2018 Ewald Enzinger (FVCKIT)"
__license__ = "LGPL"
__maintainer__ = "Ewald Enzinger"
__email__ = "ewald.enzinger@entn.at"
__status__ = "Production"
__docformat__ = 'reStructuredText'
