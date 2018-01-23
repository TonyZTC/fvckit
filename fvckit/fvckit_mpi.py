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

    :mod:`fvckit_mpi` provides methods to run using Message Passing interface

"""

import copy
import numpy
import os
import logging
import h5py
import scipy
import sys
from fvckit.features_server import FeaturesServer
from fvckit.statserver import StatServer, sum_log_probabilities
from fvckit.factor_analyser import FactorAnalyser
from fvckit.mixture import Mixture
from fvckit.fvckit_io import write_matrix_hdf5, read_matrix_hdf5
from fvckit import IdMap, Ndx, Key, STAT_TYPE
from fvckit.sv_utils import serialize
from fvckit.factor_analyser import e_on_batch, e_on_batch_log_evidence
from mpi4py import MPI
from psutil import virtual_memory
from scipy.spatial import distance
from scipy.special import erfcinv, gamma
import time


__license__ = "LGPL"
__author__ = "Anthony Larcher, Sylvain Meignier, Andreas Nautsch (SIDEIT), Ewald Enzinger (FVCKIT)"
__copyright__ = "Copyright 2014-2017 Anthony Larcher, Sylvain Meignier, Andreas Nautsch (SIDEKIT), 2018 Ewald Enzinger (FVCKIT)"
__maintainer__ = "Ewald Enzinger"
__email__ = "ewald.enzinger@entn.at"
__status__ = "Production"
__docformat__ = 'reStructuredText'

data_type = numpy.float32

def total_variability(stat_server_file_name,
                      ubm,
                      tv_rank,
                      start_iter=0,
                      nb_iter=20,
                      min_div=True,
                      tv_init=None,
                      save_init=False,
                      output_file_name=None):
    """
    Train a total variability model using multiple process on multiple nodes with MPI.

    Example of how to train a total variability matrix using MPI.
    Here is what your script should look like:

    ----------------------------------------------------------------

    import fvckit

    fa = fvckit.FactorAnalyser()
    fa.total_variability_mpi("/lium/spk1/larcher/expe/MPI_TV/data/statserver.h5",
                             ubm,
                             tv_rank,
                             nb_iter=tv_iteration,
                             min_div=True,
                             tv_init=tv_new_init2,
                             output_file_name="data/TV_mpi")

    ----------------------------------------------------------------

    This script should be run using mpirun command (see MPI4PY website for
    more information about how to use it
        http://pythonhosted.org/mpi4py/
    )

        mpirun --hostfile hostfile ./my_script.py

    :param comm: MPI.comm object defining the group of nodes to use
    :param stat_server_file_name: name of the StatServer file to load (make sure you provide absolute path and that
    it is accessible from all your nodes).
    :param ubm: a Mixture object
    :param tv_rank: rank of the total variability model
    :param nb_iter: number of EM iteration
    :param min_div: boolean, if True, apply minimum divergence re-estimation
    :param tv_init: initial matrix to start the EM iterations with
    :param output_file_name: name of the file where to save the matrix
    """
    comm = MPI.COMM_WORLD

    comm.Barrier()

    # this lines allows to process a single StatServer or a list of StatServers
    if not isinstance(stat_server_file_name, list):
        stat_server_file_name = [stat_server_file_name]

    # Initialize useful variables
    sv_size = ubm.get_mean_super_vector().shape[0]
    gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"
    nb_distrib, feature_size = ubm.mu.shape
    upper_triangle_indices = numpy.triu_indices(tv_rank)

    # Initialize the FactorAnalyser, mean and Sigma are initialized at ZEROS as statistics are centered
    factor_analyser = FactorAnalyser()
    factor_analyser.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
    factor_analyser.F = serialize(numpy.zeros((sv_size, tv_rank)).astype(data_type))
    if tv_init is None:
        factor_analyser.F = numpy.random.randn(sv_size, tv_rank).astype(data_type)
    else:
        factor_analyser.F = tv_init
    factor_analyser.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

    # Save init if required
    if comm.rank == 0:
        if output_file_name is None:
            output_file_name = "temporary_factor_analyser"
        if save_init:
            factor_analyser.write(output_file_name + "_init.h5")

    # Iterative training of the FactorAnalyser
    if start_iter != 0:
        start_iter = start_iter + 1

    for it in range(start_iter, nb_iter):
        if comm.rank == 0:
            logging.critical("Start it {}".format(it))

        _A = numpy.zeros((nb_distrib, tv_rank * (tv_rank + 1) // 2), dtype=data_type)
        _C = numpy.zeros((tv_rank, sv_size), dtype=data_type)
        _R = numpy.zeros((tv_rank * (tv_rank + 1) // 2), dtype=data_type)
        log_evidence_local = 0

        if comm.rank == 0:
            total_session_nb = 0

        # E-step
        for stat_server_file in stat_server_file_name:

            with h5py.File(stat_server_file, 'r') as fh:
                nb_sessions = fh["segset"].shape[0]

                if comm.rank == 0:
                    total_session_nb += nb_sessions

                comm.Barrier()
                if comm.rank == 0:
                    logging.critical("Process file: {}".format(stat_server_file))

                # Allocate a list of sessions to process to each node
                local_session_idx = numpy.array_split(range(nb_sessions), comm.size)
                stat0 = fh['stat0'][local_session_idx[comm.rank], :]
                stat1 = fh['stat1'][local_session_idx[comm.rank], :]
                e_h, e_hh, logE = e_on_batch_log_evidence(stat0, stat1, ubm, factor_analyser.F)

                if numpy.isnan(logE):
                    logging.critical('nan log evidence on comm {}'.format(comm.rank))

                log_evidence_local += logE / nb_sessions / comm.size

                _A += stat0.T.dot(e_hh)
                _C += e_h.T.dot(stat1)
                _R += numpy.sum(e_hh, axis=0)
 
            comm.Barrier()

        comm.Barrier()

        # Sum all statistics
        if comm.rank == 0:
            # only processor 0 will actually get the data
            total_A = numpy.zeros_like(_A)
            total_C = numpy.zeros_like(_C)
            total_R = numpy.zeros_like(_R)
            log_evidence = numpy.zeros(1)
        else:
            total_A = [None] * _A.shape[0]
            total_C = None
            total_R = None
            log_evidence = None

        # Accumulate _A, using a list in order to avoid limitations of MPI (impossible to reduce matrices bigger
        # than 4GB)
        for ii in range(_A.shape[0]):
            _tmp = copy.deepcopy(_A[ii])
            if comm.rank == 0:
                _total_A = numpy.zeros_like(total_A[ii])
            else:
                _total_A = None

            comm.Reduce(
                [_tmp, MPI.FLOAT],
                [_total_A, MPI.FLOAT],
                op=MPI.SUM,
                root=0
            )
            if comm.rank == 0:
                total_A[ii] = copy.deepcopy(_total_A)

        comm.Reduce(
            [_C, MPI.FLOAT],
            [total_C, MPI.FLOAT],
            op=MPI.SUM,
            root=0
        )

        comm.Reduce(
            [_R, MPI.FLOAT],
            [total_R, MPI.FLOAT],
            op=MPI.SUM,
            root=0
        )

        comm.Reduce(
            [log_evidence_local, MPI.FLOAT],
            [log_evidence, MPI.FLOAT],
            op=MPI.SUM,
            root=0
        )

        comm.Barrier()
           
        # M-step
        if comm.rank == 0:
            logging.info('log-evidence: {}'.format(log_evidence[0]))

            total_R /= total_session_nb

            if logging.getLogger().level == logging.DEBUG:
                # in case of debugging, save matrixes
                write_matrix_hdf5(total_A, output_file_name + '_total_A_it-{}.h5'.format(it))
                write_matrix_hdf5(total_C, output_file_name + '_total_C_it-{}.h5'.format(it))
                write_matrix_hdf5(total_R, output_file_name + '_total_R_it-{}.h5'.format(it))

            _A_tmp = numpy.zeros((tv_rank, tv_rank), dtype=data_type)
            for c in range(nb_distrib):
                distrib_idx = range(c * feature_size, (c + 1) * feature_size)
                _A_tmp[upper_triangle_indices] = _A_tmp.T[upper_triangle_indices] = total_A[c, :]
                _A_tmp_cond = numpy.linalg.cond(_A_tmp)
                if _A_tmp_cond > 1e6:
                    logging.debug('large condition number {} found at component {}'.format(_A_tmp_cond, c))
                factor_analyser.F[distrib_idx, :] = scipy.linalg.solve(_A_tmp, total_C[:, distrib_idx]).T

            logging.debug('max(abs(TV)) {}'.format(numpy.abs(factor_analyser.F).max()))

            # minimum divergence
            if min_div:
                _R_tmp = numpy.zeros((tv_rank, tv_rank), dtype=data_type)
                _R_tmp[upper_triangle_indices] = _R_tmp.T[upper_triangle_indices] = total_R
                # if not positive definite, fix by minimum impact to diagonal
                _R_tmp_min_eigval = numpy.linalg.eigvals(_R_tmp).min()
                if _R_tmp_min_eigval <= 0:
                    while _R_tmp_min_eigval <= 0:
                        logging.debug('ill-conditioned _R_tmp, not positive-definite, add {} to diagonal'.format(_R_tmp_min_eigval))
                        _R_tmp = _R_tmp - _R_tmp_min_eigval * numpy.eye(tv_rank, tv_rank)
                        _R_tmp_min_eigval = numpy.linalg.eigvals(_R_tmp).min()
                ch = scipy.linalg.cholesky(_R_tmp).T
                factor_analyser.F = factor_analyser.F.dot(ch)
                logging.debug('max(abs(TV)) after min. div. {}'.format(numpy.abs(factor_analyser.F).max()))

            # Save the current FactorAnalyser
            if output_file_name is not None:
                if it < nb_iter - 1:
                    factor_analyser.write(output_file_name + "_it-{}.h5".format(it))
                else:
                    factor_analyser.write(output_file_name + ".h5")
        factor_analyser.F = comm.bcast(factor_analyser.F, root=0)
        comm.Barrier()


def accumulate_statistics(idmap, feature_server, ubm, statserver_file_name, channel_extension=("", "_b")):
    assert (isinstance(feature_server, FeaturesServer))
    assert (isinstance(ubm, Mixture))

    comm = MPI.COMM_WORLD
    comm.Barrier()

    if not os.path.exists(statserver_file_name):
        if isinstance(idmap, Ndx) or isinstance(idmap, Key):
            idmap_tmp = IdMap()
            idmap_tmp.leftids = idmap.segset
            idmap_tmp.rightids = idmap.segset
            idmap_tmp.start = numpy.empty(idmap.segset.shape[0], dtype="|O")
            idmap_tmp.stop = numpy.empty(idmap.segset.shape[0], dtype="|O")
            assert (idmap_tmp.validate())
            idmap = idmap_tmp
        assert (isinstance(idmap, IdMap))
        feature_server.keep_all_features = True
        comm.Barrier()

        if comm.rank == 0:
            # create temporary statsserver file
            tmp_stats_file = '{}_empty_tmp'.format(statserver_file_name)
            stat_server = StatServer(statserver_file_name=idmap,
                                     distrib_nb=ubm.get_distrib_nb(),
                                     feature_size=ubm.dim(),
                                     index=None)
            stat_server.write(tmp_stats_file)
            logging.debug('accumulating stats for: {} - {}'.format(stat_server.segset.shape, stat_server.segset))
        else:
            tmp_stats_file = None
        tmp_stats_file = comm.bcast(tmp_stats_file, root=0)
        comm.Barrier()

        # Set useful variables
        nb_distrib = ubm.w.shape[0]
        feature_size = ubm.mu.shape[1]
        stat0_rank = nb_distrib
        stat1_rank = nb_distrib * feature_size

        # Work on each node with different data
        nb_sessions = idmap.leftids.shape[0]
        # max. 2^32 - 1 stat1 values to gather back due to mpi4py
        max_nb_sessions_per_cycle = int(numpy.floor((2 ** 31 - 1) / ubm.sv_size()))
        nb_cycles = int(numpy.ceil(nb_sessions / max_nb_sessions_per_cycle)) + 1 # +1 is merely assuring in-memory
        all_session_idx = numpy.array_split(numpy.arange(nb_sessions), nb_cycles, axis=0)
        for cycle_idx in range(nb_cycles):
            indices = numpy.array_split(all_session_idx[cycle_idx], comm.size, axis=0)

            sendcounts_stat0 = numpy.array([idx.shape[0] * stat0_rank for idx in indices])
            displacements_stat0 = numpy.hstack((0, numpy.cumsum(sendcounts_stat0)[:-1]))
            sendcounts_stat1 = numpy.array([idx.shape[0] * stat1_rank for idx in indices])
            displacements_stat1 = numpy.hstack((0, numpy.cumsum(sendcounts_stat1)[:-1]))

            time.sleep(comm.rank * 5) # give 5s for each cpu to read data, as memory in subroutine can get larger
            stat_server = StatServer.read_subset(tmp_stats_file, indices[comm.rank])
            comm.Barrier()

            # accumulate stats
            if comm.rank == 0:
                stat0 = numpy.zeros((all_session_idx[cycle_idx].shape[0], stat0_rank))
                stat1 = numpy.zeros((all_session_idx[cycle_idx].shape[0], stat1_rank))
            else:
                stat0 = None
                stat1 = None

            local_stat0 = numpy.zeros((stat_server.modelset.shape[0], stat0_rank))
            local_stat1 = numpy.zeros((stat_server.modelset.shape[0], stat1_rank))

            # Replicate stat0
            index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)
            for idx in range(stat_server.segset.shape[0]):
                logging.debug('Compute statistics for {}'.format(stat_server.segset[idx]))

                show = stat_server.segset[idx]

                # If using a FeaturesExtractor, get the channel number by checking the extension of the show
                channel = 0
                if feature_server.features_extractor is not None and show.endswith(channel_extension[1]):
                    channel = 1
                show = show[:show.rfind(channel_extension[channel])]

                cep, vad = feature_server.load(show, channel=channel)
                stop = vad.shape[0] if stat_server.stop[idx] is None else min(stat_server.stop[idx], vad.shape[0])
                data = cep[stat_server.start[idx]:stop, :]
                data = data[vad[stat_server.start[idx]:stop], :]

                # Verify that frame dimension is equal to gmm dimension
                if not ubm.dim() == data.shape[1]:
                    raise Exception('dimension of ubm and features differ: {:d} / {:d}'.format(ubm.dim(), data.shape[1]))
                else:
                    if ubm.invcov.ndim == 2:
                        lp = ubm.compute_log_posterior_probabilities(data)
                    else:
                        lp = ubm.compute_log_posterior_probabilities_full(data)
                    pp, _ = sum_log_probabilities(lp)
                    # Compute 0th-order statistics
                    local_stat0[idx, :] = pp.sum(0)
                    if (local_stat0[idx, :]==0).any():
                        logging.critical('unexpected zero probability encountered at: {}'.format(numpy.unique(numpy.where(local_stat0[idx, :]==0)[0])))
                    # Compute 1st-order statistics
                    local_stat1[idx, :] = numpy.reshape(numpy.transpose(numpy.dot(data.transpose(), pp)), ubm.sv_size()).astype(STAT_TYPE)
            comm.Barrier()

            comm.Gatherv(local_stat0, [stat0, sendcounts_stat0, displacements_stat0, MPI.DOUBLE], root=0)
            comm.Gatherv(local_stat1, [stat1, sendcounts_stat1, displacements_stat1, MPI.DOUBLE], root=0)

            if comm.rank == 0:
                stat_server = StatServer.read_subset(tmp_stats_file, all_session_idx[cycle_idx])
                """
                stat_server = StatServer(statserver_file_name=idmap.filter_on_left(stat_server.modelset[indices], keep=True),
                                         distrib_nb=ubm.get_distrib_nb(),
                                         feature_size=ubm.dim(),
                                         index=None)
                """
                stat_server.stat0 = stat0
                stat_server.stat1 = stat1
                logging.debug('saving sub stat0 {}, stat1 {}, statserver {}'.format(stat0.shape, stat1.shape, stat_server))
                stat_server.write('{}_sub-{}.h5'.format(statserver_file_name, cycle_idx))

        if comm.rank == 0:
            stat_server_lst = []
            for cycle_idx in range(nb_cycles):
                stat_server_lst.append(StatServer(statserver_file_name='{}_sub-{}.h5'.format(statserver_file_name, cycle_idx)))
            stat_server = StatServer.merge(*stat_server_lst)
            stat_server.write(statserver_file_name)
            os.remove(tmp_stats_file)


def extract_ivector(tv,
                    stat_server_file_name,
                    ubm,
                    output_file_name,
                    uncertainty=False,
                    prefix=''):
    """
    Estimate i-vectors for a given StatServer using multiple process on multiple nodes.

    :param comm: MPI.comm object defining the group of nodes to use
    :param stat_server_file_name: file name of the sufficient statistics StatServer HDF5 file
    :param ubm: Mixture object (the UBM)
    :param output_file_name: name of the file to save the i-vectors StatServer in HDF5 format
    :param uncertainty: boolean, if True, saves a matrix with uncertainty matrices (diagonal of the matrices)
    :param prefix: prefixe of the dataset to read from in HDF5 file
    """
    assert(isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"

    comm = MPI.COMM_WORLD

    comm.Barrier()

    gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

    # Set useful variables
    tv_rank = tv.F.shape[1]
    feature_size = ubm.mu.shape[1]
    nb_distrib = ubm.w.shape[0]

    # Get the number of sessions to process
    with h5py.File(stat_server_file_name, 'r') as fh:
        nb_sessions = fh["segset"].shape[0]

    # Work on each node with different data
    indices = numpy.array_split(numpy.arange(nb_sessions), comm.size, axis=0)
    sendcounts = numpy.array([idx.shape[0] * tv_rank  for idx in indices])
    displacements = numpy.hstack((0, numpy.cumsum(sendcounts)[:-1]))

    stat_server = StatServer.read_subset(stat_server_file_name, indices[comm.rank])

    # Whiten the statistics for diagonal or full models
    if gmm_covariance == "diag":
        stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
    elif gmm_covariance == "full":
        stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

    # Estimate i-vectors
    if comm.rank == 0:
        iv = numpy.zeros((nb_sessions, tv_rank))
        iv_sigma = numpy.zeros((nb_sessions, tv_rank))
    else:
        iv = None
        iv_sigma = None

    local_iv = numpy.zeros((stat_server.modelset.shape[0], tv_rank))
    local_iv_sigma = numpy.ones((stat_server.modelset.shape[0], tv_rank))

    # Replicate stat0
    index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)
    for sess in range(stat_server.segset.shape[0]):

         inv_lambda = scipy.linalg.inv(numpy.eye(tv_rank) + (tv.F.T * stat_server.stat0[sess, index_map]).dot(tv.F))

         Aux = tv.F.T.dot(stat_server.stat1[sess, :])
         local_iv[sess, :] = Aux.dot(inv_lambda)
         local_iv_sigma[sess, :] = numpy.diag(inv_lambda + numpy.outer(local_iv[sess, :], local_iv[sess, :]))
    comm.Barrier()

    comm.Gatherv(local_iv,[iv, sendcounts, displacements,MPI.DOUBLE], root=0)
    comm.Gatherv(local_iv_sigma,[iv_sigma, sendcounts, displacements,MPI.DOUBLE], root=0)

    if comm.rank == 0:

        with h5py.File(stat_server_file_name, 'r') as fh:
            iv_stat_server = StatServer()
            iv_stat_server.modelset = fh.get(prefix+"modelset").value
            iv_stat_server.segset = fh.get(prefix+"segset").value

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                iv_stat_server.modelset = iv_stat_server.modelset.astype('U', copy=False)
                iv_stat_server.segset = iv_stat_server.segset.astype('U', copy=False)

            tmpstart = fh.get(prefix+"start").value
            tmpstop = fh.get(prefix+"stop").value
            iv_stat_server.start = numpy.empty(fh[prefix+"start"].shape, '|O')
            iv_stat_server.stop = numpy.empty(fh[prefix+"stop"].shape, '|O')
            iv_stat_server.start[tmpstart != -1] = tmpstart[tmpstart != -1]
            iv_stat_server.stop[tmpstop != -1] = tmpstop[tmpstop != -1]
            iv_stat_server.stat0 = numpy.ones((nb_sessions, 1))
            iv_stat_server.stat1 = iv

        iv_stat_server.write(output_file_name)
        if uncertainty:
            path = os.path.splitext(output_file_name)
            write_matrix_hdf5(iv_sigma, path[0] + "_uncertainty" + path[1])


def EM_split(ubm,
             features_server,
             feature_list,
             distrib_nb,
             output_filename,
             iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8),
             llk_gain=0.01,
             save_partial=False,
             ceil_cov=10,
             floor_cov=1e-2,
             ifs_eu=None,
             num_thread=1):
    """Expectation-Maximization estimation of the Mixture parameters.

    :param comm:
    :param features: a 2D-array of feature frames (one raow = 1 frame)
    :param distrib_nb: final number of distributions
    :param iterations: list of iteration number for each step of the learning process
    :param llk_gain: limit of the training gain. Stop the training when gain between
            two iterations is less than this value
    :param save_partial: name of the file to save intermediate mixtures,
           if True, save before each split of the distributions
    :param ceil_cov:
    :param floor_cov:

    :return llk: a list of log-likelihoods obtained after each iteration
    """

    comm = MPI.COMM_WORLD
    comm.Barrier()

    if comm.rank == 0:
        import sys
        llk = []
        logging.debug('broadcasting data for parallel training, in total {} samples'.format(len(feature_list)))
    else:
        llk = None

    # Work on each node with different data
    indices = numpy.array_split(numpy.arange(len(feature_list)), comm.size, axis=0)

    comm.Barrier()
    if comm.rank == 0:
        logging.debug('data broadcasted, barrier established, iterations ...')

    # compute local features, conduct intelligent feature selection based on Euclidean distance (IFS-EU)
    # see T. Hasan and J.H.L. Hansen: A Study on Universal Background Model Training in Speaker Verification,
    # TASLP, 19(7), p. 1890-1899, 2011.
    local_show_list = list(numpy.array(feature_list)[indices[comm.rank]])
    logging.debug('{} comm selects features from {} samples'.format(comm.rank, len(local_show_list)))
    local_features = []
    if ifs_eu is not None:
        alpha_quantile = 0.1
        beta_mu = 0.8
        beta_std = 0.6
        """
        alpha_quantile = ifs_eu[0]
        beta_mu = ifs_eu[1]
        beta_std = ifs_eu[2]
        """
        alpha_erfcinv = erfcinv(2*alpha_quantile)
        for show in local_show_list:
            show_features = features_server.load(show)[0]
            k = show_features.shape[1]
            gamma_ratio_k = gamma((1+k)/2) / gamma(k/2)
            i = 0
            local_features.append(show_features[i,:])
            while i < show_features.shape[0]-1 - 1:
                # init
                mu_x = show_features[i:i+2,:].mean(axis=0)
                std_x = 0.5 * (show_features[i,:]**2 + show_features[i+1,:]**2) - mu_x**2
                for j in range(i+1, show_features.shape[0]):
                    avg_std_x = std_x.mean()
                    mu_d = 2 * numpy.sqrt(avg_std_x) * k * gamma_ratio_k
                    std_d = 2 * k * avg_std_x - mu_d**2
                    dth = mu_d + numpy.sqrt(2) * std_d * alpha_erfcinv

                    d = distance.euclidean(show_features[i,:], show_features[j,:])
                    if d > dth:
                        i = j
                        local_features.append(show_features[i,:])
                        break # start procedure all over again

                    mu_x = beta_mu * mu_x + (1-beta_mu) * show_features[j,:]
                    std_x = beta_std * std_x + (1-beta_std) * distance.sqeuclidean(show_features[j,:], mu_x)
                break # as j reached show_features.shape[0], and any i = j will cause a break, unavoidably
        local_features = numpy.stack(local_features)
        logging.debug('comm {} selected local features via IFS-EU, starting EM with {} frames'.format(comm.rank, local_features.shape))
    else:
        for show in local_show_list:
            show_features = features_server.load(show)[0]
            local_features.append(show_features)
        local_features = numpy.concatenate(local_features, axis=0)
        logging.debug('comm {} selected local features (all), starting EM with {} frames'.format(comm.rank, local_features.shape))

    # if UBM is empty, initialize with features
    if ubm.get_distrib_nb() == 0:
        local_num = numpy.array([local_features.shape[0]], dtype=STAT_TYPE)
        local_mu = local_features.mean(axis=0) * local_num

        if comm.rank == 0:
            features_num = numpy.zeros(1)
            features_mu = numpy.zeros(local_mu.shape[0])
            features_cov = numpy.zeros(local_mu.shape[0])
        else:
            features_num = None
            features_mu = None
            features_cov = None

        comm.Barrier()
        comm.Reduce(
            [local_num, MPI.DOUBLE],
            [features_num, MPI.DOUBLE],
            op=MPI.SUM,
            root=0
        )
        comm.Reduce(
            [local_mu, MPI.DOUBLE],
            [features_mu, MPI.DOUBLE],
            op=MPI.SUM,
            root=0
        )
        comm.Barrier()
        if comm.rank == 0:
            mu = features_mu / features_num[0]
        else:
            mu = None
        mu = comm.bcast(mu, root=0)
        comm.Barrier()
        logging.debug('bcast mu: {}'.format(mu.shape))

        local_cov = numpy.mean((local_features - mu) ** 2, axis=0) * local_num
        comm.Barrier()
        logging.debug('local cov: {}'.format(local_cov.shape))

        comm.Reduce(
            [local_cov, MPI.DOUBLE],
            [features_cov, MPI.DOUBLE],
            op=MPI.SUM,
            root=0
        )
        comm.Barrier()

        # Initialize the mixture for empty ubm, otherwise take ubm as basis
        if comm.rank == 0:
            cov = features_cov / features_num[0]
            logging.debug('ubm cov: {}'.format(cov.shape))
            ubm.mu = mu[None]
            ubm.invcov = 1. / cov[None]
            ubm.w = numpy.asarray([1.0])
            ubm.cst = numpy.zeros(ubm.w.shape)
            ubm.det = numpy.zeros(ubm.w.shape)
            ubm.cov_var_ctl = 1.0 / copy.deepcopy(ubm.invcov)
            ubm._compute_all()
        else:
            ubm = None
        # Broadcast the UBM on each process
        ubm = comm.bcast(ubm, root=0)
        comm.Barrier()

    # for N iterations:
    for nbg, it in enumerate(iterations[int(numpy.log2(ubm.get_distrib_nb())):int(numpy.log2(distrib_nb))]):

        if comm.rank == 0:
            logging.critical("Start training model with {} distributions".format(2**numpy.log2(ubm.get_distrib_nb())))
            # Save current model before spliting
            if save_partial:
                ubm.write(output_filename + '_{}g.h5'.format(ubm.get_distrib_nb()), prefix='')

        comm.Barrier()

        ubm._split_ditribution()
            
        if comm.rank == 0:
            accum = copy.deepcopy(ubm)
        else:
            accum = Mixture()
            accum.w = accum.mu = accum.invcov = None

        # Create one accumulator for each process
        local_accum = copy.deepcopy(ubm)
        for i in range(it):

            local_accum._reset()

            if comm.rank == 0:
                logging.critical("\titeration {} / {}".format(i+1, it))
                _tmp_llk = numpy.array(0)
                accum._reset()

            else:
                _tmp_llk = numpy.array([None])

            comm.Barrier()

            # E step
            logging.critical("Start E-step, rank {}".format(comm.rank))

            # use up to 80% of available bytes in memory of server running current MPI process
            assert(isinstance(local_features, numpy.ndarray))
            weight_memory_allocation = 4 # following mem estimate is too restrictive # 0.8 # (0, 1], e.g. 0.8 for 80%
            # considering: #features, #distr, #cep and covariance calculation of features
            max_features_per_accum = virtual_memory()._asdict()['available']\
                                     / local_features.dtype.itemsize\
                                     / (ubm.get_distrib_nb()**2)\
                                     / ubm.dim()\
                                     / comm.size\
                                     * weight_memory_allocation

            if max_features_per_accum <= 1.0:
                logging.critical("memory will run out, choose less #cores, e.g. {}".format(max_features_per_accum * comm.size))
                assert(max_features_per_accum > 1.0)
            max_features_per_accum = int(max_features_per_accum)

            comm.Barrier()

            if local_features.shape[0] > max_features_per_accum:
                local_llk = ubm._expectation(local_accum, local_features[:max_features_per_accum, :])
                num_accums = int(local_features.shape[0] / max_features_per_accum) + 1
                for accum_id in range(1, num_accums):
                    start_accum = int(accum_id * max_features_per_accum)
                    end_accum = int(min((accum_id + 1) * max_features_per_accum, local_features.shape[0]))
                    if accum_id % 1000 == 0:
                        logging.debug('accumulating (rank: {}) {} of {}'.format(comm.rank, accum_id, num_accums))
                    local_llk += ubm._expectation(local_accum, local_features[start_accum:end_accum, :])
                local_llk = numpy.array(local_llk)
            else:
                local_llk = numpy.array(ubm._expectation(local_accum, local_features))

            # Reduce all accumulators in process 1
            comm.Barrier()
            comm.Reduce(
                [local_accum.w, MPI.DOUBLE],
                [accum.w, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            comm.Reduce(
                [local_accum.mu, MPI.DOUBLE],
                [accum.mu, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            comm.Reduce(
                [local_accum.invcov, MPI.DOUBLE],
                [accum.invcov, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            comm.Reduce(
                [local_llk, MPI.DOUBLE],
                [_tmp_llk, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )
            comm.Barrier()
                
            if comm.rank == 0:
                llk.append(_tmp_llk / numpy.sum(accum.w))

                # M step
                logging.critical("\nStart M-step, rank {}".format(comm.rank))
                ubm._maximization(accum, ceil_cov=ceil_cov, floor_cov=floor_cov)

                if i > 0:
                    # gain = llk[-1] - llk[-2]
                    # if gain < llk_gain:
                        # logging.debug(
                        #    'EM (break) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        #    self.mu.shape[0], i + 1, it, gain, self.name,
                        #    len(cep))
                    #    break
                    # else:
                        # logging.debug(
                        #    'EM (continu) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        #    self.mu.shape[0], i + 1, it, gain, self.name,
                        #    len(cep))
                    #    break
                    pass
                else:
                    # logging.debug(
                    #    'EM (start) distrib_nb: %d %i/%i llk: %f -- %s, %d',
                    #    self.mu.shape[0], i + 1, it, llk[-1],
                    #    self.name, len(cep))
                    pass
             # Send the new Mixture to all process
            comm.Barrier()
            ubm = comm.bcast(ubm, root=0)
            comm.Barrier()
    if comm.rank == 0:
        ubm.write(output_filename + '_{}g.h5'.format(ubm.get_distrib_nb()), prefix='')
    #return llk

