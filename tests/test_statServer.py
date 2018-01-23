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
Copyright 2017 Andreas Nautsch, Anthony Larcher (SIDEKIT)
          2018      Ewald Enzinger (FVCKIT)

UnitTest TestCase for StatServer
--- please feel free to augment test_* functions and to implement skipped test cases

note: for PyCharm users, edit run configuration: check params box, with value: --nologcapture
"""

from unittest import TestCase, skip
from fvckit import StatServer, IdMap, STAT_TYPE, Mixture, FeaturesServer
from tests import skipped_funcs, skipped_test_funcs
import tempfile
import numpy
import copy
import os
import logging


def create_small_test_idmap():
    idmap = IdMap()
    idmap.leftids = numpy.array(['foo'], dtype='object')
    idmap.rightids = numpy.array(['bar'], dtype='object')
    idmap.start = numpy.empty(len(idmap.rightids), dtype='object')
    idmap.stop = numpy.empty(len(idmap.rightids), dtype='object')
    return idmap


class ChildStatServer(StatServer):
    """
    exemplary implementation for testing StatServer inheritance to ChildStatServers
    """

    def __init__(self, statserver):
        super(ChildStatServer, self).__init__()
        self.modelset = copy.deepcopy(statserver.modelset)
        self.segset = copy.deepcopy(statserver.segset)
        self.start = copy.deepcopy(statserver.start)
        self.stop = copy.deepcopy(statserver.stop)
        self.stat0 = copy.deepcopy(statserver.stat0)
        self.stat1 = copy.deepcopy(statserver.stat1)

        if hasattr(statserver, 'stat_extension'):
            self.stat_extension = statserver.stat_extension
        else:
            self.stat_extension = numpy.empty(self.stat0.shape, dtype=STAT_TYPE)

    def merge(*arg):
        id_set, dim_stat0, dim_stat1 = StatServer.merge_get_list_unique_mod_seg_ids(arg)
        new_stat_server = StatServer.merge(*arg)

        new_stat_server.stat_extension = numpy.zeros((len(id_set), dim_stat0), dtype=STAT_TYPE)
        for ss in arg:
            for idx, segment in enumerate(ss.segset):
                new_idx = numpy.argwhere(new_stat_server.segset == segment)
                new_stat_server.stat_extension[new_idx] = ss.stat_extension[idx]

        new_stat_server = ChildStatServer(new_stat_server)

        assert(new_stat_server.validate()), "Problem in StatServer Merging"
        return new_stat_server

    def accumulate_stat_nested_template(self, ubm, feature_server, feature_server_stat1,
                                        count, idx, show, data, lp, pp, log_lk):
        self.stat_extension = 42


class TestStatServer(TestCase):
    def setUp(self):
        self.tempfile_list = []

    def tearDown(self):
        for tmp_path in self.tempfile_list:
            os.remove(tmp_path)

    @classmethod
    def tearDownClass(cls):
        # check untested methods in parent class
        methods_totest = StatServer.__dict__.keys()
        tested_methods = TestStatServer.__dict__.keys()
        test_skip_funcs = ['create_idmap_statsserver_with_subset_index', 'create_empty_statserver',
                           'create_small_statserver', 'create_file_statserver']
        for func in methods_totest:
            if 'test_' + func not in tested_methods:
                if func not in skipped_funcs:
                    logging.warning('untested function: {}'.format(func))
        for func in tested_methods:
            class_func = func[5:]
            if class_func not in methods_totest:
                if func not in skipped_test_funcs + test_skip_funcs:
                    logging.warning('tested function not in class under test: {}'.format(func))

    def create_empty_statserver(self):
        s = StatServer()
        self.assertTrue(s.validate())
        return s

    def create_idmap_statsserver_with_subset_index(self):
        idmap_a = IdMap()
        idmap_a.leftids = numpy.array(['a', 'a'], dtype='object')
        idmap_a.rightids = numpy.array(['test/show', 'foo/bar_a'], dtype='object')
        idmap_a.start = numpy.empty(len(idmap_a.rightids), dtype='object')
        idmap_a.stop = numpy.empty(len(idmap_a.rightids), dtype='object')
        self.assertTrue(idmap_a.validate())
        idmap_b = IdMap()
        idmap_b.leftids = numpy.array(['b', 'c'], dtype='object')
        idmap_b.rightids = numpy.array(['fvckit/show_b', 'show'], dtype='object')
        idmap_b.start = numpy.empty(len(idmap_b.rightids), dtype='object')
        idmap_b.stop = numpy.empty(len(idmap_b.rightids), dtype='object')
        self.assertTrue(idmap_b.validate())
        idmap = IdMap.merge(idmap_a, idmap_b)
        self.assertTrue(idmap.validate())
        s = StatServer(idmap, distrib_nb=2, feature_size=3)
        self.assertTrue(s.validate())
        return s, idmap_a, idmap_b, idmap

    def create_small_statserver(self):
        idmap = create_small_test_idmap()
        s = StatServer(idmap, distrib_nb=2, feature_size=3)
        self.assertTrue(s.validate())
        return s

    def create_file_statserver(self, statserver):
        assert(isinstance(statserver, StatServer) and statserver.validate()), "Argument must be proper StatServer"
        tmp, filename = tempfile.mkstemp()
        self.tempfile_list.append(filename)
        statserver.write(filename)
        return filename

    def test___init__(self):
        empty_statserver = self.create_empty_statserver()
        self.assertTrue(empty_statserver.validate())
        idmap_statserver, idmap_a, idmap_b, idmap = self.create_idmap_statsserver_with_subset_index()
        self.assertTrue(idmap_statserver.validate())
        statserver_tmpfile = self.create_file_statserver(idmap_statserver)
        idmap_tmp_statserver = StatServer(statserver_tmpfile)
        self.assertTrue(idmap_tmp_statserver.validate())
        idmap_a_statserver = StatServer(statserver_tmpfile, distrib_nb=2, feature_size=3, index=idmap_a)
        self.assertTrue(idmap_a_statserver.validate())
        idmap_b_statserver = StatServer(statserver_tmpfile, distrib_nb=2, feature_size=3, index=idmap_b)
        self.assertTrue(idmap_b_statserver.validate())

    def test_validate(self):
        # validate should return ok = True
        # from test_init
        empty_statserver = self.create_empty_statserver()
        self.assertTrue(empty_statserver.validate())
        idmap_statserver, idmap_a, idmap_b, idmap = self.create_idmap_statsserver_with_subset_index()
        self.assertTrue(idmap_statserver.validate())
        statserver_tmpfile = self.create_file_statserver(idmap_statserver)
        idmap_tmp_statserver = StatServer(statserver_tmpfile, distrib_nb=2, feature_size=3)
        self.assertTrue(idmap_tmp_statserver.validate())
        idmap_a_statserver = StatServer(statserver_tmpfile, distrib_nb=2, feature_size=3, index=idmap_a)
        self.assertTrue(idmap_a_statserver.validate())
        idmap_b_statserver = StatServer(statserver_tmpfile, distrib_nb=2, feature_size=3, index=idmap_b)
        self.assertTrue(idmap_b_statserver.validate())
        # others // note: not yet implemented

        # validate should return ok = False
        malicious_empty_statserver = copy.deepcopy(empty_statserver)
        # modelset ndim > 1, rest is fine
        malicious_empty_statserver.modelset = numpy.array([['foo'],['bar']], dtype='object')
        malicious_empty_statserver.segset = malicious_empty_statserver.modelset
        malicious_empty_statserver.start = malicious_empty_statserver.modelset
        malicious_empty_statserver.stop = malicious_empty_statserver.modelset
        malicious_empty_statserver.stat0 = numpy.zeros((1, 3), dtype=STAT_TYPE)
        malicious_empty_statserver.stat1 = numpy.zeros((1, 12), dtype=STAT_TYPE)
        self.assertFalse(malicious_empty_statserver.validate())
        malicious_empty_statserver.stat0 = numpy.zeros((1, 3), dtype=STAT_TYPE)
        malicious_empty_statserver.stat1 = numpy.zeros((1, 9), dtype=STAT_TYPE)
        self.assertFalse(malicious_empty_statserver.validate())

        # shape mismatch in modelset, segset, start, stop
        malicious_empty_statserver.segset = empty_statserver.segset
        self.assertFalse(malicious_empty_statserver.validate())

        # shape[0] mismatch in modelset, stat0, stat1
        malicious_empty_statserver.modelset = numpy.array(['foo', 'bar'], dtype='object')
        malicious_empty_statserver.segset = malicious_empty_statserver.modelset
        malicious_empty_statserver.start = malicious_empty_statserver.modelset
        malicious_empty_statserver.stop = malicious_empty_statserver.modelset
        malicious_empty_statserver.stat0 = numpy.zeros((1, 3), dtype=STAT_TYPE)
        malicious_empty_statserver.stat1 = numpy.zeros((1, 12), dtype=STAT_TYPE)
        self.assertFalse(malicious_empty_statserver.validate())

        # shape[1] mismatch in stat1 as multiple to stat0 (ndistribution * nfeatures)
        malicious_empty_statserver.stat0 = numpy.zeros((1, 3), dtype=STAT_TYPE)
        malicious_empty_statserver.stat1 = numpy.zeros((1, 11, ), dtype=STAT_TYPE)
        self.assertFalse(malicious_empty_statserver.validate())
        malicious_empty_statserver.stat0 = numpy.zeros((1, 3), dtype=STAT_TYPE)
        malicious_empty_statserver.stat1 = numpy.zeros((1, 10, ), dtype=STAT_TYPE)
        self.assertFalse(malicious_empty_statserver.validate())

    @skip("not-yet implemented UnitTest")
    def test_merge_get_list_unique_mod_seg_ids(self):
        self.fail()

    def test_merge(self):
        # merge two statservers
        idmap_statserver, idmap_a, idmap_b, idmap = self.create_idmap_statsserver_with_subset_index()
        statserver_tmpfile = self.create_file_statserver(idmap_statserver)
        idmap_a_statserver = StatServer(statserver_tmpfile, distrib_nb=2, feature_size=3, index=idmap_a)
        idmap_b_statserver = StatServer(statserver_tmpfile, distrib_nb=2, feature_size=3, index=idmap_b)
        merged_statserver = StatServer.merge(idmap_a_statserver, idmap_b_statserver)
        self.assertTrue(merged_statserver.validate())
        self.assertTrue(merged_statserver.stat0.shape == idmap_statserver.stat0.shape)

        # merge two childstatservers
        child = ChildStatServer(idmap_statserver)
        child_a = ChildStatServer(idmap_a_statserver)
        child_b = ChildStatServer(idmap_b_statserver)
        merged_child = ChildStatServer.merge(child_a, child_b)
        self.assertTrue(merged_child.validate())
        self.assertTrue(merged_child.stat0.shape == child.stat0.shape)

    def test_accumulate_stat_nested_template(self):
        # solely relevant to inherited classes
        small_statserver = self.create_small_statserver()
        small_child = ChildStatServer(small_statserver)
        ubm = Mixture()
        ubm.w = numpy.ones(2) / 2
        ubm.mu = numpy.zeros((2, 3))
        ubm.invcov = numpy.ones((2, 3))
        ubm._split_ditribution()
        ubm._compute_all()
        self.assertTrue(ubm.validate())
        feature_server = FeaturesServer()
        # dummy previous load
        feature_server.show = 'bar'
        feature_server.input_feature_filename = None
        feature_server.start_stop = (None, None)
        feature_server.previous_load = numpy.ones((4, 3)), numpy.ones(4, dtype=bool)
        # call nested_template's calling function
        small_child.accumulate_stat(ubm, feature_server)
        self.assertEqual(small_child.stat_extension, 42)

    @skip("not-yet implemented UnitTest")
    def test_read(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_write(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_model_stat0(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_model_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_model_stat0_by_index(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_model_stat1_by_index(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_segment_stat0(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_segment_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_segment_stat0_by_index(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_segment_stat1_by_index(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_model_segments(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_model_segments_by_index(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_align_segments(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_align_models(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_accumulate_stat(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_mean_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_norm_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_rotate_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_center_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_subtract_weighted_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_whiten_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_whiten_cholesky_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_total_covariance_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_within_covariance_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_between_covariance_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_lda_matrix_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_mahalanobis_matrix_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_wccn_choleski_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_get_nap_matrix_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_adapt_mean_map(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_adapt_mean_map_multisession(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_precompute_svm_kernel_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_ivector_extraction_weight(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_ivector_extraction_eigen_decomposition(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_estimate_spectral_norm_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_spectral_norm_stat1(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_sum_stat_per_model(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_mean_stat_per_model(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test__expectation(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test__maximization(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_estimate_between_class(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_estimate_within_class(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_estimate_map(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_estimate_hidden(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_factor_analysis(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_generator(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_read_subset(self):
        self.fail()
