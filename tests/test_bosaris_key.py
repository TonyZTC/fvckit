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

UnitTest TestCase for bosaris.Key
--- please feel free to augment test_* functions and to implement skipped test cases

note: for PyCharm users, edit run configuration: check params box, with value: --nologcapture
"""
from unittest import TestCase, skip
import logging
import os
import numpy
import h5py
import tempfile
from fvckit import Key
from tests import skipped_funcs, skipped_test_funcs


class TestKey(TestCase):
    def setUp(self):
        self.tempfile_list = []

    def create_file_key(self, key, flag_matlab=False):
        assert(isinstance(key, Key) and key.validate()), "Argument must be proper Key"
        tmp, filename = tempfile.mkstemp()
        self.tempfile_list.append(filename)
        if flag_matlab:
            key.write_matlab(filename)
        else:
            key.write(filename)
        return filename

    def tearDown(self):
        for tmp_path in self.tempfile_list:
            os.remove(tmp_path)

    @classmethod
    def tearDownClass(cls):
        # check untested methods in parent class
        methods_totest = Key.__dict__.keys()
        tested_methods = TestKey.__dict__.keys()
        test_skip_funcs = ['create_small_key', 'create_file_key', 'create_empty_key']
        for func in methods_totest:
            if 'test_' + func not in tested_methods:
                if func not in skipped_funcs:
                    logging.warning('untested function: {}'.format(func))
        for func in tested_methods:
            class_func = func[5:]
            if class_func not in methods_totest:
                if func not in skipped_test_funcs + test_skip_funcs:
                    logging.warning('tested function not in class under test: {}'.format(func))

    def create_empty_key(self):
        k = Key()
        self.assertTrue(k.validate())
        return k

    def create_small_key(self):
        k = Key()
        k.modelset = numpy.array(['foo', 'bar'], dtype='object')
        k.segset = numpy.array(['xy_a', 'xy_b', 'z'], dtype='object')
        k.tar = numpy.array([[1, 0, 0], [0, 1, 0]], dtype='bool')
        k.non = ~k.tar
        self.assertTrue(k.validate())
        return k

    def test_write_matlab(self):
        kempty = self.create_empty_key()
        kempty_file = self.create_file_key(kempty, flag_matlab=True)
        kempty_h5 = h5py.File(kempty_file)
        kempty_keys = list(kempty_h5.keys())
        self.assertEqual(len(numpy.setdiff1d(kempty_keys, ['ID', 'file_format', 'trial_mask'])), 0)
        kempty_id_keys = list(kempty_h5['ID'].keys())
        self.assertEqual(len(numpy.setdiff1d(kempty_id_keys, ['row_ids', 'column_ids'])), 0)

        ksmall = self.create_small_key()
        ksmall_file = self.create_file_key(ksmall, flag_matlab=True)
        ksmall_h5 = h5py.File(ksmall_file)
        ksmall_keys = list(ksmall_h5.keys())
        self.assertEqual(len(numpy.setdiff1d(ksmall_keys, ['ID', 'file_format', 'trial_mask'])), 0)
        ksmall_id_keys = list(ksmall_h5['ID'].keys())
        self.assertEqual(len(numpy.setdiff1d(ksmall_id_keys, ['row_ids', 'column_ids'])), 0)

    def test_read(self):
        kempty = self.create_empty_key()
        kempty_file = self.create_file_key(kempty)
        kempty_file_matlab = self.create_file_key(kempty, flag_matlab=True)
        ksmall = self.create_small_key()
        ksmall_file = self.create_file_key(ksmall)
        ksmall_file_matlab = self.create_file_key(ksmall, flag_matlab=True)

        kempty_read = Key.read(kempty_file)
        self.assertTrue(kempty_read.validate())
        kempty_read_matlab = Key.read(kempty_file_matlab)
        self.assertTrue(kempty_read_matlab.validate())
        ksmall_read = Key.read(ksmall_file)
        self.assertTrue(ksmall_read.validate())
        ksmall_read_matlab = Key.read(ksmall_file_matlab)
        self.assertTrue(ksmall_read_matlab.validate())

    def test_write(self):
        kempty = self.create_empty_key()
        kempty_file = self.create_file_key(kempty)
        kempty_h5 = h5py.File(kempty_file)
        kempty_keys = list(kempty_h5.keys())
        self.assertEqual(len(numpy.setdiff1d(kempty_keys, ['modelset', 'segset', 'trial_mask'])), 0)

        ksmall = self.create_small_key()
        ksmall_file = self.create_file_key(ksmall)
        ksmall_h5 = h5py.File(ksmall_file)
        ksmall_keys = list(ksmall_h5.keys())
        self.assertEqual(len(numpy.setdiff1d(ksmall_keys, ['modelset', 'segset', 'trial_mask'])), 0)

    @skip("not-yet implemented UnitTest")
    def test_write_txt(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_filter(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_to_ndx(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_validate(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_read_txt(self):
        self.fail()

    @skip("not-yet implemented UnitTest")
    def test_merge(self):
        self.fail()
