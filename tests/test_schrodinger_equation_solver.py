#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `schrodinger_equation_solver` package."""

import unittest
import math
import tensorflow as tf
tf.enable_eager_execution()

from schrodinger_equation_solver import schrodinger_equation_solver as ses

class Test_schrodinger_equation_solver(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_parse_args(self):
        args = ses.parse_args(
            [
                '-i', 'Path/To/The/File',
                '-c', '5',
                '-s', '3',
            ]
        )
        self.assertIsInstance(args, dict)
        self.assertEqual(args['input'], 'Path/To/The/File')
        self.assertEqual(args['c'], 5)
        self.assertEqual(args['size'], 3)

    def test_parse_file(self):
        position, potential_energy = \
        ses.parse_file('schrodinger_equation_solver/potential_energy.dat')
        self.assertTrue(tf.equal(position.shape, position.shape).numpy()[0])
    
    def test_fourier_n(self):
        n0 = ses.fourier_n(0)
        n1 = ses.fourier_n(1)
        n2 = ses.fourier_n(2)
        x = tf.constant([0, math.pi/2, math.pi])
        n0_result = tf.constant([1,1,1], tf.float32)
        n1_result = tf.constant([0,1,0], tf.float32)
        n2_result = tf.constant([1,0,1], tf.float32)
        self.assertTrue(tf.equal(n0(x),n0_result).numpy()[0])
        self.assertTrue(tf.equal(n1(x),n1_result).numpy()[0])
        self.assertTrue(tf.equal(n2(x),n2_result).numpy()[0])
    
    def test_form_basis(self):
        basis = ses.form_basis(1)
        x = tf.constant([0, math.pi/2, math.pi])
        n0_result = tf.constant([1,1,1], tf.float32)
        self.assertTrue(tf.equal(basis[0](x),n0_result).numpy()[0])
    
    def test_form_matrix(self):
        matrix = ses.form_matrix(4, 5)
        self.assertEqual(matrix.shape, (5,5))
    
    def test_calculate_inner_V0_b(self):
        position = tf.constant([0,1,0], tf.float32)
        potential_energy = tf.constant([0,6,0], tf.float32)
        basis = [lambda x: tf.ones([x.shape[0]], tf.float32)]
        result = ses.calculate_inner_V0_b(position, potential_energy, basis)
        self.assertEqual(result.shape, tf.Dimension(1))
    
    def test_calculate_inner_V0hat_b(self):
        position = tf.constant([0,1,0], tf.float32)
        basis = [lambda x: tf.ones([x.shape[0]], tf.float32)]
        result = ses.calculate_inner_V0hat_b(position, basis)
        self.assertEqual(result.shape, (1,1))

    def test_form_H(self):
        a_tensor = tf.ones([3], tf.float32)
        c_tensor = tf.ones([3,3], tf.float32)
        H = ses.form_H(a_tensor, c_tensor, 3)
        self.assertTrue(H.shape, (3,3))

    def test_main(self):
        args = {'size':3, 
        'c':5.0, 
        'input':'schrodinger_equation_solver/potential_energy.dat'
        }
        e, v = ses.main(args)
        self.assertEqual(e.numpy().shape[0], 3)
        self.assertEqual(v.numpy().shape, (3,3))