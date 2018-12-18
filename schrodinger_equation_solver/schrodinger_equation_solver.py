# -*- coding: utf-8 -*-

"""Main module."""

import sys
import argparse
import tensorflow as tf
tf.enable_eager_execution()

def parse_args(args):
    '''
    Parsing arguments into variables
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-P', '--potential_energy', type=str, \
    default='potential_energy.dat', \
    help='The path to the potential energy table')

    parser.add_argument('-c', type=float, required=True, \
    default='1', \
    help='The constant c in the equation')

    parser.add_argument('-s', '--size', type=int, required=True, \
    help='The size of the basis set')

    parser.add_argument('-d', '--domain', type=str, required=True, \
    help='a domain')
    return vars(parser.parse_args(args))


def parse_file(file):
    '''
    parse the input text file into python lists

    Args:
    file: the path to the input potential energy file

    Returns:
    position: a position list 
    potential_energy: a potential energy list
    '''
    position = list()
    potential_energy = list()
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                line = line.strip('\n')
                line = line.split()
                position.append(float(line[0]))
                potential_energy.append(float(line[1]))
    position = tf.constant(position, tf.float32)
    potential_energy = tf.constant(potential_energy, tf.float32)
    return position, potential_energy
        

def form_basis(size):
    '''
    form the fourier basis with given size.

    Args:
    size: the size of the basis set that required by the user

    Returns:
    basis: a list of basis set functions
    '''
    basis = list()
    for i in range(size):
        if i == 0:
            basis.append(lambda x: tf.math.sin(x) - tf.math.sin(x) + 1)
        elif i % 2 == 1: # sin function
            basis.append(lambda x: tf.math.sin((i+1)/2*x))
        else: #cos function
            basis.append(lambda x: tf.math.cos((i/2)*x))
    return basis

def form_matrix(c, size):
    '''
    Form the first part of the hamiltonian matrix. The definition of the 
    hamiltonian is -c(del^2)psi, and this function evaluates that. 

    Args:
    c: the constant c
    size: size of the basis set

    Returns:

    '''
    for i in range(size):
        row = [0.0] * size
        if i == 0:
            matrix = tf.constant(row, tf.float32, shape=(1,size))
        else:
            row[i] = ((i + 1) // 2) ** 2 * (-c)
            row = tf.constant(row, tf.float32, shape=(1,size))
            matrix = tf.concat([matrix, row], 0)
    return matrix

def calculate_inner_V0_b(position, potential_energy, basis):
    '''
    Calculates the left hand side when projecting H onto basis.
    The left hand side is the inner product between V0 and the basis.

    Args:
    position: a tensor that contains the position values
    potential_energy: a tensor that contains the potential energy of the system
    basis: a list of basis
    '''
    length = position.shape[0] # the length of the input data
    row = tf.reshape(tf.multiply(basis[0](position), potential_energy),[1,length])
    #print("row: ", row)
    for i, b in enumerate(basis):
        if i == 0:
            continue
        row = tf.concat([row, tf.reshape(tf.multiply(b(position), potential_energy),[1,length])], 0)
        #print(row.shape)
    row = tf.reduce_sum(row, 1)
    #print(row)
    return row

def calculate_inner_V0hat_b(position, basis):
    '''
    calcuate the right hand side.
    '''
    basis_size = len(basis)
    position_size = position.shape[0]
    #print(position_size)
    for bi in basis:
        m2 = bi(position)
        #print("m2: ", m2)
        multiply = tf.constant([basis_size])
        m2 = tf.tile(m2, multiply)
        #print(m2.shape)
        m2 = tf.reshape(m2,[basis_size, position_size])
        bj = basis[0]
        m1 = tf.reshape(bj(position),[1, position_size])
        for i,bj in enumerate(basis):
            if i == 0:
                continue
            else:
                m1 = tf.concat([m1,tf.reshape(bj(position),[1, position_size])], 0)
        try:
            result
        except NameError:
            result = tf.reshape(tf.reduce_sum(tf.multiply(m1, m2),1),[1,-1])
        else:
            result = tf.concat([result, tf.reshape(tf.reduce_sum(tf.multiply(m1, m2),1),[1,-1])],0)
    return result

def form_H(a_tensor, c_tensor, basis_size):
    multiply = tf.constant([basis_size])
    H = tf.reshape(tf.tile(tf.squeeze(a_tensor), multiply),[-1,basis_size])
    H = tf.transpose(H)
    #print(H)
    return H

def main():
    args = parse_args(sys.argv[1:])
    position, potential_energy = \
    parse_file('schrodinger_equation_solver/potential_energy.dat')
    #print("positions: ", position)
    #print("potential_energy: ", potential_energy)
    # print(potential_energy.shape[0])
    basis = form_basis(args['size'])
    #print("len of basis: ", len(basis))
    matrix = form_matrix(args['c'], args['size'])
    #print("matrix: ", matrix)
    rhs = calculate_inner_V0_b(position, potential_energy, basis)
    #print("rhs: ", rhs)
    lhs = calculate_inner_V0hat_b(position, basis)
    #print("lhs: ", lhs)
    a_tensor = tf.linalg.solve(lhs,tf.reshape(rhs,[rhs.shape[0],1]))
    #print(a_tensor)
    H = form_H(a_tensor, matrix, args['size'])
    e,v = tf.linalg.eigh(H)
    print("The lowest energy is: ", e[0].numpy())
    print("The coefficient for the basis set of the corresponding wavefunction is: ", v[0].numpy())

if __name__ == "__main__":
    import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
