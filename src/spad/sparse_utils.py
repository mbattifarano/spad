"""Sparse matrix utilities."""
import numpy as np
from scipy.sparse import coo_matrix, spmatrix
import tensorflow as tf


def coo_indices(a: coo_matrix) -> np.array:
    """Return the indices of a COO matrix as a nnz x 2 array.

    It is assumed that the matrix does not contain duplicate
    entried and that the indices are sorted.

    Args:
        a (coo_matrix): A scipy sparse matrix in COO format.

    Returns:
        np.array: An (nnz, 2) array of indices.
    """
    return np.vstack([a.row, a.col]).T


def coo_values(a: coo_matrix) -> np.array:
    """Return the (non-zero) values of a COO sparse matrix.

    Args:
        a (coo_matrix): A scipy sparse matrix in COO format.

    Returns:
        np.array: An (nnz,) array of values
    """
    return a.data


def to_tf_sparse(a: spmatrix) -> tf.SparseTensor:
    """Convert a scipy sparse matrix to a tensorflow sparse matrix.

    Args:
        a (spmatrix): A scipy sparse matrix (any format).

    Returns:
        tf.SparseTensor: An equivalent tensorflow sparse matrix.
    """
    coo = a.tocoo()
    return tf.SparseTensor(
        list(zip(coo.row, coo.col)),
        coo.data,
        coo.shape
    )


def tf_sparse_scalar_multiply(m: float, a: tf.SparseTensor) -> tf.SparseTensor:
    """Multiply the values of a sparse tensor by a scalar.

    Args:
        m (float): scalar multiplier.
        a (tf.SparseTensor): tensor to scale.

    Returns:
        tf.SparseTensor: the scaled sparse tensor.
    """
    return tf.sparse.map_values(tf.multiply, m, a)


def tf_sparse_negate(a: tf.SparseTensor) -> tf.SparseTensor:
    """Return the negative of a sparse tensor

    Args:
        a (tf.SparseTensor): the tensor to negate

    Returns:
        tf.SparseTensor: the negated tensor
    """
    return tf_sparse_scalar_multiply(-1.0, a)


def tf_sparse_repmat(a: tf.SparseTensor, n: int, axis: int = 0) \
        -> tf.SparseTensor:
    """Replicate the sparse tensor a n times over the specified axis.

    Args:
        a (tf.SparseTensor): the sparse tensor to replicate
        n (int): the number of times to replicate
        axis (int, optional): the axis to replicate over. Defaults to 0.

    Returns:
        tf.SparseTensor: A sparse tensor with n compies of the input
            concatenated over the specified axis.
    """
    return tf.sparse.concat(
        sp_inputs=[a] * n,
        axis=axis
    )
