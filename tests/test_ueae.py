"""Test the UEAE module."""
import pytest
from spad import ueae
from spad.sparse_utils import to_tf_sparse
import tensorflow as tf
from hypothesis import given, note
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
import numpy as np


@given(
    arrays(np.float64, (5, 5), elements=floats(0, 10, allow_nan=False))
)
def test_logit_layer(link_cost):
    """Test that the logit layer correctly computes utilities."""
    layer = ueae.LogitLayer()
    assert not layer.trainable
    actual = layer(tf.sparse.from_dense(link_cost))
    assert isinstance(actual, tf.SparseTensor)
    np_actual = tf.sparse.to_dense(actual).numpy()
    expected = np.where(link_cost == 0.0, 0.0, np.exp(-link_cost))
    note(f"actual = {np_actual}")
    note(f"expected = {expected}")
    assert np.allclose(np_actual, expected)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64], ids=str)
def test_stochastic_network_loading_layer(dtype,
                                          braess_od_demand,
                                          braess_ue_link_cost,
                                          braess_ue_link_flow):
    """Test that the SNL layer correctly loads the network."""
    cost = tf.cast(
        tf.sparse.expand_dims(to_tf_sparse(braess_ue_link_cost), 0),
        dtype
    )
    print(f"link cost = {cost.values}")
    node_constants = tf.constant([0.0, 0.0, 10.0, 50.0], dtype=dtype)
    logits = ueae.LogitLayer(node_constants, dtype=dtype)
    snl = ueae.StochasticNetworkLoading(braess_od_demand, dtype=dtype)
    link_logits = logits(cost)
    print(f"link logits = {link_logits.values}")
    link_flow = snl(link_logits)
    assert isinstance(link_flow, tf.SparseTensor)
    print(f"output link flow = {link_flow.values}")
    print(f"expected link flow = {braess_ue_link_flow.data}")
    assert np.allclose(
        tf.sparse.to_dense(link_flow).numpy(),
        braess_ue_link_flow.toarray()
    )


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64], ids=str)
def test_ueae_ue(dtype, braess_od_demand, braess_ue_link_cost, braess_ue_link_flow):
    """Test that the UEAE correctly represents the fixed point."""
    print(dtype)
    assert False
