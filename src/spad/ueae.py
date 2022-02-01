"""User Equilibrium AutoEncoder."""
from os import lockf
import numpy as np
import tensorflow as tf
from tensosflow_probability import bijectors
from tensorflow.keras.layers import Layer
from scipy.sparse import coo_matrix
from .sparse_utils import (
    coo_indices,
    coo_values,
    tf_sparse_negate,
    tf_sparse_repmat,
)

from .common import default_on_none


class UEAE(tf.keras.Model):
    """Simultaneous Preference and Demand (SPAD) Estimator via a User
    Equilibrium Auto Encorder (UEAE)

    The SPAD estimator functions as an autoencoder in observation space:

        link measurments (observed)
         |
        [preprocessor (forward)]
         v
        link flow
         |
        [link cost function]
         v
        link cost
         |
        [logit transformation]
         v
        link utilities
         |
        [stochastic network loading]
         v
        link flow (recovered)
         |
        [preprocessor (inverse)]
         v
        link measurements (recovered)

    As this is an auto-encoder, it should be fit with the training data as its
    own target: `model.fit(obs, obs)`
    """

    def __init__(
        self,
        link_cost_estimator: Layer,
        link_flow_loader: "StochasticNetworkLoading",
        logits: "LogitLayer",
        preprocessor: bijectors.Bijector = None,
    ) -> None:
        """Initialize the SPAD model.

        Args:
            link_cost_estimator (Layer): The estimator that convert link
                flow to link cost.
            link_flow_loader (StochasticNetworkLoading): The estimator that
                loads the network as a function of utilities.
            logits (LogitLayer): [description] Convert cost to utilities in
                logits.
            preprocessor (bijectors.Bijector, optional): a tensorflow bijector
                to convert between observation space and link flow space.
                Defaults to the identity bijector.
        """
        super().__init__()
        self.preprocessor = default_on_none(preprocessor, bijectors.Identity())
        self.link_cost_estimator = link_cost_estimator
        self.logits = logits
        self.link_flow_loader = link_flow_loader

    def call(self, inputs):
        flow_obs = self.preprocessor.forward(inputs)
        utilities = self.logits(self.link_cost_estimator(flow_obs))
        flow_est = self.link_flow_loader(utilities)
        inputs_est = self.preprocessor.inverse(flow_est)
        return inputs_est


class StochasticNetworkLoading(Layer):
    """Stochastic Network Loading (SNL) layer."""

    def __init__(
        self,
        q_init: coo_matrix,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        """Initialize the SNL layer with an initial OD demand.

        When the demand is known, this layer should set `trainable=False`.

        Args:
            q_init (coo_matrix): Initial OD demand matrix estimate.
            trainable (bool, optional): Whether the layer is trainable.
                Defaults to True.
            name (str optional): Name of the layer. Defaults to None.
            dtype (tf.DType, optional): Datatype of the layer.
                Defaults to None.
            dynamic (bool, optional): Defaults to False.
        """
        super().__init__(
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
            **kwargs,
        )
        validate_demand_initializer(q_init)
        self.n_nodes = q_init.shape[0]
        self.trips = coo_indices(q_init)
        self._demand = tf.Variable(
            np.sqrt(coo_values(q_init)),
            trainable=self.trainable,
            dtype=self.dtype,
        )
        self.identity = tf.sparse.eye(self.n_nodes, dtype=self.dtype)

    @property
    def demand(self) -> tf.Tensor:
        """Return the OD demand vector.

        Returns:
            tf.Tensor: The vector of OD demand
        """
        return self._demand ** 2

    @property
    def Q(self) -> tf.SparseTensor:
        """Return the OD demand matrix.

        Returns:
            tf.SparseTensor: The OD demand maxtrix
        """
        return tf.SparseTensor(
            indices=self.trips,
            values=self.demand,
            dense_shape=(self.n_nodes, self.n_nodes),
        )

    def link_weights(self, inputs: tf.SparseTensor) -> tf.SparseTensor:
        """Compute the link assignment weight matrix.

        The weight matrix is a samples x nodes x nodes matrix of values such
        that when elementwise multiplied by the samples x nodes x nodes matrix
        of link logits yields the link flow assignment.

        If L is the link logit tensor then the weight tensor W can be written;

        $$W_{iuv} = sum_{rs} Q_{rs} V{iru} V_{ivs} (V_{irs}^{-1})$$.

        where V = (I-L)^{-1}. The assignment is performed as;

        $$x_{iuv} = L_{iuv} W_{iuv}$$.

        For further details see
        Boyles, S. D., N. E. Lownes, and A. Unnikrishnan. (2021) Transportation
        Network Analysis, Volume I, Version 0.89.
        https://sboyles.github.io/book.pdf

        Args:
            inputs (tf.SparseTensor): A samples x nodes x nodes sparse tensor
                of link logits

        Returns:
            tf.SparseTensor: The weight tensor
        """
        n_samples = inputs.shape[0]
        identity = tf_sparse_repmat(
            tf.sparse.expand_dims(self.identity, 0), n_samples, axis=0
        )
        Q = tf.sparse.to_dense(self.Q)
        V = tf.linalg.inv(
            tf.sparse.to_dense(
                tf.sparse.add(identity, tf_sparse_negate(inputs))
            )
        )
        normalizer = 1.0 / tf.where(tf.equal(V, 0.0), 1.0, V)
        return tf.einsum("rs, iru, ivs, irs -> iuv", Q, V, V, normalizer)

    def call(self, inputs: tf.SparseTensor) -> tf.SparseTensor:
        """Perform a Stochastic Network Loading based on input link logits.

        Args:
            inputs (tf.SparseTensor): A samples x nodes x nodes sparse tensor
                of link logits.

        Returns:
            tf.SparseTensor: A samples x nodes x nodes sparse tensor of link
                flows.
        """
        W = self.link_weights(inputs)
        return inputs * W


class LogitLayer(Layer):
    """Convert cost to utility in logits."""

    def __init__(
        self,
        node_constants: tf.Tensor = None,
        trainable=False,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        """Initialize the logit layer.

        This layer is non-trainable by default.

        Args:
            trainable (bool, optional): Defaults to False.
            name ([type], optional): Defaults to None.
            dtype ([type], optional): Defaults to None.
            dynamic (bool, optional): Defaults to False.
        """
        super().__init__(
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
            **kwargs,
        )
        self.link_constants = (
            tf.constant(0.0, dtype=self.dtype)
            if node_constants is None
            else to_link_constants(node_constants)
        )
        self.rationality = tf.Variable(
            1.0, trainable=self.trainable, dtype=self.dtype
        )

    def to_logits(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return the utility (in logits) of the input cost.

        Args:
            inputs (tf.Tensor): Cost, assumed to be non-negative.

        Returns:
            tf.Tensor: Utility in logits.
        """
        return tf.exp(-self.rationality * inputs)

    def condition_costs(self, inputs: tf.SparseTensor) -> tf.SparseTensor:
        """Compute conditioned link costs for numerical stability.

        Args:
            inputs (tf.SparseTensor): [description]

        Returns:
            tf.SparseTensor: [description]
        """
        mask = inputs.with_values(tf.ones_like(inputs.values))
        return tf.sparse.add(
            tf.broadcast_to(self.link_constants, mask.shape) * mask, inputs
        )

    def call(self, inputs: tf.SparseTensor) -> tf.SparseTensor:
        """Compute the utility (in logits) of the sparse input cost.

        Args:
            inputs (tf.SparseTensor): Cost, assumed to be non-negative.

        Returns:
            tf.SparseTensor: Utility in logits
        """
        return tf.sparse.map_values(
            self.to_logits, self.condition_costs(inputs)
        )


class SparseLinearLayer(Layer):
    """A linear layer for sparse tensors."""

    # TODO: implement
    # idea: W (link x links sparse) * x (X.values.reshape((batch, n_links))) + b


def validate_demand_initializer(q_init: coo_matrix):
    """Raise an Exception if q_init is maformed.

    Args:
        q_init (coo_matrix): The initial estimate of OD demand.
    """
    if not isinstance(q_init, coo_matrix):
        raise UEAEException(
            f"OD demand matrix must be in COO format, got {type(q_init)}."
        )
    n, m = q_init.shape
    if n != m:
        raise UEAEException(
            f"OD demand matrix must be square (nodes x nodes) got: {n} x {m}."
        )


class UEAEException(Exception):
    """Base Exception for the UEAE module."""

    pass


def to_link_constants(node_constants: tf.Tensor) -> tf.Tensor:
    """Convert a 1D tensor of node constants to a 2D tensor of link constants.

    The link constant is a nodes x nodes matrix L such that for a 1D tensor N
    of node constants, we have;

    $$L_{ij} = N_i - N_j$$

    This constant will be added to the link cost so that the utility (in
    logits) of a link is computed;

    $$U_{ij} = exp(-(L_{ij} + C_{ij})) = exp(N_j - N_i - t_{ij})$$

    Args:
        node_constants (tf.Tensor): A 1D tensor of node constants

    Returns:
        tf.Tensor: A nodes x nodes tensor of link constants
    """
    (n_nodes,) = node_constants.shape
    nc = tf.repeat(tf.expand_dims(node_constants, 1), n_nodes, axis=1)
    return nc - tf.transpose(nc)
