# -*- coding: utf-8 -*-

"""Module containing several losses usable for supervised and unsupervised
training.

A loss is of the form::

    def loss(target, prediction, ...):
        ...

The results depends on the exact nature of the loss. Some examples are:

    - coordinate wise loss, such as a sum of squares or a Bernoulli cross
      entropy with a one-of-k target,
    - sample wise, such as neighbourhood component analysis.

In case of the coordinate wise losses, the dimensionality of the result should
be the same as that of the predictions and targets. In all other cases, it is
important that the sample axes (usually the first axis) stays the same. The
individual data points lie along the coordinate axis, which might change to 1.

Some examples of valid shape transformations::

    (n, d) -> (n, d)
    (n, d) -> (n, 1)

These are not valid::

    (n, d) -> (1, d)
    (n, d) -> (n,)


For some examples, consult the source code of this module.
"""

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy

from misc import distance_matrix


# TODO add hinge loss
# TODO add huber loss

def fmeasure(target, prediction, alpha=0.5):
    """Return the approximated f-measure loss between the `target` and
    the `prediction`. This is an overall loss, not a sample-wise one,
    that is, the loss is computed for all the samples at once.

    Jansche, Martin. "Maximum expected F-measure training of logistic
    regression models." Proceedings of the conference on Human Language
    Technology and Empirical Methods in Natural Language Processing.
    Association for Computational Linguistics, 2005.

    Parameters
    ----------

    target : Theano variable
        An array of arbitrary shape representing representing the targets.

    prediction : Theano variable
        An array of arbitrary shape representing representing the predictions.

    Returns
    -------

    res : Theano variable
        An array of the same columns as ``target`` and ``prediction``
        representing the overall f-measure loss."""
    n_pos = target.sum(axis=0)
    m_pos = prediction.sum(axis=0)
    true_positives_approx = target*prediction
    A = true_positives_approx.sum(axis=0)
    return 1-A/(alpha*n_pos+(1-alpha)*m_pos)

def squared(target, prediction):
    """Return the element wise squared loss between the `target` and
    the `prediction`.

    Parameters
    ----------

    target : Theano variable
        An array of arbitrary shape representing representing the targets.

    prediction : Theano variable
        An array of arbitrary shape representing representing the predictions.

    Returns
    -------

    res : Theano variable
        An array of the same shape as ``target`` and ``prediction``
        representing the pairwise distances."""
    return (target - prediction) ** 2


def _closest_pair_distance_cell(target, prediction, n_pred=20, lambda_cpd=1.0, lambda_confidence=1.0):

    n_pred = target.shape[0] // 6
    
    prediction_a = prediction[0:3]
    prediction_b = prediction[3:6]
    prediction_c = prediction[6]
    
    target_c = target[n_pred * 6]
    target_reshape = target[0:n_pred * 6].reshape((-1, 6))
    target_reshape = target_reshape[T.nonzero(T.gt(target_reshape, -1))].reshape((-1, 6))
    target_a = target_reshape[:, 0:3]
    target_b = target_reshape[:, 3:6]

    prediction_a_sum_square = T.sum(prediction_a ** 2, axis=0, keepdims=True)
    prediction_b_sum_square = T.sum(prediction_b ** 2, axis=0, keepdims=True)
    target_a_sum_square = T.sum(target_a ** 2, axis=1, keepdims=True)
    target_b_sum_square = T.sum(target_b ** 2, axis=1, keepdims=True)
    
    P_a = T.dot(prediction_a, target_a.T)
    P_b = T.dot(prediction_b, target_b.T)

    D_a = T.sqrt(prediction_a_sum_square + target_a_sum_square.T - 2 * P_a)
    D_b = T.sqrt(prediction_b_sum_square + target_b_sum_square.T - 2 * P_b)
    D = D_a + D_b

    loss_cpd = ifelse(
        T.gt(target_c, 0.5),
        D.min(),
        0.0
    )

    loss = lambda_cpd * loss_cpd  + lambda_confidence * T.nnet.binary_crossentropy(prediction_c, target_c)

    return loss


def _closest_pair_distance_volume(target, prediction, lambda_cpd=1.0, lambda_confidence=1.0):

    n_cells = prediction.shape[0]
    prediction_reshaped = prediction.reshape((n_cells * n_cells, -1))
    target_reshaped = target.reshape((n_cells * n_cells, -1))
    losses, updates = theano.map(_closest_pair_distance_cell, [prediction_reshaped, target_reshaped], non_sequences=[lambda_cpd, lambda_confidence])

    return losses.sum()


def closest_pair_distance_tensor4(target, prediction, lambda_cpd=1.0, lambda_confidence=1.0):

    losses, updates = theano.map(_closest_pair_distance_volume, [prediction, target], non_sequences=[lambda_cpd, lambda_confidence])

    return losses.reshape((-1, 1))



def _closest_pair_distance_vector(target, prediction):
    """Return the closest pair distance between the `target` and
    the `predicition`.

    Closest pair loss
    target:     {(x_i^a, y_i^a, z_i^a, x_i^b, y_i^b, z_i^b), i \in {1, ..., n}} = {(p_i, q_i), i \in {1, ..., n}}
    prediction: (X^a, Y^a, Z^a, X^b, Y^b, Z^b) = (P, Q)
    loss:       \min_i {d((x_i^a, y_i^a, z_i^a), (X^a, Y^a, Z^a)) + d((x_i^b, y_i^b, z_i^b), (X^b, Y^b, Z^b))}
              = \min_i {d(p_i, P) + d(q_i, Q)}

    Parameters
    ----------

    target : Theano variable
        An array of shape (n, 6) representing the target,
    where each entry is a pair of 3-d points

    prediction : Theano variable
        An array of shape (6, ) representing the prediction,
    a pair of 3-d points

    Returns
    -------

    res : Theano variable
        A float representing the closest pair loss between the target and the prediction
    """

    # filter target entries
    target = target[T.nonzero(T.gt(target, -1))].reshape((-1, 6))
    
    prediction_a_sum_square = T.sum(prediction[:3] ** 2, axis=0, keepdims=True)
    prediction_b_sum_square = T.sum(prediction[3:] ** 2, axis=0, keepdims=True)
    target_a_sum_square = T.sum(target[:, :3] ** 2, axis=1, keepdims=True)
    target_b_sum_square = T.sum(target[:, 3:] ** 2, axis=1, keepdims=True)
    
    P_a = T.dot(prediction[:3], target[:, :3].T)
    P_b = T.dot(prediction[3:], target[:, 3:].T)
    
    D_a = T.sqrt(prediction_a_sum_square + target_a_sum_square.T - 2 * P_a)
    D_b = T.sqrt(prediction_b_sum_square + target_b_sum_square.T - 2 * P_b)
    D = D_a + D_b
    
    loss = D.min()
    
    return loss


def closest_pair_distance(target, prediction):
    """Return the closest pair distance  between the `target` and
    the `predicition`.
    See definition above

    Parameters
    ----------

    target : Theano variable
        An array of shape (n_samples, n, 6) representing the target,
    where each entry is a pair of 3-d points

    prediction : Theano variable
        An array of shape (n_samples, 6) representing the prediction,
    a pair of 3-d points

    Returns
    -------

    res : Theano variable
        An array of shape (n_samples, ) representing the closest pair
    loss between the target and the prediction
    """

    distances, updates = theano.map(
        _closest_pair_distance_vector,
        [target, prediction]
    )

    return distances.reshape((-1, 1))


def _modified_hausdorff_distance_matrix(target, prediction):
    """Return the Modified Hausdorff Distance between the `target` and
    the `predicition`.

    Parameters
    ----------

    target : Theano variable
        An array of shape (1, height, width) representing the target,
    where each point encodes the x and y coordinates and the value the z coordinate.

    prediction : Theano variable
        An array of shape (1, height, width) representing the target,
    where each point encodes the x and y coordinates and the value the z coordinate.

    Returns
    -------

    res : Theano variable
        A float representing the MHD between the target and the prediction
    """

    inf = numpy.float64(1e4)

    target = target.squeeze()
    max_target = T.max(target)
    indices = T.nonzero(target > 0)
    values = target[indices].reshape((-1, 1))
    indices = T.stack(indices, axis=1)
    target_indices = indices
    target = ifelse(
        max_target > 0,
        T.concatenate((indices, values), axis=1),
        T.zeros((2, 3), dtype=numpy.float64)
    )

    prediction = prediction.squeeze()
    n = prediction.shape[0]
    max_prediction = T.max(prediction)
    neg_i, neg_j = T.nonzero(T.le(prediction, 0.))
    prediction_neg_indices = n * neg_i + neg_j
    indices = T.repeat(T.arange(0, n), n), T.tile(T.arange(0, n), (n,))
    values = prediction[indices].reshape((-1, 1))
    indices = T.stack(indices, axis=1)
    prediction = T.concatenate((indices, values), axis=1)

    prediction_sum_square = T.sum(prediction ** 2, axis=1, keepdims=True)
    target_sum_square = T.sum(target ** 2, axis=1, keepdims=True)
    
    dot_prod = T.dot(prediction, target.T)

    distances = T.sqrt(prediction_sum_square + target_sum_square.T - 2 * dot_prod)
    distances = T.set_subtensor(distances[prediction_neg_indices], 0.)

    g_yz = distances.min(axis=1).mean()
    g_zy = distances.min(axis=0).mean()

    mhd = ifelse(
        max_target > 0,
        ifelse(
            max_prediction > 0,
            T.max([g_yz, g_zy]),
            target_indices.shape[0] * 2.0),
        ifelse(
            max_prediction > 0,
            (n ** 2 - prediction_neg_indices.shape[0]) * 2.0,
            numpy.float64(0.))
    )

    return mhd


def modified_hausdorff_distance(target, prediction):
    """Return the Modified Hausdorff Distance between the `target` and
    the `predicition`.

    Parameters
    ----------

    target : Theano variable
        An array of shape (n_samples, height, width) representing the target,
    where each point encodes the x and y coordinates and the value the z coordinate.

    prediction : Theano variable
        An array of shape (n_samples, height, width) representing the target,
    where each point encodes the x and y coordinates and the value the z coordinate.

    Returns
    -------

    res : Theano variable
        An array of shape (n_samples, ) representing the MHD between the 
    target and the prediction
    """
    
    print(target, prediction)
    distances, updates = theano.map(
        _modified_hausdorff_distance_matrix,
        [target, prediction]
    )

    return distances.reshape((-1, 1))
    

def binary_hinge_loss(target, prediction):
    """Return the binary hinge loss between the ``target`` and
    the ``prediction``.

    Parameters
    ----------

    target : Theano variable
        An array of arbitrary shape representing representing the targets.

    prediction : Theano variable
        An array of arbitrary shape representing representing the predictions.

    Returns
    -------

    res : Theano variable
        An array of the same shape as ``target`` and ``prediction``
        representing the binary hinge loss."""
    target = 2 * target - 1
    return T.nnet.relu(1 - prediction * target)


def absolute(target, prediction):
    """Return the element wise absolute difference between the ``target`` and
    the ``prediction``.

    Parameters
    ----------

    target : Theano variable
        An array of arbitrary shape representing representing the targets.

    prediction : Theano variable
        An array of arbitrary shape representing representing the predictions.

    Returns
    -------

    res : Theano variable
        An array of the same shape as ``target`` and ``prediction``
        representing the pairwise distances."""
    return abs(target - prediction)



def bin_ce(target, prediction, eps=1e-8):
    """Return the binary cross entropy between the ``target`` and the ``prediction``,
    where ``prediction`` is a summary of the statistics of a categorial
    distribution and ``target`` is a some outcome.

    Used for binary classification purposes.

    Parameters
    ----------

    target : Theano variable
        An array of shape ``(n, )`` where ``n`` is the number of samples.

    prediction : Theano variable
        An array of shape ``(n, )``.

    Returns
    -------

    res : Theano variable.
        An array of the same size as ``target`` and ``prediction`` representing
        the pairwise divergences."""
    prediction = T.clip(prediction, eps, 1 - eps)
    return T.nnet.binary_crossentropy(prediction, target)



def cat_ce(target, prediction, eps=1e-8):
    """Return the cross entropy between the ``target`` and the ``prediction``,
    where ``prediction`` is a summary of the statistics of a categorial
    distribution and ``target`` is a some outcome.

    Used for multiclass classification purposes.

    The loss is different to ``ncat_ce`` by that ``target`` is not
    an array of integers but a hot k coding.

    Note that predictions are clipped between ``eps`` and ``1 - eps`` to ensure
    numerical stability.

    Parameters
    ----------

    target : Theano variable
        An array of shape ``(n, k)`` where ``n`` is the number of samples and
        ``k`` is the number of classes. Each row represents a hot k
        coding. It should be zero except for one element, which has to be
        exactly one.

    prediction : Theano variable
        An array of shape ``(n, k)``. Each row is interpreted as a categorical
        probability. Thus, each row has to sum up to one and be non-negative.

    Returns
    -------

    res : Theano variable.
        An array of the same size as ``target`` and ``prediction`` representing
        the pairwise divergences."""
    prediction = T.clip(prediction, eps, 1 - eps)
    return -(target * T.log(prediction))


def ncat_ce(target, prediction):
    """Return the cross entropy between the ``target`` and the ``prediction``,
    where ``prediction`` is a summary of the statistics of the categorical
    distribution and ``target`` is a some outcome.

    Used for classification purposes.

    The loss is different to ``cat_ce`` by that ``target`` is not a hot
    k coding but an array of integers.

    Parameters
    ----------

    target : Theano variable
        An array of shape ``(n,)`` where `n` is the number of samples. Each
        entry of the array should be an integer between ``0`` and ``k-1``,
        where ``k`` is the number of classes.
    prediction : Theano variable
        An array of shape ``(n, k)`` or ``(t, n , k)``. Each row (i.e. entry in
        the last dimension) is interpreted as a categorical probability. Thus,
        each row has to sum up to one and be non-negative.


    Returns
    -------

    res : Theano variable
        An array of shape ``(n, 1)`` as ``target`` containing the log
        probability that that example is classified correctly."""

    # The following code might seem more complicated as necessary. Yet,
    # at the time of writing the gradient of AdvancedIncSubtensor does not run
    # on the GPU, which is why we reduce it to using AdvancedSubtensor.

    if prediction.ndim == 3:
        # We are looking at a 3D problem (e.g. via recurrent nets) and this
        # make it a 2D problem.
        target_flat = target.flatten()
        prediction_flat = prediction.flatten()
    elif prediction.ndim == 2:
        target_flat = target
        prediction_flat = prediction.flatten()
    else:
        raise ValueError('only 2 or 3 dims supported for nnce')
    target_flat.name = 'target_flat'
    prediction_flat.name = 'prediction_flat'

    target_flat += T.arange(target_flat.shape[0]) * prediction.shape[-1]

    # This cast needs to be explicit, because in the case of the GPU, the
    # targets will always be floats.
    target_flat = T.cast(target_flat, 'int32')
    loss = -T.log(prediction_flat)[target_flat]

    # In both forks below, a trailing 1 is added to the shape because that
    # is what the caller expects. (As it is e.g. with the squared error.)
    if prediction.ndim == 3:
        # Convert back from 2D to 3D.
        loss = loss.reshape((prediction.shape[0], prediction.shape[1], 1))
    elif prediction.ndim == 2:
        loss = loss.reshape((prediction.shape[0], 1))

    return loss


def bern_ces(target, prediction):
    """Return the Bernoulli cross entropies between binary vectors ``target``
    and a number of Bernoulli variables ``prediction``.

    Used in regression on binary variables, not classification.

    Parameters
    ----------

    target : Theano variable
        An array of shape ``(n, k)`` where ``n`` is the number of samples and k
        is the number of outputs. Each entry should be either 0 or 1.

    prediction : Theano variable.
        An array of shape ``(n, k)``. Each row is interpreted as a set of
        statistics of Bernoulli variables. Thus, each element has to lie in
        ``(0, 1)``.

    Returns
    -------

    res : Theano variable
        An array of the same size as ``target`` and ``prediction`` representing
        the pairwise divergences.
    """
    prediction *= 0.999
    prediction += 0.0005
    return -(target * T.log(prediction) + (1 - target) * T.log(1 - prediction))


def bern_bern_kl(X, Y):
    """Return the Kullback-Leibler divergence between Bernoulli variables
    represented by their sufficient statistics.

    Parameters
    ----------

    X : Theano variable
        An array of arbitrary shape where each element represents
        the statistic of a Bernoulli variable and thus should lie in
        ``(0, 1)``.
    Y : Theano variable
        An array of the same shape as ``target`` where each element represents
        the statistic of a Bernoulli variable and thus should lie in
        ``(0, 1)``.

    Returns
    -------

     res : Theano variable
        An array of the same size as ``target`` and ``prediction`` representing
        the pairwise divergences."""
    return X * T.log(X / Y) + (1 - X) * T.log((1 - X) / (1 - Y))


def ncac(target, embedding):
    """Return the NCA for classification loss.

    This corresponds to the probability that a point is correctly classified
    with a soft knn classifier using leave-one-out. Each neighbour is weighted
    according to an exponential of its negative Euclidean distance. Afterwards,
    a probability is calculated for each class depending on the weights of the
    neighbours. For details, we refer you to

    'Neighbourhood Component Analysis' by
    J Goldberger, S Roweis, G Hinton, R Salakhutdinov (2004).

    Parameters
    ----------

    target : Theano variable
        An array of shape ``(n,)`` where ``n`` is the number of samples. Each
        entry of the array should be an integer between ``0`` and ``k - 1``,
        where ``k`` is the number of classes.
    embedding : Theano variable
        An array of shape ``(n, d)`` where each row represents a point in``d``-dimensional space.

    Returns
    -------

    res : Theano variable
        Array of shape `(n, 1)` holding a probability that a point is
        classified correclty.
    """
    # Matrix of the distances of points.
    dist = distance_matrix(embedding)
    thisid = T.identity_like(dist)

    # Probability that a point is neighbour of another point based on
    # the distances.
    top = T.exp(-dist) + 1e-8       # Add a small constant for stability.
    bottom = (top - thisid * top).sum(axis=0)
    p = top / bottom

    # Create a matrix that matches same classes.
    sameclass = T.eq(distance_matrix(target), 0) - thisid
    loss_vector = -(p * sameclass).sum(axis=1)
    # To be compatible with the API, we make this a (n, 1) matrix.
    return T.shape_padright(loss_vector)


def ncar(target, embedding):
    """Return the NCA for regression loss.

    This is similar to NCA for classification, except that not soft KNN
    classification but regression performance is maximized. (Actually, the
    negative performance is minimized.)

    For details, we refer you to

    'Pose-sensitive embedding by nonlinear nca regression' by
    Taylor, G. and Fergus, R. and Williams, G. and Spiro, I. and Bregler, C.
    (2010)

    Parameters
    ----------

    target : Theano variable
        An array of shape ``(n, d)`` where ``n`` is the number of samples and
        ``d`` the dimensionalty of the target space.
    embedding : Theano variable
        An array of shape ``(n, d)`` where each row represents a point in
        ``d``-dimensional space.

    Returns
    -------

    res : Theano variable
        Array of shape ``(n, 1)``.
    """
    # Matrix of the distances of points.
    dist = distance_matrix(embedding) ** 2
    thisid = T.identity_like(dist)

    # Probability that a point is neighbour of another point based on
    # the distances.
    top = T.exp(-dist) + 1E-8  # Add a small constant for stability.
    bottom = (top - thisid * top).sum(axis=0)
    p = top / bottom

    # Create matrix of distances.
    target_distance = distance_matrix(target, target, 'soft_l1')
    # Set diagonal to 0.
    target_distance -= target_distance * T.identity_like(target_distance)

    loss_vector = (p * target_distance ** 2).sum(axis=1)
    # To be compatible with the API, we make this a (n, 1) matrix.
    return T.shape_padright(loss_vector)


def drlim(push_margin, pull_margin, c_contrastive,
          push_loss='squared', pull_loss='squared'):
    """Return a function that implements the

    'Dimensionality reduction by learning an invariant mapping' by
    Hadsell, R. and Chopra, S. and LeCun, Y. (2006).

    For an example of such a function, see `drlim1` with a margin of 1.

    Parameters
    ----------

    push_margin : Float
        The minimum margin that negative pairs should be seperated by.
        Pairs seperated by higher distance than push_margin will not
        contribute to the loss.

    pull_margin: Float
        The maximum margin that positive pairs may be seperated by.
        Pairs seperated by lower distances do not contribute to the loss.

    c_contrastive : Float
        Coefficient to weigh the contrastive term relative to the
        positive term

    push_loss : One of {'squared', 'absolute'}, optional, default: 'squared'
        Loss to encourage Euclidean distances between non pairs.

    pull_loss : One of {'squared', 'absolute'}, optional, default: 'squared'
        Loss to punish Euclidean distances between pairs.

    Returns
    -------

    loss : callable
        Function that takes two arguments, a target and an embedding."""

    # One might think that we'd need to use abs as the non-squared loss here.
    # Yet, due to the maximum operation later one we can just take the identity
    # as well.
    f_push_loss = T.square if push_loss == 'squared' else lambda x: x
    f_pull_loss = T.square if pull_loss == 'squared' else lambda x: x

    def inner(target, embedding):
        """Return a theano expression of a vector containing the sample wise
        loss of drlim.

        The push_margin, pull_margin and coefficient for the contrastives
        used are %.f, %.f and %.f respectively.

        Parameters
        ----------

        target : array_like
            A vector of length `n`. If 1, sample `2 * n` and sample
            `2 * n + 1` are deemed similar.

        embedding : array_like
            Array containing the embeddings of samples row wise.
        """ % (push_margin, pull_margin, c_contrastive)
        target = target[:, 0]
        n_pair = embedding.shape[0] // 2
        n_feature = embedding.shape[1]

        # Reshape array to get pairs.
        embedding = embedding.reshape((n_pair, n_feature * 2))

        # Calculate distances of pairs.
        diff = (embedding[:, :n_feature] - embedding[:, n_feature:])
        dist = T.sqrt((diff ** 2).sum(axis=1) + 1e-8)

        pull = target * f_pull_loss(T.maximum(0, dist - pull_margin))
        push = (1 - target) * f_push_loss(T.maximum(0, push_margin - dist))

        loss = pull + c_contrastive * push
        return loss.dimshuffle(0, 'x')

    return inner


drlim1 = drlim(1, 0, 0.5)


def ebm_loss(push_margin, pull_margin, c_contrastive):

    def inner(target, distance):
        
        push = target * T.sqr(T.maximum(0, push_margin - distance))
        pull = (1 - target) * T.sqr(T.maximum(0, distance - pull_margin))

        loss = pull + c_contrastive * push
        return loss.reshape((-1, 1))

    return inner


ebm_loss1 = ebm_loss(1, 0, 0.5)
ebm_loss2 = ebm_loss(2, 0, 0.5)
ebm_loss4 = ebm_loss(4, 0, 0.5)
