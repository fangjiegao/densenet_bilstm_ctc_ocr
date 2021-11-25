"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf
import densenet

slim = tf.contrib.slim

networks_map = {
    'densenet121': densenet.densenet121,
    'densenet161': densenet.densenet161,
    'densenet169': densenet.densenet169,
    'densenet121_fine_tuning': densenet.densenet121_fine_tuning,
    'densenet161_fine_tuning': densenet.densenet161_fine_tuning,
    'densenet169_fine_tuning': densenet.densenet169_fine_tuning,
}

arg_scopes_map = {
    'densenet121': densenet.densenet_arg_scope,
    'densenet161': densenet.densenet_arg_scope,
    'densenet169': densenet.densenet_arg_scope,
    'densenet121_fine_tuning': densenet.densenet_arg_scope,
    'densenet161_fine_tuning': densenet.densenet_arg_scope,
    'densenet169_fine_tuning': densenet.densenet_arg_scope,
}


def get_network_fn(name, num_classes, weight_decay=0.0, data_format='NHWC',
                   is_training=False):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay, data_format=data_format)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, data_format=data_format, is_training=is_training)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
