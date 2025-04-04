import numpy as np

from groupy.garray.D4_array import D4
from p4g import P4GFuncArray
from groupy.gconv.make_gconv_indices import flatten_indices, make_d4_z2_indices

def make_d4_p4g_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(8, ksize, ksize)
    f = P4GFuncArray(v=x)
    li = f.left_translation_indices(D4.flatten()[:, None, None, None])
    return li.astype('int32')

def p4gconv2d_util(h_input, in_channels, out_channels, ksize):
    """
    Convenience function for setting up static data required for the G-Conv.
     This function returns:
      1) an array of indices used in the filter transformation step of gconv2d
      2) shape information required by gconv2d
      5) the shape of the filter tensor to be allocated and passed to gconv2d
    :param h_input: one of ('Z2', 'C4', 'D4'). Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
    :param h_output: one of ('Z2', 'C4', 'D4'). What kind of transformations to use (rotations or roto-reflections).
      The choice of h_output of one layer should equal h_input of the next layer.
    :param in_channels: the number of input channels. Note: this refers to the number of (3D) channels on the group.
    The number of 2D channels will be 1, 4, or 8 times larger, depending the value of h_input.
    :param out_channels: the number of output channels. Note: this refers to the number of (3D) channels on the group.
    The number of 2D channels will be 1, 4, or 8 times larger, depending on the value of h_output.
    :param ksize: the spatial size of the filter kernels (typically 3, 5, or 7).
    :return: gconv_indices
    """


    if h_input == 'Z2':
        gconv_indices = flatten_indices(make_d4_z2_indices(ksize=ksize))
        nti = 1
        nto = 8
    elif h_input == 'D4':
        gconv_indices = flatten_indices(make_d4_p4g_indices(ksize=ksize))
        nti = 8
        nto = 8

    else:
        raise ValueError('Unknown (h_input, h_output) pair:' + str((h_input, 'D4')))


    w_shape = (ksize, ksize, in_channels * nti, out_channels)

    gconv_shape_info = (out_channels, nto, in_channels, nti, ksize)
    return gconv_indices, gconv_shape_info, w_shape
