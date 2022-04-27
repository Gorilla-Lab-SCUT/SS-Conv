# --------------------------------------------------------
# Sparse Steerable Convolution Lib.

# Operations on Sparse Tensors
# Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
# and https://github.com/dvlab-research/PointGroup
# --------------------------------------------------------

import torch
from ss_conv import _ext

def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] *
                (kernel_size[i] - 1) - 1) // stride[i] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation,
                            output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[
            i] + output_padding[i]
        output_size.append(size)
    return output_size


def get_indice_pairs(indices,
             batch_size,
             spatial_shape,
             ksize=3,
             stride=1,
             padding=0,
             dilation=1,
             out_padding=0,
             subm=False,
             transpose=False,
             grid=None):
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    for d, s in zip(dilation, stride):
        assert any([s == 1, d == 1]), "don't support this."

    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(spatial_shape, ksize, stride, padding,
                                            dilation, out_padding)
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride, padding,
                                            dilation)
    else:
        out_shape = spatial_shape

    if grid is None:
        if ndim == 2:
            get_indice_pairs_func = _ext.get_indice_pairs_2d
        elif ndim == 3:
            get_indice_pairs_func = _ext.get_indice_pairs_3d
        else:
            raise NotImplementedError

        return get_indice_pairs_func(indices, batch_size, out_shape, spatial_shape, ksize,
                            stride, padding, dilation, out_padding, int(subm), int(transpose))
    else:
        if ndim == 2:
            get_indice_pairs_func = _ext.get_indice_pairs_grid_2d
        elif ndim == 3:
            get_indice_pairs_func = _ext.get_indice_pairs_grid_3d
        else:
            raise NotImplementedError

        return get_indice_pairs_func(indices, grid, batch_size, out_shape, spatial_shape, ksize,
                        stride, padding, dilation, out_padding, int(subm), int(transpose))


def indice_conv_forward(features,
              filters,
              indice_pairs,
              indice_pair_num,
              num_activate_out,
              inverse=False,
              subm=False):
    if filters.dtype == torch.float32:
        return _ext.indice_conv_fp32(features, filters, indice_pairs,
                                               indice_pair_num, num_activate_out,
                                               int(inverse), int(subm))
    elif filters.dtype == torch.half:
        return _ext.indice_conv_half(features, filters, indice_pairs,
                                               indice_pair_num, num_activate_out,
                                               int(inverse), int(subm))
    else:
        raise NotImplementedError


def indice_conv_backward(features,
                       filters,
                       out_bp,
                       indice_pairs,
                       indice_pair_num,
                       inverse=False,
                       subm=False):
    if filters.dtype == torch.float32:
        return _ext.indice_conv_backward_fp32(
            features, filters, out_bp, indice_pairs, indice_pair_num, int(inverse), int(subm))
    elif filters.dtype == torch.half:
        return _ext.indice_conv_backward_half(
            features, filters, out_bp, indice_pairs, indice_pair_num, int(inverse), int(subm))
    else:
        raise NotImplementedError


def indice_avgpool_forward(features, indice_pairs, indice_pair_num, num_activate_out, summaryrf):
    if features.dtype == torch.float32:
        return _ext.indice_avgpool_fp32(features, indice_pairs, indice_pair_num,
                                                  num_activate_out, summaryrf)
    elif features.dtype == torch.half:
        return _ext.indice_avgpool_half(features, indice_pairs, indice_pair_num,
                                                  num_activate_out, summaryrf)
    else:
        raise NotImplementedError


def indice_avgpool_backward(features, out_features, out_bp, indice_pairs, indice_pair_num, summaryrf):
    if features.dtype == torch.float32:
        return _ext.indice_avgpool_backward_fp32(
            features, out_features, out_bp, indice_pairs, indice_pair_num, summaryrf)
    elif features.dtype == torch.half:
        return _ext.indice_avgpool_backward_half(
            features, out_features, out_bp, indice_pairs, indice_pair_num, summaryrf)
    else:
        raise NotImplementedError



def indice_maxpool_forward(features, indice_pairs, indice_pair_num, num_activate_out):
    if features.dtype == torch.float32:
        return _ext.indice_maxpool_fp32(features, indice_pairs, indice_pair_num,
                                                  num_activate_out)
    elif features.dtype == torch.half:
        return _ext.indice_maxpool_half(features, indice_pairs, indice_pair_num,
                                                  num_activate_out)
    else:
        raise NotImplementedError


def indice_maxpool_backward(features, out_features, out_bp, indice_pairs, indice_pair_num):
    if features.dtype == torch.float32:
        return _ext.indice_maxpool_backward_fp32(
            features, out_features, out_bp, indice_pairs, indice_pair_num)
    elif features.dtype == torch.half:
        return _ext.indice_maxpool_backward_half(
            features, out_features, out_bp, indice_pairs, indice_pair_num)
    else:
        raise NotImplementedError


def indice_field_maxpool_forward(features, indice_pairs, indice_pair_num, num_activate_out):
    if features.dtype == torch.float32:
        return _ext.indice_field_maxpool_fp32(features, indice_pairs, indice_pair_num,
                                                  num_activate_out)
    elif features.dtype == torch.half:
        return _ext.indice_field_maxpool_half(features, indice_pairs, indice_pair_num,
                                                  num_activate_out)
    else:
        raise NotImplementedError


def indice_field_maxpool_backward(features, indice_pairs, indice_pair_num, num_activate_out, feature_norms):
    if features.dtype == torch.float32:
        return _ext.indice_maxpool_backward_fp32(
            features, indice_pairs, indice_pair_num, num_activate_out, feature_norms)
    elif features.dtype == torch.half:
        return _ext.indice_maxpool_backward_half(
            features, indice_pairs, indice_pair_num, num_activate_out, feature_norms)
    else:
        raise NotImplementedError