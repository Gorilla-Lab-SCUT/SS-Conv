# --------------------------------------------------------
# Sparse Steerable Convolution Lib.

# Functions on Sparse Tensors
# Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
# and https://github.com/dvlab-research/PointGroup
# --------------------------------------------------------
import numpy as np
from typing import Tuple

import torch
from torch.autograd import Function

from ss_conv import _ext
from .ops import indice_conv_forward, indice_conv_backward
from .ops import indice_maxpool_forward, indice_maxpool_backward
from .ops import indice_field_maxpool_forward, indice_field_maxpool_backward
from .ops import indice_avgpool_forward, indice_avgpool_backward


class SparseConvFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out):
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            filters)
        return indice_conv_forward(features, filters, indice_pairs, indice_pair_num, num_activate_out, False)

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = indice_conv_backward(features, filters, grad_output.contiguous(), indice_pairs, indice_pair_num, False)

        return input_bp, filters_bp, None, None, None


class SparseInverseConvFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out):
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            filters)
        return indice_conv_forward(features, filters, indice_pairs, indice_pair_num, num_activate_out, True, False)

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = indice_conv_backward(features, filters, grad_output.contiguous(), indice_pairs, indice_pair_num, True, False)

        return input_bp, filters_bp, None, None, None


class SubMConvFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out):
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            filters)
        return indice_conv_forward(features, filters, indice_pairs, indice_pair_num, num_activate_out, False, True)

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = indice_conv_backward(features, filters, grad_output.contiguous(), indice_pairs, indice_pair_num, False, True)

        return input_bp, filters_bp, None, None, None


class SparseMaxPoolFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            indice_pairs,
            indice_pair_num,
            num_activate_out):
        out = indice_maxpool_forward(features, indice_pairs, indice_pair_num, num_activate_out)
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            out)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out = ctx.saved_tensors
        input_bp = indice_maxpool_backward(features, out, grad_output.contiguous(), indice_pairs, indice_pair_num)
        return input_bp, None, None, None


class SparseFieldMaxPoolFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            feature_norms):
        out = indice_field_maxpool_forward(features, indice_pairs, indice_pair_num, num_activate_out, feature_norms)
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            out)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out = ctx.saved_tensors
        input_bp = indice_field_maxpool_backward(features, out, grad_output.contiguous(), indice_pairs, indice_pair_num)
        return input_bp, None, None, None, None


class SparseAvgPoolFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            indice_pairs,
            indice_pair_num,
            num_activate_out):
        summaryrf = _ext.get_indice_summaryrf(indice_pairs, indice_pair_num, num_activate_out)
        out = indice_avgpool_forward(features, indice_pairs, indice_pair_num, num_activate_out, summaryrf)
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            out,
            summaryrf)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out, summaryrf = ctx.saved_tensors
        input_bp = indice_avgpool_backward(features, out, grad_output.contiguous(), indice_pairs, indice_pair_num, summaryrf)
        return input_bp, None, None, None

class SparseAvgPoolFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            use_gs=True):
        if not use_gs:
            summaryrf = _ext.indiceSummaryRF(indice_pairs, indice_pair_num, num_activate_out)
            # print(summaryrf.shape, "  ", summaryrf.dtype)
        else:
            kernel_volume = indice_pairs.shape[0]
            summaryrf = indice_pairs.new_zeros(num_activate_out) + int(kernel_volume)

        out = indice_avgpool_forward(features, indice_pairs, indice_pair_num, num_activate_out, summaryrf)
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            out,
            summaryrf)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out, summaryrf = ctx.saved_tensors
        input_bp = indice_avgpool_backward(features, out, grad_output.contiguous(), indice_pairs, indice_pair_num, summaryrf)
        return input_bp, None, None, None, None


class GlobalMaxPool(Function):
    @staticmethod
    def forward(ctx, feats, proposals_offset):
        '''
        :param ctx:
        :param feats: (sumNPoint, C) float
        :param proposals_offset: (nProposal + 1) int
        :return: output_feats (nProposal, C) float
        '''
        nProposal = proposals_offset.size(0) - 1
        sumNPoint, C = feats.size()

        assert feats.is_contiguous()
        assert proposals_offset.is_contiguous()

        output_feats = torch.cuda.FloatTensor(nProposal, C).zero_()
        output_maxidx = torch.cuda.IntTensor(nProposal, C).zero_()

        _ext.global_maxpool_fp(feats, proposals_offset, output_feats, output_maxidx, nProposal, C)

        ctx.for_backwards = (output_maxidx, proposals_offset, sumNPoint)

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        nProposal, C = d_output_feats.size()

        output_maxidx, proposals_offset, sumNPoint = ctx.for_backwards

        d_feats = torch.cuda.FloatTensor(sumNPoint, C).zero_()

        _ext.global_maxpool_bp(d_feats, proposals_offset, output_maxidx, d_output_feats.contiguous(), nProposal, C)

        return d_feats, None


class GlobalAvgPool(Function):
    @staticmethod
    def forward(ctx, feats, proposals_offset):
        '''
        :param ctx:
        :param feats: (sumNPoint, C) float
        :param proposals_offset: (nProposal + 1) int
        :return: output_feats (nProposal, C) float
        '''
        nProposal = proposals_offset.size(0) - 1
        sumNPoint, C = feats.size()

        assert feats.is_contiguous()
        assert proposals_offset.is_contiguous()

        output_feats = torch.cuda.FloatTensor(nProposal, C).zero_()
        output_maxidx = torch.cuda.IntTensor(nProposal, C).zero_()

        _ext.global_avgpool_fp(feats, proposals_offset, output_feats, output_maxidx, nProposal, C)

        ctx.for_backwards = (output_maxidx, proposals_offset, sumNPoint)

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        nProposal, C = d_output_feats.size()

        output_maxidx, proposals_offset, sumNPoint = ctx.for_backwards

        d_feats = torch.cuda.FloatTensor(sumNPoint, C).zero_()

        _ext.global_avgpool_bp(d_feats, proposals_offset, output_maxidx, d_output_feats.contiguous(), nProposal, C)

        return d_feats, None


class Voxelization_Idx(Function):
    @staticmethod
    def forward(ctx, coords, batchsize, mode=4):
        '''
        :param ctx:
        :param coords:  long (N, dimension + 1) or (N, dimension) dimension = 3
        :param batchsize
        :param mode: int 4=mean
        :param dimension: int
        :return: output_coords:  long (M, dimension + 1) (M <= N)
        :return: output_map: int M * (maxActive + 1)
        :return: input_map: int N
        '''
        assert coords.is_contiguous()
        N = coords.size(0)
        output_coords = coords.new()
        device = coords.device

        input_map = torch.IntTensor(N).zero_().to(device)
        output_map = input_map.new()

        _ext.voxelize_idx(coords, output_coords, input_map, output_map, batchsize, mode)
        return output_coords, input_map, output_map


    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None


class Voxelization(Function):
    @staticmethod
    def forward(ctx, feats, map_rule, mode=4):
        '''
        :param ctx:
        :param map_rule: cuda int M * (maxActive + 1)
        :param feats: cuda float N * C
        :return: output_feats: cuda float M * C
        '''
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        N, C = feats.size()
        M = map_rule.size(0)
        maxActive = map_rule.size(1) - 1

        output_feats = torch.cuda.FloatTensor(M, C).zero_()

        ctx.for_backwards = (map_rule, mode, maxActive, N)
        _ext.voxelize_fp(feats, output_feats, map_rule, mode, M, maxActive, C)
        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, mode, maxActive, N = ctx.for_backwards
        M, C = d_output_feats.size()

        d_feats = torch.cuda.FloatTensor(N, C).zero_()

        _ext.voxelize_bp(d_output_feats.contiguous(), d_feats, map_rule, mode, M, maxActive, C)
        return d_feats, None, None


class Voxel2Point_Map(Function):
    @staticmethod
    def forward(ctx, inverse, nActive, maxActive, mode=4):
        '''
        :param ctx:
        :param inverse:  int (nInputPoint, )
        :param nActive:  int
        :param maxActive: int
        :param mode: int [1:first; 2:last; 3:max; 4: mean]
        :return: output_maps: int nActive * (maxActive + 1)
        '''
        inverse = inverse.view(-1).contiguous().int()
        nInputPoint = inverse.size(0)

        ids = torch.arange(nInputPoint).int().to(inverse.device)
        if mode == 3 or mode == 4:
            output_maps = torch.zeros(nActive, maxActive+1).int().to(inverse.device)
        elif mode == 2:
            output_maps = torch.ones(nActive, maxActive+1).int().to(inverse.device) * -1
        elif mode == 1:
            output_maps = torch.ones(nActive, maxActive+1).int().to(inverse.device) * nInputPoint
        else:
            assert False


        _ext.voxel2point_map(inverse, ids, output_maps, nInputPoint, maxActive, mode)
        return output_maps

    @staticmethod
    def backward(ctx, grad=None):
        return None, None, None, None

class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (N, 3)
        :param known: (M, 3)
        :return:
            dist: (N, 3) l2 distance to the three nearest neighbors
            idx: (N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()
        device = unknown.device

        N, _ = unknown.size()
        m = known.size(0)
        dist2 = torch.FloatTensor(N, 3).to(device)
        idx = torch.IntTensor(N, 3).to(device)

        _ext.three_nn(N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (M, C) Features descriptors to be interpolated from
        :param idx: (n, 3) three nearest neighbors of the target features in features
        :param weight: (n, 3) weights
        :return:
            output: (N, C) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()
        device = features.device

        m, c = features.size()
        n = idx.size(0)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.FloatTensor(n, c).to(device)

        _ext.three_interpolate_fp(c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (N, C) tensor with gradients of outputs
        :return:
            grad_features: (M, C) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        n, c = grad_out.size()
        device = grad_out.device

        grad_features = torch.FloatTensor(m, c).zero_().to(device)
        grad_out_data = grad_out.data.contiguous()

        _ext.three_interpolate_bp( c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None





indice_conv = SparseConvFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_maxpool = SparseMaxPoolFunction.apply
indice_fieldmaxpool = SparseFieldMaxPoolFunction.apply
indice_avgpool = SparseAvgPoolFunction.apply
global_maxpool = GlobalMaxPool.apply
global_avgpool = GlobalAvgPool.apply
voxelization_idx = Voxelization_Idx.apply
voxelization = Voxelization.apply
voxel2point_mapping = Voxel2Point_Map.apply
three_nn = ThreeNN.apply
three_interpolate = ThreeInterpolate.apply
