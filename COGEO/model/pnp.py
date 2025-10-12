# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Differentiable Efficient PnP utilities adapted for the CQ project."""
from __future__ import annotations

import warnings
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like torch.linalg.eigh, assuming the argument is a symmetric real matrix."""
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "eigh"):
        return torch.linalg.eigh(A)
    return torch.symeig(A, eigenvectors=True)


class SimilarityTransform(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    s: torch.Tensor


AMBIGUOUS_ROT_SINGULAR_THR = 1e-15


def corresponding_points_alignment(
    X: torch.Tensor,
    Y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    eps: float = 1e-9,
) -> SimilarityTransform:
    if X.shape != Y.shape:
        raise ValueError(
            "Point sets X and Y must share the same batch, point count and dims."
        )
    if weights is not None and X.shape[:2] != weights.shape:
        raise ValueError("weights must share the first two dims with X")

    Xt = X
    Yt = Y
    num_points = torch.full(
        (Xt.shape[0],), Xt.shape[1], device=Xt.device, dtype=torch.int64
    )

    if weights is not None:
        if any(
            xd != wd and xd != 1 and wd != 1
            for xd, wd in zip(Xt.shape[-2::-1], weights.shape[::-1])
        ):
            raise ValueError("weights are not compatible with the tensor")

    Xmu = wmean(Xt, weight=weights, eps=eps)
    Ymu = wmean(Yt, weight=weights, eps=eps)

    Xc = Xt - Xmu
    Yc = Yt - Ymu

    total_weight = torch.clamp(num_points, 1)
    if weights is not None:
        Xc *= weights[:, :, None]
        Yc *= weights[:, :, None]
        total_weight = torch.clamp(weights.sum(1), eps)

    if (num_points < (Xt.shape[2] + 1)).any():
        warnings.warn(
            "Point cloud size <= dim+1. Alignment may not recover unique rotation."
        )

    XYcov = torch.bmm(Xc.transpose(2, 1), Yc)
    XYcov = XYcov / total_weight[:, None, None]

    U, S, V = torch.svd(XYcov)

    if (S.abs() <= AMBIGUOUS_ROT_SINGULAR_THR).any() and not (
        num_points < (Xt.shape[2] + 1)
    ).any():
        warnings.warn(
            "Cross-correlation has low rank - rotation may be ambiguous."
        )

    E = torch.eye(Xt.shape[2], dtype=XYcov.dtype, device=XYcov.device)[None].repeat(
        Xt.shape[0], 1, 1
    )

    if not allow_reflection:
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        Xcov = (Xc * Xc).sum((1, 2)) / total_weight
        s = trace_ES / torch.clamp(Xcov, eps)
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]
        s = T.new_ones(Xt.shape[0])

    return SimilarityTransform(R, T, s)


def wmean(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    dim: Union[int, Tuple[int, ...]] = -2,
    keepdim: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    args = {"dim": dim, "keepdim": keepdim}

    if weight is None:
        return x.mean(**args)

    if any(
        xd != wd and xd != 1 and wd != 1
        for xd, wd in zip(x.shape[-2::-1], weight.shape[::-1])
    ):
        raise ValueError("wmean: weights are not compatible with the tensor")

    return (x * weight[..., None]).sum(**args) / weight[..., None].sum(**args).clamp(
        eps
    )


class EpnpSolution(NamedTuple):
    x_cam: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    err_2d: torch.Tensor
    err_3d: torch.Tensor


def _define_control_points(x, weight, storage_opts=None):
    storage_opts = storage_opts or {}
    x_mean = wmean(x, weight)
    c_world = F.pad(torch.eye(3, **storage_opts), (0, 0, 0, 1), value=0.0).expand_as(
        x[:, :4, :]
    )
    return c_world + x_mean


def _compute_alphas(x, c_world):
    x = F.pad(x, (0, 1), value=1.0)
    c = F.pad(c_world, (0, 1), value=1.0)
    return torch.matmul(x, torch.inverse(c))


def _build_M(y, alphas, weight):
    bs, n, _ = y.size()

    def prepad(t, v):
        return F.pad(t, (1, 0), value=v)

    if weight is not None:
        alphas = alphas * weight[:, :, None]

    def lm_alphas(t):
        return torch.matmul(alphas[..., None], t).reshape(bs, n, 12)

    M = torch.cat(
        (
            lm_alphas(prepad(prepad(-y[:, :, 0, None, None], 0.0), 1.0)),
            lm_alphas(prepad(prepad(-y[:, :, 1, None, None], 1.0), 0.0)),
        ),
        dim=-1,
    ).reshape(bs, -1, 12)

    return M


def _null_space(m, kernel_dim):
    mTm = torch.bmm(m.transpose(1, 2), m)
    s, v = eigh(mTm)
    return v[:, :, :kernel_dim].reshape(-1, 4, 3, kernel_dim), s[:, :kernel_dim]


def _reproj_error(y_hat, y, weight, eps=1e-9):
    y_hat = y_hat / torch.clamp(y_hat[..., 2:], eps)
    dist = ((y - y_hat[..., :2]) ** 2).sum(dim=-1, keepdim=True) ** 0.5
    return wmean(dist, weight)[:, 0, 0]


def _algebraic_error(x_w_rotated, x_cam, weight):
    dist = ((x_w_rotated - x_cam) ** 2).sum(dim=-1, keepdim=True)
    return wmean(dist, weight)[:, 0, 0]


def _compute_norm_sign_scaling_factor(c_cam, alphas, x_world, y, weight, eps=1e-9):
    x_cam = torch.matmul(alphas, c_cam)

    x_cam = x_cam * (1.0 - 2.0 * (wmean(x_cam[..., 2:], weight) < 0).float())
    if torch.any(x_cam[..., 2:] < -eps):
        neg_rate = wmean((x_cam[..., 2:] < 0).float(), weight, dim=(0, 1)).item()
        warnings.warn(f"EPnP: {neg_rate * 100.0:2.2f}% points have z<0.")

    R, T, s = corresponding_points_alignment(
        x_world, x_cam, weight, estimate_scale=True
    )
    s = s.clamp(eps)
    x_cam = x_cam / s[:, None, None]
    T = T / s[:, None]
    x_w_rotated = torch.matmul(x_world, R) + T[:, None, :]
    err_2d = _reproj_error(x_w_rotated, y, weight)
    err_3d = _algebraic_error(x_w_rotated, x_cam, weight)

    return EpnpSolution(x_cam, R, T, err_2d, err_3d)


def _gen_pairs(input, dim=-2, reducer=lambda a, b: ((a - b) ** 2).sum(dim=-1)):
    n = input.size()[dim]
    idx = torch.combinations(torch.arange(n, device=input.device)).long()
    left = input.index_select(dim, idx[:, 0])
    right = input.index_select(dim, idx[:, 1])
    return reducer(left, right)


def _kernel_vec_distances(v):
    dv = _gen_pairs(v, dim=-3, reducer=lambda a, b: a - b)
    rows_2ij = 2.0 * _gen_pairs(dv, dim=-1, reducer=lambda a, b: (a * b).sum(dim=-2))
    rows_ii = (dv ** 2).sum(dim=-2)
    return torch.cat((rows_ii, rows_2ij), dim=-1)


def _solve_lstsq_subcols(rhs, lhs, lhs_col_idx):
    lhs = lhs.index_select(-1, torch.tensor(lhs_col_idx, device=lhs.device).long())
    return torch.matmul(torch.pinverse(lhs), rhs[:, :, None])


def _binary_sign(t):
    return (t >= 0).to(t) * 2.0 - 1.0


def _find_null_space_coords_1(kernel_dsts, cw_dst, eps=1e-9):
    beta = _solve_lstsq_subcols(cw_dst, kernel_dsts, [0, 4, 5, 6])
    beta = beta * _binary_sign(beta[:, :1, :])
    return beta / torch.clamp(beta[:, :1, :] ** 0.5, eps)


def _find_null_space_coords_2(kernel_dsts, cw_dst):
    beta = _solve_lstsq_subcols(cw_dst, kernel_dsts, [0, 4, 1])

    coord_0 = (beta[:, :1, :].abs() ** 0.5) * _binary_sign(beta[:, 1:2, :])
    coord_1 = (beta[:, 2:3, :].abs() ** 0.5) * (
        (beta[:, :1, :] >= 0) == (beta[:, 2:3, :] >= 0)
    ).float()

    return torch.cat((coord_0, coord_1, torch.zeros_like(beta[:, :2, :])), dim=1)


def _find_null_space_coords_3(kernel_dsts, cw_dst, eps=1e-9):
    beta = _solve_lstsq_subcols(cw_dst, kernel_dsts, [0, 4, 1, 5, 7])

    coord_0 = (beta[:, :1, :].abs() ** 0.5) * _binary_sign(beta[:, 1:2, :])
    coord_1 = (beta[:, 2:3, :].abs() ** 0.5) * (
        (beta[:, :1, :] >= 0) == (beta[:, 2:3, :] >= 0)
    ).float()
    coord_2 = beta[:, 3:4, :] / torch.clamp(coord_0[:, :1, :], eps)

    return torch.cat(
        (coord_0, coord_1, coord_2, torch.zeros_like(beta[:, :1, :])), dim=1
    )


def _solve_quadratic_eq(a, b, c, eps=1e-9):
    delta = (b ** 2 - 4 * a * c).clamp(min=0)
    sqrt_delta = delta ** 0.5
    return torch.stack(
        (
            (-b + sqrt_delta) / torch.clamp(2 * a, eps),
            (-b - sqrt_delta) / torch.clamp(2 * a, eps),
        ),
        dim=-1,
    )


def _solve_cubic_eq(a, b, c, d, eps=1e-9):
    a, b, c, d = a / a[..., :1], b / a[..., :1], c / a[..., :1], d / a[..., :1]
    p = (3 * a * c - b ** 2) / 3
    q = (2 * b ** 3 - 9 * a * b * c + 27 * (a ** 2) * d) / 27
    delta = (q ** 2) / 4 + (p ** 3) / 27

    sqrt_delta = delta.clamp(min=0) ** 0.5
    t1 = (-q / 2) + sqrt_delta
    t2 = (-q / 2) - sqrt_delta
    u = t1.abs() ** (1.0 / 3.0) * _binary_sign(t1)
    v = t2.abs() ** (1.0 / 3.0) * _binary_sign(t2)

    roots = torch.stack(
        (
            u + v,
            -1 / 2 * (u + v) + np.sqrt(3) / 2 * (u - v) * 1j,
            -1 / 2 * (u + v) - np.sqrt(3) / 2 * (u - v) * 1j,
        ),
        dim=-1,
    )

    roots = roots - b[..., None] / (3 * a[..., None])
    real_roots = roots.real.abs() * (roots.imag.abs() <= eps)
    return real_roots


def _estimate_scale_and_r(cws, cs, idx, eps=1e-9):
    def _gen_pairs_idx(idx):
        d = idx.size(-1)
        idx_i, idx_j = torch.meshgrid(
            (torch.arange(d, device=idx.device),) * 2, indexing="xy"
        )
        idx_i, idx_j = idx_i.reshape(-1), idx_j.reshape(-1)
        mask = idx_i < idx_j
        return idx[..., idx_i[mask]], idx[..., idx_j[mask]]

    idx_i, idx_j = _gen_pairs_idx(idx)

    cws_norm, cws_proj = cws.norm(dim=-1), (cws[..., None, :] * idx).sum(dim=-1)
    cs_norm, cs_proj = cs.norm(dim=-1), (cs[..., None, :] * idx).sum(dim=-1)

    mcws_norm = cws_norm[..., idx_i], cws_norm[..., idx_j]
    mcs_norm = cs_norm[..., idx_i], cs_norm[..., idx_j]
    mcws_proj = cws_proj[..., idx_i], cws_proj[..., idx_j]
    mcs_proj = cs_proj[..., idx_i], cs_proj[..., idx_j]

    scale = torch.sqrt(
        torch.clamp(
            (mcws_norm[0] ** 2 - mcws_norm[1] ** 2)
            / torch.clamp(mcs_norm[0] ** 2 - mcs_norm[1] ** 2, eps),
            eps,
        )
    )
    scale = scale.mean(dim=-1)

    idx_norm = idx.norm(dim=-1, keepdim=True)
    dir_cws = cws_proj / torch.clamp(idx_norm, eps)
    dir_cs = cs_proj / torch.clamp(idx_norm * scale[..., None], eps)
    dir_vec = (dir_cws - dir_cs).mean(dim=-1)

    return scale, dir_vec


def _closest_rotation_matrix(R):
    U, _, Vh = torch.linalg.svd(R)
    R_closest = torch.matmul(U, Vh)
    det = torch.det(R_closest)
    det_mask = det < 0
    if det_mask.any():
        Vh[det_mask, -1, :] *= -1
        R_closest = torch.matmul(U, Vh)
    return R_closest


def efficient_pnp(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    skip_quadratic_eq: bool = False,
) -> EpnpSolution:
    if x.ndim != 3 or x.size(-1) != 3:
        raise ValueError("x must be of shape (B, N, 3)")
    if y.ndim != 3 or y.size(-1) != 2:
        raise ValueError("y must be of shape (B, N, 2)")
    if x.shape[:2] != y.shape[:2]:
        raise ValueError("x and y must share batch size and point count")

    c_world = _define_control_points(
        x.detach(), weights, storage_opts={"dtype": x.dtype, "device": x.device}
    )
    alphas = _compute_alphas(x, c_world)
    M = _build_M(y, alphas, weights)
    kernel, spectrum = _null_space(M, 4)

    c_world_distances = _gen_pairs(c_world)
    kernel_dsts = _kernel_vec_distances(kernel)

    betas = (
        []
        if skip_quadratic_eq
        else [
            fnsc(kernel_dsts, c_world_distances)
            for fnsc in [
                _find_null_space_coords_1,
                _find_null_space_coords_2,
                _find_null_space_coords_3,
            ]
        ]
    )

    c_cam_variants = [kernel] + [
        torch.matmul(kernel, beta[:, None, :, :]) for beta in betas
    ]

    solutions = [
        _compute_norm_sign_scaling_factor(c_cam[..., 0], alphas, x, y, weights)
        for c_cam in c_cam_variants
    ]

    sol_zipped = EpnpSolution(*(torch.stack(list(col)) for col in zip(*solutions)))
    best = torch.argmin(sol_zipped.err_2d, dim=0)

    def gather1d(source, idx):
        return source.gather(
            0,
            idx.reshape(1, -1, *([1] * (len(source.shape) - 2))).expand_as(source[:1]),
        )[0]

    gathered = [gather1d(sol_col, best) for sol_col in sol_zipped]
    x_cam_best, R_best, T_best, err2d_best, err3d_best = gathered
    R_best = _closest_rotation_matrix(R_best)
    return EpnpSolution(x_cam_best, R_best, T_best, err2d_best, err3d_best)
