# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _mark_active_cells(
    cell_indices: wp.array2d[int],
    vertex_sdf: wp.array[float],
    active_cells: wp.array[int],
):
    """Mark a cell active when any corner has non-positive SDF (inside / on surface)."""
    i = wp.tid()
    vidx = cell_indices[i]

    min_sdf = float(1.0e8)
    for k in range(cell_indices.shape[1]):
        min_sdf = wp.min(min_sdf, vertex_sdf[vidx[k]])

    active_cells[i] = wp.where(min_sdf <= 0.0, 1, 0)


def _resolution_from_node_count(node_count: int) -> int:
    """Infer uniform Grid3D resolution from ``(res + 1)**3`` node SDF samples."""
    res_plus_one = int(round(node_count ** (1.0 / 3.0)))
    if res_plus_one**3 != node_count:
        raise ValueError(
            f"grid_sdf length {node_count} is not a perfect cube; expected (resolution + 1)**3 samples."
        )
    return res_plus_one - 1


def grid_cell_vertex_indices(resolution: int, device=None) -> wp.array:
    """Builds a lookup table for every hex cell on the background grid, which 8 grid nodes are its corners. It helps
    connect each of the grid nodes to the corresponding hex cell. 

    Args:
        resolution: Number of cells along each axis.
        device: Warp device for the returned array.

    Returns:
        A wp.array of shape [resolution**3, 8] with corner node indices.
    """
    res = int(resolution)
    n = res + 1

    # Corner offsets in (x, y, z) for the 8 hex vertices used by gen_hexmesh / Grid3D.
    corner_offsets = np.array(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ],
        dtype=np.int32,
    )

    cell_vtx = np.empty((res * res * res, 8), dtype=np.int32)
    cell = 0
    for x in range(res):
        for y in range(res):
            for z in range(res):
                for k, (dx, dy, dz) in enumerate(corner_offsets):
                    cell_vtx[cell, k] = (x + dx) * n * n + (y + dy) * n + (z + dz)
                cell += 1

    return wp.array(cell_vtx, dtype=int, device=device)


def find_active_cells(
    grid_sdf: wp.array,
    active_cells: wp.array,
    *,
    resolution: int | None = None,
    cell_vtx: wp.array | None = None,
    device=None,
) -> None:
    """Mark grid cells that intersect the interior of an SDF.

    Writes ``1`` into ``active_cells[i]`` when the minimum SDF at that cell's
    corner nodes is ``<= 0``, otherwise ``0``.

    Args:
        grid_sdf: Signed distance samples at grid nodes, shape ``[(res + 1)**3]``.
        active_cells: Output mask, shape ``[cell_count]``; mutated in place.
        resolution: Uniform grid resolution. Inferred from ``grid_sdf`` length when
            omitted and ``cell_vtx`` is not provided.
        cell_vtx: Optional ``[cell_count, 8]`` corner-index array. When omitted,
            indices are built for a uniform :class:`~warp.fem.Grid3D`.
        device: Warp device; defaults to ``grid_sdf.device``.
    """
    if device is None:
        device = grid_sdf.device

    if cell_vtx is None:
        if resolution is None:
            resolution = _resolution_from_node_count(int(grid_sdf.shape[0]))
        cell_vtx = grid_cell_vertex_indices(resolution, device=device)
    elif resolution is not None and cell_vtx.shape[0] != resolution**3:
        raise ValueError(
            f"cell_vtx has {cell_vtx.shape[0]} cells but resolution**3 = {resolution**3}."
        )

    if active_cells.shape[0] != cell_vtx.shape[0]:
        raise ValueError(
            f"active_cells has length {active_cells.shape[0]} but cell_vtx has "
            f"{cell_vtx.shape[0]} cells."
        )

    wp.launch(
        _mark_active_cells,
        dim=cell_vtx.shape[0],
        inputs=[cell_vtx, grid_sdf, active_cells],
        device=device,
    )
