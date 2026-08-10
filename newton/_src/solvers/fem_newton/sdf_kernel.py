import warp as wp


@wp.kernel
def mesh_sdf_kernel(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    sdf: wp.array[float],
):
    """Builds an SDF using mesh closest-point queries."""
    i = wp.tid()
    pos = points[i]

    max_dist = 1.0
    # accuracy / threshold have Warp defaults (2.0 / 0.5); pass them for type stubs
    query = wp.mesh_query_point_sign_winding_number(mesh, pos, max_dist, 2.0, 0.5)

    if query.result:
        mesh_pos = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        sdf[i] = query.sign * wp.length(pos - mesh_pos)
    else:
        sdf[i] = 1.0
