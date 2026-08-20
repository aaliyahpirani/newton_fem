# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody FEM Self Collision
#
# Experimental classic FEM Newton soft body. A hollow soft arch (outer walls
# only) stands inside the solver's [-1, 1]^3 background grid and collapses
# under gravity onto the ground so the two legs meet, exercising mesh
# self-collision contacts.
#
# Command: uv run -m newton.examples softbody_fem_self_collision
#
###########################################################################

import warp as wp

import newton
import newton.examples


def _add_soft_hollow_arch(
    builder: newton.ModelBuilder,
    *,
    pos: wp.vec3,
    cell: float,
    dim_x: int,
    dim_y: int,
    dim_z: int,
    arm_cells: int,
    top_cells: int,
    wall_cells: int,
    density: float,
    k_mu: float,
    k_lambda: float,
    k_damp: float,
    particle_radius: float | None = None,
) -> None:
    """Add a voxelized hollow arch (inverted U, outer walls only) as tetrahedra.

    Included cells form the left and right legs plus the top bridge. Interior
    cells of those members are skipped so the arch is a shell, not a solid
    fill. Each included hex is split into 5 tets, matching
    :meth:`ModelBuilder.add_soft_grid`.
    """
    mass = cell * cell * cell * density

    def cell_in_arch(x: int, y: int, z: int) -> bool:
        if x < 0 or x >= dim_x or y < 0 or y >= dim_y or z < 0 or z >= dim_z:
            return False
        return x < arm_cells or x >= dim_x - arm_cells or z >= dim_z - top_cells

    def cell_on_shell(x: int, y: int, z: int) -> bool:
        if not cell_in_arch(x, y, z):
            return False
        # Keep cells within wall_cells of the arch exterior (Chebyshev), matching
        # the hollow-cube face test: interior of each member is empty.
        for dz in range(-wall_cells, wall_cells + 1):
            for dy in range(-wall_cells, wall_cells + 1):
                for dx in range(-wall_cells, wall_cells + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if not cell_in_arch(x + dx, y + dy, z + dz):
                        return True
        return False

    used: set[tuple[int, int, int]] = set()
    for z in range(dim_z):
        for y in range(dim_y):
            for x in range(dim_x):
                if not cell_on_shell(x, y, z):
                    continue
                for dz in (0, 1):
                    for dy in (0, 1):
                        for dx in (0, 1):
                            used.add((x + dx, y + dy, z + dz))

    index_of: dict[tuple[int, int, int], int] = {}
    for iz in range(dim_z + 1):
        for iy in range(dim_y + 1):
            for ix in range(dim_x + 1):
                key = (ix, iy, iz)
                if key not in used:
                    continue
                index_of[key] = builder.particle_count
                p = pos + wp.vec3(ix * cell, iy * cell, iz * cell)
                builder.add_particle(p, wp.vec3(0.0, 0.0, 0.0), mass, radius=particle_radius)

    faces: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    def add_face(i: int, j: int, k: int) -> None:
        key = tuple(sorted((i, j, k)))
        if key not in faces:
            faces[key] = (i, j, k)
        else:
            del faces[key]

    def add_tet(i: int, j: int, k: int, l: int) -> None:
        builder.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)
        add_face(i, k, j)
        add_face(j, k, l)
        add_face(i, j, l)
        add_face(i, l, k)

    for z in range(dim_z):
        for y in range(dim_y):
            for x in range(dim_x):
                if not cell_on_shell(x, y, z):
                    continue
                v0 = index_of[(x, y, z)]
                v1 = index_of[(x + 1, y, z)]
                v2 = index_of[(x + 1, y, z + 1)]
                v3 = index_of[(x, y, z + 1)]
                v4 = index_of[(x, y + 1, z)]
                v5 = index_of[(x + 1, y + 1, z)]
                v6 = index_of[(x + 1, y + 1, z + 1)]
                v7 = index_of[(x, y + 1, z + 1)]

                if (x & 1) ^ (y & 1) ^ (z & 1):
                    add_tet(v0, v1, v4, v3)
                    add_tet(v2, v3, v6, v1)
                    add_tet(v5, v4, v1, v6)
                    add_tet(v7, v6, v3, v4)
                    add_tet(v4, v1, v6, v3)
                else:
                    add_tet(v1, v2, v5, v0)
                    add_tet(v3, v0, v7, v2)
                    add_tet(v4, v7, v0, v5)
                    add_tet(v6, v5, v2, v7)
                    add_tet(v5, v2, v7, v0)

    for i, j, k in faces.values():
        builder.add_triangle(i, j, k)


class Example:
    """Collapse a hollow soft arch so the legs self-collide with SolverFEMNewton."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        # ClassicFEM uses a fairly large implicit step; one Newton frame per display frame.
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # Hollow arch inside [-1, 1]^3. Members are thick enough to leave an
        # empty interior; the gap is still narrow enough that the legs meet
        # as the bridge sags under gravity.
        cell = 0.08
        dim_x, dim_y, dim_z = 10, 6, 8
        arm_cells, top_cells = 3, 3
        wall_cells = 1
        extent_x = dim_x * cell
        extent_y = dim_y * cell
        # Sit slightly above the ground plane near the bottom of the FEM domain.
        z0 = -0.65
        ground_height = z0 - 0.02
        _add_soft_hollow_arch(
            builder,
            pos=wp.vec3(-0.5 * extent_x, -0.5 * extent_y, z0),
            cell=cell,
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            arm_cells=arm_cells,
            top_cells=top_cells,
            wall_cells=wall_cells,
            density=1.0,
            k_mu=1.0e2,
            k_lambda=1.0e2,
            k_damp=0.0,
            particle_radius=args.collision_radius,
        )

        self.model = builder.finalize()

        # No Dirichlet clamp: free soft body with ground + self contacts.
        # SolverFEMNewton.gravity is a positive magnitude; acceleration is -g * up.
        self.solver = newton.solvers.SolverFEMNewton(
            model=self.model,
            resolution=args.resolution,
            up_axis=2,
            gravity=args.gravity,
            young_modulus=args.young_modulus,
            poisson_ratio=0.45,
            density=1.0,
            dt=self.sim_dt,
            n_newton=args.newton_iters,
            cg_iters=args.cg_iters,
            y_min=-2.0,
            y_max=2.0,
            quiet=False,
            ground=True,
            ground_height=ground_height,
            collision_stiffness=12,
            collision_radius=args.collision_radius,
            collision_detection_ratio=args.collision_detection_ratio,
            self_immunity_radius_ratio=10,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.graph = None

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(2.0, -2.0, 0.4), pitch=-20.0, yaw=135.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Arch should remain near the FEM domain while ground + self contacts act.
        p_lower = wp.vec3(-1.25, -1.25, -1.25)
        p_upper = wp.vec3(1.25, 1.25, 1.25)
        newton.examples.test_particle_state(
            self.state_0,
            "particles remain inside the FEM domain neighborhood",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--resolution", type=int, default=32, help="Background FEM grid resolution")
        parser.add_argument("--newton-iters", type=int, default=3, help="Newton iterations per frame")
        parser.add_argument("--cg-iters", type=int, default=150, help="Linear CG iterations per Newton step")
        parser.add_argument(
            "--young-modulus",
            type=float,
            default=50.0,
            help="Young modulus [Pa] (keep soft so the arch collapses)",
        )
        parser.add_argument(
            "--gravity",
            type=float,
            default=20.0,
            help="Gravity magnitude (positive); acceleration is -g along up_axis",
        )
        parser.add_argument(
            "--collision-stiffness",
            type=float,
            default=8.0,
            help="Contact energy stiffness scale",
        )
        parser.add_argument(
            "--collision-radius",
            type=float,
            default=0.02,
            help="Contact activation radius [m]",
        )
        parser.add_argument(
            "--collision-detection-ratio",
            type=float,
            default=2.0,
            help="Broad-phase radius multiplier for collision detection",
        )
        parser.add_argument(
            "--self-immunity-radius-ratio",
            type=float,
            default=6.0,
            help="Rest-space self-contact immunity radius ratio",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
