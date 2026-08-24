# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody FEM Hanging
#
# Experimental classic FEM Newton soft body. A hollow soft cube (outer walls
# only) sits inside the solver's [-1, 1]^3 background grid and sags under
# gravity onto a static box.
#
# By default a dynamic rigid sphere is coupled through SolverCoupledProxy:
# CollisionPipeline detects particle-shape contacts, FEM applies the soft
# penalty, harvest writes the equal-and-opposite wrench into body_f, and
# SolverMuJoCo integrates that force so the sphere reacts.
#
# Pass --fem-only to drop the sphere and run FEM against the static floor.
#
# Command: uv run -m newton.examples softbody_fem_hanging
#          uv run -m newton.examples softbody_fem_hanging --fem-only
#
###########################################################################

import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverMuJoCo


def _add_soft_hollow_cube(
    builder: newton.ModelBuilder,
    *,
    pos: wp.vec3,
    cell: float,
    dim: int,
    wall_cells: int,
    density: float,
    k_mu: float,
    k_lambda: float,
    k_damp: float,
    particle_radius: float | None = None,
) -> None:
    """Add a voxelized hollow cube (outer walls only) as tetrahedra.

    Interior cells are skipped so the cube is a shell, not a solid fill.
    Each included hex is split into 5 tets, matching
    :meth:`ModelBuilder.add_soft_grid`.
    """
    mass = cell * cell * cell * density

    def cell_in_cube(x: int, y: int, z: int) -> bool:
        return 0 <= x < dim and 0 <= y < dim and 0 <= z < dim

    def cell_on_shell(x: int, y: int, z: int) -> bool:
        if not cell_in_cube(x, y, z):
            return False
        return (
            x < wall_cells
            or x >= dim - wall_cells
            or y < wall_cells
            or y >= dim - wall_cells
            or z < wall_cells
            or z >= dim - wall_cells
        )

    used: set[tuple[int, int, int]] = set()
    for z in range(dim):
        for y in range(dim):
            for x in range(dim):
                if not cell_on_shell(x, y, z):
                    continue
                for dz in (0, 1):
                    for dy in (0, 1):
                        for dx in (0, 1):
                            used.add((x + dx, y + dy, z + dz))

    index_of: dict[tuple[int, int, int], int] = {}
    for iz in range(dim + 1):
        for iy in range(dim + 1):
            for ix in range(dim + 1):
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

    for z in range(dim):
        for y in range(dim):
            for x in range(dim):
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
    """Rest a hollow soft cube on the ground with the experimental FEM Newton solver."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fem_only = bool(args.fem_only)
        self.sim_time = 0.0
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        # ClassicFEM uses a fairly large implicit step; one Newton frame per display frame.
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Match Newton viewer convention (Z up) with the FEM grid domain.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # Hollow cube inside [-1, 1]^3. No Dirichlet clamp; it sits on the ground.
        cell = 0.08  # edge length of one voxel
        dim = 8  # number of elements along each axis so it is 8x8x8 cells
        wall_cells = 1  # outer-face thickness in voxels; interior is empty
        extent = dim * cell  # full side length of the cube
        # Match SolverFEMNewton's default collision_radius = 0.5 / resolution so
        # pipeline detection and the FEM penalty see the same particle size.
        particle_radius = 0.5 / float(args.resolution)

        # Infinite ground under the cube so it does not fall through the domain.
        builder.add_ground_plane()

        # Center in XY, a short drop above the ground plane.
        z0 = 0.04
        _add_soft_hollow_cube(
            builder,
            pos=wp.vec3(-0.5 * extent, -0.5 * extent, z0),
            cell=cell,
            dim=dim,
            wall_cells=wall_cells,
            density=1.0,
            k_mu=1.0e2,
            k_lambda=1.0e2,
            k_damp=0.0,
            particle_radius=particle_radius,
        )

        # Static floor under the cube. FEM ground=False so Newton particle-shape
        # contacts are what generate the reaction, not the built-in plane.
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
            hx=1.0,
            hy=1.0,
            hz=0.05,
        )

        sphere_radius = 0.06
        sphere_body = None
        sphere_joint = None
        if not self.fem_only:
            newton.use_coord_layout_targets = True
            # Drop onto the hanging cube from a small gap above the top face.
            cube_top = z0 + extent
            sphere_xform = wp.transform(wp.vec3(0.0, 0.0, cube_top + sphere_radius + 0.04), wp.quat_identity())
            sphere_cfg = newton.ModelBuilder.ShapeConfig(density=50.0, ke=1.0e5, kd=1.0e-4, kf=1.0e3, mu=0.3)
            sphere_body = builder.add_link(xform=sphere_xform, label="sphere")
            sphere_joint = builder.add_joint_free(child=sphere_body, label="sphere_free")
            builder.add_articulation([sphere_joint], label="sphere")
            builder.add_shape_sphere(
                sphere_body,
                radius=sphere_radius,
                cfg=sphere_cfg,
                color=wp.vec3(0.95, 0.43, 0.18),
                label="rigid_sphere",
            )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -float(args.gravity)))
        self.sphere_body = sphere_body
        self.particle_radius = particle_radius

        # SolverFEMNewton.gravity is a positive magnitude; acceleration is -g * up.
        axis_max = z0 + extent - 2.0 * cell
        fem_kwargs = {
            "resolution": args.resolution,
            "up_axis": 2,
            "gravity": args.gravity,
            "young_modulus": 200.0,
            "poisson_ratio": 0.1,
            "density": 1.0,
            "dt": self.sim_dt,
            "n_newton": args.newton_iters,
            "cg_iters": args.cg_iters,
            "y_min": -2.0,
            "y_max": 2.0,
            "quiet": True,
            "ground": False,
            "collision_stiffness": args.collision_stiffness,
        }

        if self.fem_only:
            self.solver = newton.solvers.SolverFEMNewton(model=self.model, **fem_kwargs)
        else:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
            self.solver = SolverCoupledProxy(
                model=self.model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name="mjc",
                        solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=64),
                        bodies=[sphere_body],
                        joints=[sphere_joint],
                    ),
                    SolverCoupledProxy.Entry(
                        name="fem",
                        solver=lambda v, kwargs=fem_kwargs: newton.solvers.SolverFEMNewton(model=v, **kwargs),
                        particles=list(range(self.model.particle_count)),
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="mjc",
                            destination="fem",
                            bodies=[sphere_body],
                            mass_scale=args.mass_scale,
                            mode=args.coupling_mode,
                            collision_pipeline=lambda model: newton.CollisionPipeline(
                                model, soft_contact_margin=self.particle_radius
                            ),
                            collide_interval=1,
                        )
                    ],
                    iterations=args.proxy_iterations,
                ),
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=particle_radius)
        self.contacts = self.collision_pipeline.contacts()
        self.graph = None

        if self.fem_only:
            self.viewer.set_model(self.model)
        else:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
            newton.examples.configure_coupled_view(self, args)

        if hasattr(self.viewer, "set_camera"):
            # Frame the small cube near the origin.
            self.viewer.set_camera(pos=wp.vec3(1.8, -1.8, 1.0), pitch=-20.0, yaw=135.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.fem_only:
                self.viewer.apply_forces(self.state_0)
            else:
                newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Cube should remain near the FEM domain while sitting on the ground.
        p_lower = wp.vec3(-1.0, -1.0, -1.0)
        p_upper = wp.vec3(1.0, 1.0, 1.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles remain inside the FEM domain neighborhood",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        if self.sphere_body is None:
            return
        sphere_z = float(self.state_0.body_q.numpy()[self.sphere_body, 2])
        assert sphere_z == sphere_z, "sphere pose is not finite"
        # Floor top is z=0.10; a sphere that falls through the cube rests near 0.22.
        # Two-way coupling should keep it on the hanging cube (top ~0.79).
        assert sphere_z > 0.15, f"sphere fell through the FEM cube: z={sphere_z:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self.fem_only:
            self.viewer.log_state(self.state_0)
        else:
            newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument("--resolution", type=int, default=24, help="Background FEM grid resolution")
        parser.add_argument("--newton-iters", type=int, default=3, help="Newton iterations per frame")
        parser.add_argument("--cg-iters", type=int, default=150, help="Linear CG iterations per Newton step")
        parser.add_argument(
            "--young-modulus",
            type=float,
            default=10.0,
            help="Young modulus [Pa]",
        )
        parser.add_argument(
            "--poisson-ratio",
            type=float,
            default=0.45,
            help="Poisson ratio",
        )
        parser.add_argument(
            "--gravity",
            type=float,
            default=1.0,
            help="Gravity magnitude (positive); acceleration is -g along up_axis",
        )
        parser.add_argument(
            "--fem-only",
            action="store_true",
            help="Skip the dynamic sphere and MuJoCo coupling (static floor only)",
        )
        parser.add_argument(
            "--collision-stiffness",
            type=float,
            default=10.0,
            help="FEM particle-shape penalty stiffness scale",
        )
        parser.add_argument(
            "--coupling-mode",
            type=str,
            choices=["lagged", "staggered"],
            default="lagged",
            help="Proxy state transfer mode",
        )
        parser.add_argument(
            "--mass-scale",
            type=float,
            default=1.0,
            help="Scale factor for rigid effective mass used by the FEM proxy",
        )
        parser.add_argument(
            "--proxy-iterations",
            type=int,
            default=1,
            help="Number of proxy relaxation passes per substep",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
