# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody FEM Hanging
#
# Experimental classic FEM Newton soft body. A soft cube sits inside the
# solver's [-1, 1]^3 background grid; nodes above a Z clamp are fixed so the
# cube hangs and sags under gravity onto a static box.
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


class Example:
    """Rest a soft cube on the ground with the experimental FEM Newton solver."""

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

        # Soft cube inside [-1, 1]^3. No Dirichlet clamp; it sits on the ground.
        cell = 0.08  # edge length of one voxel
        dim = 8  # number of elements along each axis so it is 8x8x8 cells
        extent = dim * cell  # full side length of the cube
        # Match SolverFEMNewton's default collision_radius = 0.5 / resolution so
        # pipeline detection and the FEM penalty see the same particle size.
        particle_radius = 0.5 / float(args.resolution)

        # Infinite ground under the cube so it does not fall through the domain.
        builder.add_ground_plane()

        # Center in XY, a short drop above the ground plane.
        z0 = 0.04
        builder.add_soft_grid(
            pos=wp.vec3(-0.5 * extent, -0.5 * extent, z0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            dim_z=dim,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
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

        sphere_radius = 0.12
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
            "young_modulus": args.young_modulus,
            "poisson_ratio": 0.1,
            "density": 1.0,
            "dt": self.sim_dt,
            "n_newton": args.newton_iters,
            "cg_iters": args.cg_iters,
            "y_min": -2.0,
            "y_max": axis_max,
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
