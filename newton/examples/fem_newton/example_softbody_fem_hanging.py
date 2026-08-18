# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody FEM Hanging
#
# Experimental classic FEM Newton soft body. A free soft cube sits inside the
# solver's [-1, 1]^3 background grid and rests on the ground plane.
# Particle-shape contacts come from CollisionPipeline; FEM self-collision
# stays internal. FEM ground is off so Newton particle-shape contacts against
# the plane provide the reaction.
#
# Command: uv run -m newton.examples softbody_fem_hanging
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    """Rest a soft cube on the ground with the experimental FEM Newton solver."""

    def __init__(self, viewer, args):
        self.viewer = viewer
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

        self.model = builder.finalize()

        # SolverFEMNewton.gravity is a positive magnitude; acceleration is -g * up.
        # Bounds sit outside [-1, 1]^3: nothing is Dirichlet-clamped.
        self.solver = newton.solvers.SolverFEMNewton(
            model=self.model,
            resolution=args.resolution,
            up_axis=2,
            gravity=args.gravity,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            density=1.0,
            dt=self.sim_dt,
            n_newton=args.newton_iters,
            cg_iters=args.cg_iters,
            y_min=-2.0,
            y_max=2.0,
            quiet=True,
            ground=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.graph = None

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            # Frame the small cube near the origin.
            self.viewer.set_camera(pos=wp.vec3(1.8, -1.8, 0.9), pitch=-20.0, yaw=135.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
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

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--resolution", type=int, default=64, help="Background FEM grid resolution")
        parser.add_argument("--newton-iters", type=int, default=2, help="Newton iterations per frame")
        parser.add_argument("--cg-iters", type=int, default=250, help="Linear CG iterations per Newton step")
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
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
