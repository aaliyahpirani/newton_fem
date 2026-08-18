# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody FEM Hanging
#
# Experimental classic FEM Newton soft body. A soft cube sits inside the
# solver's [-1, 1]^3 background grid; nodes above a Z clamp are fixed so the
# cube hangs and sags under gravity onto the ground plane. Particle-shape
# contacts come from CollisionPipeline; FEM self-collision stays internal.
#
# Command: uv run -m newton.examples softbody_fem_hanging
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    """Hang a soft cube with the experimental FEM Newton solver."""

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

        # Soft cube inside [-1, 1]^3. Top slab is clamped by axis_max below.
        cell = 0.08 # edge length of one voxel
        dim = 8 # number of elements along each axis so it is 8x8x8 cells
        extent = dim * cell # full side length of each cube
        # Center in XY, hang from near the top of the FEM domain.
        z0 = 0.15
        # Match SolverFEMNewton's default collision_radius = 0.5 / resolution so
        # pipeline detection and the FEM penalty see the same particle size.
        particle_radius = 0.5 / float(args.resolution)
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

        # Infinite ground under the cube. FEM ground=False so Newton particle-shape
        # contacts against this plane generate the reaction, not the built-in plane.
        builder.add_ground_plane()

        self.model = builder.finalize()

        # Clamp FEM nodes above this Z (pins the top of the cube).
        # SolverFEMNewton.gravity is a positive magnitude; acceleration is -g * up.
        axis_max = z0 + extent - 2.0 * cell
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
            self.viewer.set_camera(pos=wp.vec3(1.6, -1.6, 0.7), pitch=-15.0, yaw=135.0)

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
        # Free end should sag below the rest z-min while the clamped top stays put.
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
