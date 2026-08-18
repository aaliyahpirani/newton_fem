# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody FEM Muscle
#
# Experimental classic FEM Newton soft body. Loads simple_muscle.obj,
# normalizes it into the solver's [-1, 1]^3 background grid (Y-up source
# rotated to Newton Z-up), and sags under gravity onto the ground plane.
# A MuJoCo rigid sphere is dropped onto the mesh: SolverMuJoCo integrates
# the sphere, SolverFEMNewton deforms the muscle, and SolverCoupledProxy
# exchanges contact wrenches.
#
# Command: uv run -m newton.examples softbody_fem_muscle
#
###########################################################################

import os

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverFEMNewton, SolverMuJoCo

BALL_RADIUS = 0.16
BALL_DROP_Z = 1.2
BALL_DENSITY = 80.0


def _resolve_mesh_path(mesh: str) -> str:
    """Resolve a mesh path, falling back to the example asset directory."""
    if os.path.isfile(mesh):
        return mesh
    asset = newton.examples.get_asset(os.path.basename(mesh))
    if os.path.isfile(asset):
        return asset
    raise FileNotFoundError(
        f"Mesh not found: {mesh!r}. Pass --mesh PATH or place simple_muscle.obj in {newton.examples.get_asset_directory()}."
    )


def _load_normalized_surface(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh and fit it into [-1, 1]^3, matching example_cutting.

    Cutting meshes are Y-up. Rotate (x, y, z) -> (x, -z, y) so gravity and the
    Newton viewer share Z-up.
    """
    surface = newton.Mesh.create_from_file(path, compute_inertia=False)
    points = np.asarray(surface.vertices, dtype=np.float32)
    faces = np.asarray(surface.indices, dtype=np.int32).reshape(-1, 3)

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = np.max(bbox_max - bbox_min + 0.001)
    normalized = (2.0 * points - bbox_min - bbox_max) / extent

    x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2]
    vertices = np.stack((x, -z, y), axis=1).astype(np.float32)
    return vertices, faces


def _add_surface_mesh(
    builder: newton.ModelBuilder,
    vertices: np.ndarray,
    faces: np.ndarray,
    particle_radius: float,
) -> None:
    """Add surface vertices as particles and faces as collision triangles."""
    n = int(vertices.shape[0])
    builder.add_particles(
        pos=[wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in vertices],
        vel=[wp.vec3(0.0, 0.0, 0.0)] * n,
        mass=[1.0] * n,
        radius=[particle_radius] * n,
    )
    builder.add_triangles(
        faces[:, 0].astype(np.int32).tolist(),
        faces[:, 1].astype(np.int32).tolist(),
        faces[:, 2].astype(np.int32).tolist(),
    )


class Example:
    """Drop a MuJoCo sphere onto a FEM Newton muscle mesh."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        # ClassicFEM uses a fairly large implicit step; one Newton frame per display frame.
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverMuJoCo.register_custom_attributes(builder)

        particle_radius = 0.5 / float(args.resolution)
        mesh_path = _resolve_mesh_path(args.mesh)
        vertices, faces = _load_normalized_surface(mesh_path)
        _add_surface_mesh(builder, vertices, faces, particle_radius)

        # Ground at the bottom of the FEM domain. FEM ground=False so Newton
        # particle-shape contacts generate the reaction.
        builder.add_ground_plane(height=-1.0)

        ball_cfg = newton.ModelBuilder.ShapeConfig(density=BALL_DENSITY, mu=0.4)
        ball_xform = wp.transform(wp.vec3(0.05, 0.0, BALL_DROP_Z), wp.quat_identity())
        ball_body = builder.add_link(xform=ball_xform, label="ball")
        ball_joint = builder.add_joint_free(child=ball_body, label="ball_free")
        builder.add_articulation([ball_joint], label="ball")
        builder.add_shape_sphere(
            ball_body,
            radius=BALL_RADIUS,
            cfg=ball_cfg,
            color=wp.vec3(0.95, 0.35, 0.18),
            label="drop_sphere",
        )

        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        fem_kwargs = {
            "resolution": args.resolution,
            "up_axis": 2,
            "gravity": args.gravity,
            "young_modulus": args.young_modulus,
            "poisson_ratio": args.poisson_ratio,
            "density": 1.0,
            "dt": self.sim_dt,
            "n_newton": args.newton_iters,
            "cg_iters": args.cg_iters,
            "y_min": args.axis_min,
            "y_max": args.axis_max,
            "neo_hookean": args.neo_hookean,
            "quiet": True,
            "ground": False,
        }
        mujoco_kwargs = {"use_mujoco_contacts": False, "njmax": 64}
        particle_ids = list(range(self.model.particle_count))
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(model=v, **mujoco_kwargs),
                    bodies=[ball_body],
                    joints=[ball_joint],
                ),
                SolverCoupledProxy.Entry(
                    name="fem",
                    solver=lambda v, kw=fem_kwargs: SolverFEMNewton(model=v, **kw),
                    particles=particle_ids,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="fem",
                        bodies=[ball_body],
                        collision_pipeline=lambda model: newton.CollisionPipeline(model),
                        collide_interval=1,
                    )
                ],
                iterations=1,
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.graph = None

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.examples.configure_coupled_view(self, args)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(2.4, -2.4, 0.9), pitch=-20.0, yaw=135.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        p_lower = wp.vec3(-1.25, -1.25, -1.25)
        p_upper = wp.vec3(1.25, 1.25, 1.25)
        newton.examples.test_particle_state(
            self.state_0,
            "particles remain inside the FEM domain neighborhood",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--mesh",
            type=str,
            default="simple_muscle.obj",
            help="Triangle mesh to load (default: examples/assets/simple_muscle.obj)",
        )
        parser.add_argument("--resolution", type=int, default=32, help="Background FEM grid resolution")
        parser.add_argument("--newton-iters", type=int, default=2, help="Newton iterations per frame")
        parser.add_argument("--cg-iters", type=int, default=250, help="Linear CG iterations per Newton step")
        parser.add_argument(
            "--young-modulus",
            type=float,
            default=400.0,
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
            "--axis-min",
            type=float,
            default=-0.9,
            help="Lower Dirichlet clamp along up axis (Z after Y-up rotation)",
        )
        parser.add_argument(
            "--axis-max",
            type=float,
            default=2.0,
            help="Upper Dirichlet clamp along up axis; 2.0 leaves the top free so the ball can squash it",
        )
        parser.add_argument(
            "--neo-hookean",
            action="store_true",
            default=False,
            help="Use neo-Hookean elasticity instead of corotational",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
