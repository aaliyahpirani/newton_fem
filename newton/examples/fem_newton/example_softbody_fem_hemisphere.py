# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody FEM Hemisphere
#
# Experimental classic FEM Newton soft body. A solid hemisphere sits on the
# ground inside the solver's [-1, 1]^3 background grid; the base slab is
# Dirichlet-clamped so the dome is tethered. A MuJoCo rigid cube drops onto
# the dome. SolverMuJoCo integrates the cube, SolverFEMNewton deforms the
# hemisphere, and SolverCoupledProxy exchanges contact wrenches.
#
# Command: uv run -m newton.examples softbody_fem_hemisphere
#
###########################################################################

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverFEMNewton, SolverMuJoCo

HEMI_RADIUS = 0.55
HEMI_LAT = 16
HEMI_LON = 32
CUBE_HALF = 0.12
CUBE_DROP_Z = HEMI_RADIUS + CUBE_HALF + 0.35
CUBE_DENSITY = 80.0


def _make_hemisphere_mesh(
    radius: float,
    n_lat: int,
    n_lon: int,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed hemisphere (dome + equatorial disk) with outward winding.

    Sphere center is ``origin``; the base lies on z = origin[2] and the pole
    is at origin + (0, 0, radius).
    """
    verts: list[list[float]] = []
    # Pole.
    verts.append([0.0, 0.0, radius])
    # Latitude rings from just below the pole down to the equator.
    for i in range(1, n_lat + 1):
        phi = 0.5 * np.pi * i / n_lat
        sp, cp = np.sin(phi), np.cos(phi)
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            verts.append([radius * sp * np.cos(theta), radius * sp * np.sin(theta), radius * cp])
    base_center = len(verts)
    verts.append([0.0, 0.0, 0.0])

    faces: list[list[int]] = []

    def ring_index(ring: int, j: int) -> int:
        # ring 1 is the first ring below the pole; ring n_lat is the equator.
        return 1 + (ring - 1) * n_lon + (j % n_lon)

    # Pole fan, CCW when viewed from outside (+Z).
    for j in range(n_lon):
        faces.append([0, ring_index(1, j), ring_index(1, j + 1)])

    # Dome quads as two triangles.
    for ring in range(1, n_lat):
        for j in range(n_lon):
            a = ring_index(ring, j)
            b = ring_index(ring, j + 1)
            c = ring_index(ring + 1, j + 1)
            d = ring_index(ring + 1, j)
            faces.append([a, d, b])
            faces.append([b, d, c])

    # Base disk, CCW when viewed from outside (-Z).
    for j in range(n_lon):
        faces.append([base_center, ring_index(n_lat, j + 1), ring_index(n_lat, j)])

    vertices = np.asarray(verts, dtype=np.float32) + origin.astype(np.float32)
    return vertices, np.asarray(faces, dtype=np.int32)


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
    """Drop a MuJoCo cube onto a tethered FEM Newton hemisphere."""

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
        # Sit the base on z = 0 so the Dirichlet clamp tethers the dome to the ground.
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        vertices, faces = _make_hemisphere_mesh(HEMI_RADIUS, HEMI_LAT, HEMI_LON, origin)
        _add_surface_mesh(builder, vertices, faces, particle_radius)

        builder.add_ground_plane(height=0.0)

        cube_cfg = newton.ModelBuilder.ShapeConfig(density=CUBE_DENSITY, mu=0.4)
        cube_xform = wp.transform(wp.vec3(0.04, 0.0, CUBE_DROP_Z), wp.quat_identity())
        cube_body = builder.add_link(xform=cube_xform, label="cube")
        cube_joint = builder.add_joint_free(child=cube_body, label="cube_free")
        builder.add_articulation([cube_joint], label="cube")
        builder.add_shape_box(
            cube_body,
            hx=CUBE_HALF,
            hy=CUBE_HALF,
            hz=CUBE_HALF,
            cfg=cube_cfg,
            color=wp.vec3(0.25, 0.55, 0.95),
            label="drop_cube",
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
                    bodies=[cube_body],
                    joints=[cube_joint],
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
                        bodies=[cube_body],
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
            self.viewer.set_camera(pos=wp.vec3(2.2, -2.2, 1.0), pitch=-22.0, yaw=135.0)

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
        p_lower = wp.vec3(-1.25, -1.25, -0.25)
        p_upper = wp.vec3(1.25, 1.25, 1.5)
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
        parser.add_argument("--resolution", type=int, default=64, help="Background FEM grid resolution")
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
            default=0.08,
            help="Lower Dirichlet clamp along Z; nodes below this are tethered to the ground",
        )
        parser.add_argument(
            "--axis-max",
            type=float,
            default=2.0,
            help="Upper Dirichlet clamp along Z; 2.0 leaves the dome free",
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
