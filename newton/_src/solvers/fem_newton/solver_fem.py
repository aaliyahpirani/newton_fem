# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp
import warp.fem as fem

from ...sim import Contacts, Control, Model, State
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase
from .deformable_model import ClassicFEM


@fem.integrand
def deformed_position(s: fem.Sample, domain: fem.Domain, displacement: fem.Field):
    return domain(s) + displacement(s)


class SolverFEMNewton(SolverBase, CouplingInterface):
    """Implicit Newton FEM solver for volumetric soft bodies.

    .. experimental::
        Public API and behavior may change without prior notice.
    """

    def __init__(
        self,
        model: Model,
        *,
        resolution: int = 64,
        y_min: float = -0.9,
        y_max: float = 0.99,
        degree: int = 1,
        serendipity: bool = False,
        discontinuous: bool = False,
        n_newton: int = 2,
        newton_tol: float = 1.0e-4,
        cg_tol: float = 1.0e-6,
        cg_iters: int = 250,
        n_backtrack: int = 4,
        young_modulus: float = 500.0,
        poisson_ratio: float = 0.45,
        gravity: float = 1.0,
        up_axis: int = 1,
        density: float = 1.0,
        dt: float = 0.05,
        quasi_quasistatic: bool = False,
        neo_hookean: bool = False,
        dg_jump_pen: float = 1.0,
        quiet: bool = False,
        lumped_mass: bool = False,
        fp64: bool = False,
        matrix_free: bool = False,
        collision_stiffness: float = 1.0,
        collision_radius: float | None = None,
        collision_detection_ratio: float = 2.0,
        friction: float = 0.2,
        friction_reg: float = 0.1,
        friction_fluid: float = 0.01,
        ground: bool = True,
        ground_height: float = -1.0,
        self_immunity_radius_ratio: float = 4.0,
    ):
        """Initialize the background grid and store solver/model parameters.

        Args:
            model: Newton model providing particle state layout.
            resolution: Background grid resolution used for SDF / FEM.
            y_min: Lower bound used by fixed-point boundary conditions.
            y_max: Upper bound used by fixed-point boundary conditions.
            degree: Polynomial degree of the displacement basis.
            serendipity: If True, use serendipity elements.
            discontinuous: If True, use a discontinuous displacement basis.
            n_newton: Maximum Newton iterations per frame.
            newton_tol: Newton convergence tolerance.
            cg_tol: Linear CG relative tolerance.
            cg_iters: Maximum linear CG iterations.
            n_backtrack: Maximum line-search backtracking steps.
            young_modulus: Young modulus [Pa].
            poisson_ratio: Poisson ratio.
            gravity: Gravity magnitude.
            up_axis: Gravity / ground axis index (0=x, 1=y, 2=z).
            density: Mass density [kg/m^3].
            dt: Default simulation timestep [s].
            quasi_quasistatic: If True, suppress inertial history terms.
            neo_hookean: If True, use neo-Hookean elasticity; else corotational.
            dg_jump_pen: Discontinuous-Galerkin jump penalty scale.
            quiet: If True, suppress per-iteration logging.
            lumped_mass: If True, assemble a nodal (lumped) mass matrix.
            fp64: If True, solve the Newton system in float64.
            matrix_free: If True, use a matrix-free Newton operator when supported.
            collision_stiffness: Contact energy stiffness scale.
            collision_radius: Contact activation radius; defaults to ``0.5 / resolution``.
            collision_detection_ratio: Broad-phase radius multiplier.
            friction: Coulomb friction coefficient.
            friction_reg: Friction regularization scale.
            friction_fluid: Fluid-like friction regularization scale.
            ground: If True, enable ground collisions.
            ground_height: Ground plane height along ``up_axis``.
            self_immunity_radius_ratio: Rest-space self-contact immunity radius ratio.
        """
        super().__init__(model)

        self.resolution = resolution
        self.y_min = y_min
        self.y_max = y_max

        if collision_radius is None:
            collision_radius = 0.5 / float(resolution)

        self.sim_kwargs = {
            "degree": degree,
            "serendipity": serendipity,
            "discontinuous": discontinuous,
            "n_newton": n_newton,
            "newton_tol": newton_tol,
            "cg_tol": cg_tol,
            "cg_iters": cg_iters,
            "n_backtrack": n_backtrack,
            "young_modulus": young_modulus,
            "poisson_ratio": poisson_ratio,
            "gravity": gravity,
            "up_axis": up_axis,
            "density": density,
            "dt": dt,
            "quasi_quasistatic": quasi_quasistatic,
            "neo_hookean": neo_hookean,
            "dg_jump_pen": dg_jump_pen,
            "quiet": quiet,
            "lumped_mass": lumped_mass,
            "fp64": fp64,
            "matrix_free": matrix_free,
            "collision_stiffness": collision_stiffness,
            "collision_radius": collision_radius,
            "collision_detection_ratio": collision_detection_ratio,
            "friction": friction,
            "friction_reg": friction_reg,
            "friction_fluid": friction_fluid,
            "ground": ground,
            "ground_height": ground_height,
            "self_immunity_radius_ratio": self_immunity_radius_ratio,
        }

        # Grid for evaluating SDF / FEM
        self.geo = fem.Grid3D(
            res=wp.vec3i(wp.int32(resolution)),
            bounds_lo=wp.vec3(-1.0),
            bounds_hi=wp.vec3(1.0),
        )
        self.grid_node_positions = fem.make_polynomial_space(self.geo).node_positions()
        self.grid_sdf = wp.empty(self.grid_node_positions.shape[0], dtype=float)

        # Clay simulation state (topology rebuilt later via init_mesh_simulation)
        self.sim: ClassicFEM | None = None
        self.tri_mesh = None
        self.tri_vtx_quadrature = None
        self.rest_points = None
        self._sim_initialized = False

        

    def init_mesh_simulation(self, active_cells: wp.array | None = None):
        """Create the ClassicFEM simulation and initialize displacement spaces."""
        sim = ClassicFEM(self.geo, active_cells, **self.sim_kwargs)
        sim.init_displacement_space()
        self.sim = sim
        self._sim_initialized = False

    def is_initialized(self) -> bool:
        return self._sim_initialized

    def ensure_sim_is_initialized(self):
        if self.sim is None:
            raise RuntimeError("Simulation has not been created; call init_mesh_simulation() first.")

        if not self._sim_initialized:
            self.sim.init_constant_forms()
            self.sim.project_constant_forms()
            self._sim_initialized = True

    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> None:
        del state_in, control, contacts, dt
        sim = self.sim
        if sim is None:
            raise RuntimeError("Simulation has not been created; call init_mesh_simulation() first.")

        self.ensure_sim_is_initialized()
        sim.run_frame()

        if self.tri_vtx_quadrature is not None:
            fem.interpolate(
                deformed_position,
                quadrature=self.tri_vtx_quadrature,
                dest=state_out.particle_q,
                fields={"displacement": sim.u_field},
            )
