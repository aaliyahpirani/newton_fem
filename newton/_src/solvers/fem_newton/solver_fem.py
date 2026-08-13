# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp
import warp.fem as fem

from ...sim import Contacts, Control, Model, State
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase
from .deformable_model import ClassicFEM
from .quadrature import find_active_cells, grid_cell_vertex_indices
from .sdf_kernel import mesh_sdf_kernel
from .self_collision import MeshSelfCollisionHandler, CollisionPotential


@fem.integrand
def deformed_position(s: fem.Sample, domain: fem.Domain, displacement: fem.Field):
    """
    Compute the deformed position of a sample on the surface of a mesh. 
    """
    return domain(s) + displacement(s)


@fem.integrand
def fixed_points_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u_cur: fem.Field,
    u: fem.Field,
    v: fem.Field,
    up_axis: int,
    axis_min: float,
    axis_max: float,
):
    """Integrand that marks which FEM nodes are Dirichlet-fixed or the hanging-style clamp. The node is constrained if it is outside of axis_max or axis_min.
    It returns a 1.0 if the node is constrained, and a 0 otherwise. It assembles into a diagonal ish projector. 
    """
    coord = domain(s)[up_axis]
    clamped = wp.where(coord > axis_max or coord < axis_min, 1.0, 0.0)
    return wp.dot(u(s), v(s)) * clamped


class SolverFEMNewton(SolverBase, CouplingInterface):
    """Implicit Newton FEM solver for volumetric soft bodies.

    .. experimental::

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
            y_min: Lower clamp bound along ``up_axis`` for fixed-point BCs.
            y_max: Upper clamp bound along ``up_axis`` for fixed-point BCs.
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
            gravity: Gravity magnitude (positive). Acceleration is ``-gravity`` along
                ``up_axis``.
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
        # initialize base solver 
        super().__init__(model)

        # soft bodies are made of particles 
        if model.particle_count == 0:
            raise ValueError("SolverFEMNewton requires a model with particles.")
        # triangle connectivity 
        if model.tri_indices is None or model.tri_indices.size == 0:
            raise ValueError("SolverFEMNewton requires model.tri_indices (surface triangles).")

        self.resolution = resolution
        self.y_min = y_min
        self.y_max = y_max
        self.up_axis = up_axis

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
        # every soft body particle position
        self.points = model.particle_q
        # wp.Mesh expects a flat index buffer
        self.indices = model.tri_indices.flatten()
        # creates a mesh object from the poitns and indices 
        self.mesh = wp.Mesh(self.points, self.indices, support_winding_number=True)

        self.sim: ClassicFEM | None = None
        self.surface_vtx_quadrature = None
        self.rest_points = None
        self._sim_initialized = False

        # Grid for evaluating SDF / FEM
        self.geo = fem.Grid3D(
            res=wp.vec3i(wp.int32(resolution)),
            bounds_lo=wp.vec3(-1.0),
            bounds_hi=wp.vec3(1.0),
        )
        # 3d coordinates of the grid nodes
        self.grid_node_positions = fem.make_polynomial_space(self.geo).node_positions()
        # sdf of the grid nodes 
        self.grid_sdf = wp.empty(self.grid_node_positions.shape[0], dtype=float)
        wp.launch(
            mesh_sdf_kernel,
            dim=self.grid_node_positions.shape,
            inputs=[self.mesh.id, self.grid_node_positions, self.grid_sdf],
        )

        # lookup table for every hex cell on the background grid (ie which grid nodes are its corners)
        self.cell_vtx = grid_cell_vertex_indices(self.resolution)
        
        # mark active cells that intersect the interior of the SDF
        self.active_cells = wp.empty(self.cell_vtx.shape[0], dtype=int)
        find_active_cells(
            self.grid_sdf,
            self.active_cells,
            resolution=self.resolution,
            cell_vtx=self.cell_vtx,
        )

        # initialize the deformable simulation and displacement spaces 
        self.init_deformable_simulation(model, active_cells=self.active_cells)

    def init_deformable_simulation(self, model: Model, active_cells: wp.array | None = None):
        """
        Create the ClassicFEM simulation and initialize displacement spaces.
        
        Args:
            model: Newton model providing particle state layout.
            active_cells: wp.array of shape [resolution**3] marking active cells.
        
        Returns: 
            None. Initializes fields for deformable simulation. 
        """

        sim = ClassicFEM(self.geo, active_cells, **self.sim_kwargs)
        sim.init_displacement_space()
        sim.init_strain_spaces()

        # marks the nodes out side of the region as fixed points 
        sim.set_fixed_points_condition(
            fixed_points_projector_form,
            {
                "up_axis": self.up_axis,
                "axis_min": self.y_min,
                "axis_max": self.y_max,
            },
        )

        self.sim = sim
        self.rest_points = wp.clone(model.particle_q)
        # interpolates local coordinates of the surface vertices to the grid nodes 
        self.surface_vtx_quadrature = fem.PicQuadrature(
            self.sim.u_test.domain,
            self.rest_points,
            max_dist=4.0 / float(self.resolution),
        )

        self.collision_handler = MeshSelfCollisionHandler(self.surface_vtx_quadrature, self.mesh)
        collision_potential = CollisionPotential(self.sim, self.collision_handler)
        self.sim.add_energy_potential(collision_potential)
        # forms that do not change within a frame
        sim.init_constant_forms()
        # not necessary for ClassicFEM
        sim.project_constant_forms()
        
        self._sim_initialized = True

    def is_initialized(self) -> bool:
        return self._sim_initialized

    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> None:
        del control, contacts
        # define the simulation object 
        sim = self.sim
        if sim is None:
            raise RuntimeError("Simulation has not been created; call init_deformable_simulation() first.")

        # set the timestep
        sim.dt = dt
        # run the frame
        sim.run_frame()

        # store the previous particle velocities in state_in
        if state_out is not state_in:
            if state_out.particle_qd is not None and state_in.particle_qd is not None:
                state_out.particle_qd.assign(state_in.particle_qd)

        # interpolate the deformed positions to the surface vertices 
        fem.interpolate(
            deformed_position,
            at=self.surface_vtx_quadrature,
            dest=state_out.particle_q,
            fields={"displacement": sim.u_field},
        )
