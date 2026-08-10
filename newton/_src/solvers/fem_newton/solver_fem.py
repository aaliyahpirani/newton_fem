# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp
import warp.fem as fem

from ...sim import Contacts, Control, Model, State
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase

# example_cutting.py / SoftbodySim / collision defaults (tunable constants for now)
QUADRATURE_MODEL: list[str] | str | None = None
RESOLUTION = 64
FORCE_SCALE = 1.0
TEAR_FORCE_THRESHOLD = 1.0
Y_MIN = -0.9
Y_MAX = 0.99
CUT_PICK_RADIUS = -0.99
DEBUG = False
DEGREE = 1
SERENDIPITY = False
N_FRAMES = -1
N_NEWTON = 2
NEWTON_TOL = 1.0e-4
CG_TOL = 1.0e-6
CG_ITERS = 250
N_BACKTRACK = 4
YOUNG_MODULUS = 500.0
POISSON_RATIO = 0.45
GRAVITY = 1.0
UP_AXIS = 1
DENSITY = 1.0
DT = 0.05
QUASI_QUASISTATIC = False
NEO_HOOKEAN = False
DISCONTINUOUS = False
DG_JUMP_PEN = 1.0
STEP_SIZE = 0.001
QUIET = False
LUMPED_MASS = False
FP64 = False
MATRIX_FREE = False
COLLISION_STIFFNESS = 1.0
COLLISION_RADIUS = 0.5 / RESOLUTION
COLLISION_DETECTION_RATIO = 2.0
FRICTION = 0.2
FRICTION_REG = 0.1
FRICTION_FLUID = 0.01
GROUND = True
GROUND_HEIGHT = -1.0
SELF_IMMUNITY_RADIUS_RATIO = 4.0
CLIP = False


class SolverFEMNewton(SolverBase, CouplingInterface):
    """
    TODO: Add docstring
    """

    def __init__(self, model: Model):
        """
        Initialize the background grid for evaluating the SDF and FEM, 
        as well as space for storage for the grid_sdf and grid_node_positions.
        """
        super().__init__(model)

        # initialize the mesh self collision handler

        # Grid for evaluating SDF / FEM
        self.geo = fem.Grid3D(
            res=wp.vec3i(wp.int32(RESOLUTION)),
            bounds_lo=wp.vec3(-1.0),
            bounds_hi=wp.vec3(1.0),
        )
        self.grid_node_positions = fem.make_polynomial_space(self.geo).node_positions()
        self.grid_sdf = wp.empty(self.grid_node_positions.shape[0], dtype=float)

        # Clay simulation state (topology rebuilt later via create_sim)
        self.sim = None
        self.tri_mesh = None
        self.tri_vtx_quadrature = None
        self.rest_points = None
        self._sim_initialized = False

    def create_and_interpolate_mesh_data(self):
        """
        Interpolate mesh data from the previous simulation frame onto the current one. If no simulation has been 
        created yet, create a new simulation. 
        """
        # TODO: implement the sim object (softbody sim/classic fem)
        # this is what stores all the data for the simulation


    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> None:
        pass
