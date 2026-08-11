"""
A newton-based FEM simulator.
# TODO: Update explanation of the model
"""

from __future__ import annotations

import weakref
from typing import TextIO

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.fem import Domain, Field, Sample
from warp.optim.linear import LinearOperator

from .elasticity import (
    hooke_energy,
    hooke_hessian,
    hooke_stress,
    snh_energy,
    snh_hessian_proj_analytic,
    snh_stress,
    symmetric_strain,
    symmetric_strain_delta,
)
from .linalg import array_axpy, bsr_cg, diff_bsr_mv
from .linesearch_criterion import (
    LineSearchNaiveCriterion,
    LineSearchUnconstrainedArmijoCriterion,
)

@fem.integrand
def defgrad(u: fem.Field, s: fem.Sample):
    """
    Computes the deformation gradient F = grad(u) + I, if no displacement then material is undeformed. 
    """
    return fem.grad(u, s) + wp.identity(n=3, dtype=float)


@fem.integrand
def defgrad_avg(u: fem.Field, s: fem.Sample):
    """
    Computes the average deformation gradient. This applies for discontinuous fields, where the 
    field needs to be averaged across elements.
    """
    return fem.grad_average(u, s) + wp.identity(n=3, dtype=float)


@fem.integrand
def inertia_form(s: Sample, domain: Domain, u: Field, v: Field, rho: float, dt: float):
    """
    Defines the intertia form for the displacement field. 
    <rho/dt^2 u, v>
    
    Args:
        s: sample 
        domain: domain to integrate over
        u: displacement at sample s
        v: test function at sample s
        rho: density
        dt: timestep

    Returns:
        The inertia form for the displacement field.
    """

    u_rhs = rho * u(s) / (dt * dt)
    return wp.dot(u_rhs, v(s))


@fem.integrand
def dg_penalty_form(s: Sample, domain: Domain, u: Field, v: Field, k: float):
    """
    Defines the penalty for neighbouring elements. k is the penalty stiffness.
    """
    # compute the difference in u between the two sides of an element boundary
    ju = fem.jump(u, s)
    jv = fem.jump(v, s)

    # scale the penalty
    return wp.dot(ju, jv) * k * fem.measure_ratio(domain, s)


@fem.integrand
def displacement_rhs_form(
    s: Sample,
    domain: Domain,
    u: Field,
    u_prev: Field,
    v: Field,
    rho: float,
    gravity: wp.vec3,
    dt: float,
):
    """
    Displacement right hand side form, it includes the intertia form and the gravity term. 
    <rho/dt^2 u, v> + <rho g, v>
    """
    f = (
        inertia_form(s, domain, u_prev, v, rho, dt) # intertia form for previous displacement
        - inertia_form(s, domain, u, v, rho, dt) # current displacement
        + rho * wp.dot(gravity, v(s)) # gravity term
    )

    return f


@fem.integrand
def kinetic_potential_energy(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
    rho: float,
    dt: float,
    gravity: wp.vec3,
):
    """
    Computes the kinetic potential energy for the displacement field.

    Args:
        s: sample
        domain: domain to integrate over
        u: displacement at sample s
        v: previous displacement at sample s
        rho: density
        dt: timestep
        gravity: gravity

    Returns:
        The kinetic potential energy
    """
    du = u(s) # current displacement
    dv = v(s) # previous displacement

    # computing kinetic potential energy
    E = rho * (0.5 * wp.dot(du - dv, du - dv) / (dt * dt) - wp.dot(du, gravity))

    # return the kinetic potential energy
    return E


@wp.kernel(enable_backward=True)
def scale_lame(
    lame_out: wp.array(dtype=wp.vec2),
    lame_ref: wp.vec2,
    scale: wp.array(dtype=float),
):
    """
    Scale the lame parameters by a factor of scale. 

    Args:
        lame_out: output lame parameters
        lame_ref: reference lame parameters
        scale: scale factor

    Returns:
        The scaled lame parameters
    """
    i = wp.tid()
    lame_out[i] = lame_ref * scale[i]


class DisplacementPotential:
    """Base class for additional potentials that depend only on the displacement field"""

    def __init__(self, sim):
        self.sim = weakref.proxy(sim)

    def prepare_newton_step(self, dt, tape):
        pass

    def prepare_frame(self, dt):
        pass

    def init_constant_forms(self):
        pass

    def add_energy(self, E_u: wp.array):
        pass

    def add_hessian(self, lhs: sp.BsrMatrix):
        pass

    def add_forces(self, rhs: wp.array, _tape):
        pass


class Deformable:
    def __init__(
        self,
        geo: fem.Geometry,
        active_cells: wp.array | None = None,
        *,
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
        collision_radius: float = 0.5 / 64.0,
        collision_detection_ratio: float = 2.0,
        friction: float = 0.2,
        friction_reg: float = 0.1,
        friction_fluid: float = 0.01,
        ground: bool = True,
        ground_height: float = -1.0,
        self_immunity_radius_ratio: float = 4.0,
    ):
        self.geo = geo

        self.degree = degree
        self.serendipity = serendipity
        self.discontinuous = discontinuous
        self.n_newton = n_newton
        self.newton_tol = newton_tol
        self.cg_tol = cg_tol
        self.cg_iters = cg_iters
        self.n_backtrack = n_backtrack
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.gravity_magnitude = gravity
        self.up_axis_index = up_axis
        self.density = density
        self.dt = dt
        self.quasi_quasistatic = quasi_quasistatic
        self.neo_hookean = neo_hookean
        self.dg_jump_pen = dg_jump_pen
        self.quiet = quiet
        self.lumped_mass = lumped_mass
        self.fp64 = fp64
        self.matrix_free = matrix_free
        self.collision_stiffness = collision_stiffness
        self.collision_radius = collision_radius
        self.collision_detection_ratio = collision_detection_ratio
        self.friction = friction
        self.friction_reg = friction_reg
        self.friction_fluid = friction_fluid
        self.ground = ground
        self.ground_height = ground_height
        self.self_immunity_radius_ratio = self_immunity_radius_ratio

        if self.has_discontinuities() and not self.supports_discontinuities():
            raise TypeError(f"Simulator of type {type(self)} does not support discontinuities")

        if self.matrix_free and not self.supports_matrix_free():
            raise TypeError(f"Simulator of type {type(self)} does not support matrix-free solves")

        # Full geometry by default; optionally restrict to an active-cell mask.
        self.geo_partition = fem.Cells(geo).geometry_partition
        self.cells: wp.array | None = None

        if active_cells is not None:
            geo_partition = fem.ExplicitGeometryPartition(geo, cell_mask=active_cells)
            if geo_partition.cell_count() < geo.cell_count():
                self.geo_partition = geo_partition
                self.cells = self.geo_partition._cells

        if not self.quiet:
            print(f"Active cells: {self.geo_partition.cell_count()}")

        self.up_axis = np.zeros(3)
        self.up_axis[self.up_axis_index] = 1.0
        self.gravity = -self.gravity_magnitude * self.up_axis

        # reference Lame parameters for linear elasticity, stored as 2 vector (lambda, mu)
        self.lame_ref = wp.vec2(
            young_modulus
            / (1.0 + poisson_ratio)
            * np.array([poisson_ratio / (1.0 - 2.0 * poisson_ratio), 0.5])
        )

        typical_length = 1.0
        self.typical_stiffness = max(
            density * gravity * typical_length,
            min(
                young_modulus,  # handle no-gravity, quasistatic case
                density * typical_length**2 / (dt**2),  # handle no-gravity, dynamic case
            ),
        )

        # set up line search criterion (accept vs reject)
        self._ls = LineSearchNaiveCriterion(self)
        # builds interpolation basis for displacement field (defines how nodal
        # DOFS map to values inside each element)
        self._init_displacement_basis()

        self.energy_potentials: list[DisplacementPotential] = []

        self._collision_projector_form: fem.operator.Integrand | None = None
        self._collision_projector_args = {}

        self.log: TextIO | None = None

    def has_discontinuities(self) -> bool:
        return isinstance(self.geo, fem.AdaptiveNanogrid)

    def supports_discontinuities(self) -> bool:
        return False

    def supports_matrix_free(self) -> bool:
        return False

    def add_energy_potential(self, potential: DisplacementPotential):
        self.energy_potentials.append(potential)

    def _init_displacement_basis(self):
        """
        Initializes the displacement basis. Lagrange has more DOFs and a richer approximation
        while Serendipity is a reduced version of lagrange, that drops many interior nodes.
        """
        element_basis = fem.ElementBasis.SERENDIPITY if self.serendipity else fem.ElementBasis.LAGRANGE
        self._displacement_basis = fem.make_polynomial_basis_space(
            self.geo,
            degree=self.degree,
            element_basis=element_basis,
            discontinuous=self.discontinuous,
        )

    def set_displacement_basis(self, basis: fem.BasisSpace | None = None):
        if basis is None:
            self._init_displacement_basis()
        else:
            self._displacement_basis = basis

    def init_displacement_space(self, side_subdomain: fem.Domain | None = None):
        """
        Allocates the displacement space and related fields.
        """
        # create 3d vectors for the active cells (defines layout)
        u_space = fem.make_collocated_function_space(self._displacement_basis, dtype=wp.vec3)
        u_space_partition = fem.make_space_partition(
            space_topology=self._displacement_basis.topology,
            geometry_partition=self.geo_partition,
            with_halo=False,
        )

        # Defines some fields (state arrays) over our function spaces
        self.u_field = u_space.make_field(space_partition=u_space_partition)  # displacement
        self.du_field = u_space.make_field(space_partition=u_space_partition)  # displacement delta
        self.du_prev = u_space.make_field(space_partition=u_space_partition)  # displacement delta
        self.force_field = u_space.make_field(
            space_partition=u_space_partition
        )  # total force field -- for collision filtering

        # Since our spaces are constant, we can also predefine the test/trial functions that we will need for integration
        self.u_trial = fem.make_trial(space=u_space, space_partition=u_space_partition)
        self.u_test = fem.make_test(space=u_space, space_partition=u_space_partition)

        self.displacement_quadrature = fem.RegularQuadrature(self.u_test.domain, order=2 * self.degree)
        # Alias used by collision code / Mixed FEM ports
        self.vel_quadrature = self.displacement_quadrature

        # DG style integration on sides for discontinuous elements
        if self.has_discontinuities():
            if side_subdomain is None:
                sides = fem.Sides(self.geo)
            else:
                sides = side_subdomain

            self.u_side_trial = fem.make_trial(space=u_space, space_partition=u_space_partition, domain=sides)
            self.u_side_test = fem.make_test(space=u_space, space_partition=u_space_partition, domain=sides)

            self.side_quadrature = fem.RegularQuadrature(self.u_side_test.domain, order=2 * self.degree)
        else:
            self.side_quadrature = None

        # Create material parameters space with same basis as deformation field
        lame_space = fem.make_polynomial_space(self.geo, dtype=wp.vec2)

        self.lame_field = lame_space.make_field()
        self.lame_field.dof_values.fill_(self.lame_ref)

    def set_boundary_condition(
        self,
        boundary_projector_form,
        boundary_displacement_form=None):
        """
        Builds the Dirichlet boundary condition projector, which boundary DOFs are fixed 
        and to what value. 
        """
        # for the function space of the displacement field 
        u_space = self.u_field.space

        # Displacement boundary conditions (restrit to boundary faces)
        boundary = fem.BoundarySides(self.geo_partition)

        # test/trial functions on the boundary domain, so we can assemble which boundary nodes are constrained
        u_bd_test = fem.make_test(
            space=u_space,
            space_partition=self.u_test.space_partition,
            domain=boundary,
        )
        u_bd_trial = fem.make_trial(
            space=u_space,
            space_partition=self.u_test.space_partition,
            domain=boundary,
        )

        self.v_bd_rhs = None
        # if a form is provided, we use it to compute the right hand side of the boundary condition
        if boundary_displacement_form is not None:
            self.v_bd_rhs = fem.integrate(
                boundary_displacement_form,
                fields={"v": u_bd_test},
                assembly="nodal",
                output_dtype=wp.vec3f,
            )
        # integrate the boundary projector form to get the matrix to project onto constrained DOFs
        self.v_bd_matrix = fem.integrate(
            boundary_projector_form,
            fields={"u": u_bd_trial, "v": u_bd_test},
            assembly="nodal",
            output_dtype=float,
        )

        fem.normalize_dirichlet_projector(self.v_bd_matrix, self.v_bd_rhs)

    def set_fixed_points_condition(
        self,
        fixed_points_projector_form
    ):
        """
        Builds the Dirichlet boundary condition for fixed points only. 
        """

        self.v_bd_rhs = None
        # assemble matrix for fixed points only 
        self.v_bd_matrix = fem.integrate(
            fixed_points_projector_form,
            fields={"u": self.u_trial, "v": self.u_test, "u_cur": self.u_field},
            assembly="nodal",
            output_dtype=float,
        )

        fem.normalize_dirichlet_projector(self.v_bd_matrix, self.v_bd_rhs)

    def set_fixed_points_displacement(
        self,
        fixed_points_displacement_field=None,
    ):
        """
        Sets the displacement of fixed points to a target value instead of 0. 
        """
        bd_field = self.u_test.space.make_field(space_partition=self.u_test.space_partition)
        fem.interpolate(
            fixed_points_displacement_field,
            dest=bd_field,
            fields={"u_cur": self.u_field},
        )

        self.v_bd_rhs = bd_field.dof_values

    def init_constant_forms(self):
        """
        Initializes the constant forms for the linear system (these are forms
        that do not change after object initialization)
        """
        # builds the inertia matrix A
        if self.lumped_mass:
            # diagonal mass matrix (approximate)
            self.A = fem.integrate(
                inertia_form,
                fields={"u": self.u_trial, "v": self.u_test},
                values={"rho": self.density, "dt": self.dt},
                output_dtype=float,
                assembly="nodal",
            )
        else:
            self.A = fem.integrate(
                inertia_form,
                fields={"u": self.u_trial, "v": self.u_test},
                values={"rho": self.density, "dt": self.dt},
                output_dtype=float,
                quadrature=self.displacement_quadrature,
            )
        # if discontinuous penalty is used, add to inertia matrix
        if self.side_quadrature is not None and self.dg_jump_pen > 0.0:
            self.A += fem.integrate(
                dg_penalty_form,
                fields={"u": self.u_side_trial, "v": self.u_side_test},
                values={"k": self.typical_stiffness * self.dg_jump_pen},
                quadrature=self.side_quadrature,
                output_dtype=float,
            )
        # finalize the matrix's nonzero pattern
        self.A.nnz_sync()

        for potential in self.energy_potentials:
            potential.init_constant_forms()

    def project_constant_forms(self): 
        """
        Projects the constant forms onto the constrained DOFs. 
        """
        pass

    def constraint_free_rhs(self, dt=None, with_external_forces=True, tape=None):
        """
        Builds the base right hand side before applying boundary conditions. 

        Args:
            dt: timestep
            with_external_forces: whether to include external forces
            tape: tape to record gradients

        Returns:
            The base right hand side
        """
        # gravity only if external forces are included
        gravity = self.gravity if with_external_forces else wp.vec3(0.0)

        # Quasi-quasistatic: normal dt in lhs (trust region), large dt in rhs (quasistatic)
        # with_gradient: whether to record gradients
        with_gradient = tape is not None
        rhs_tape = wp.Tape() if tape is None else tape
        rhs = wp.zeros(
            dtype=wp.vec3,
            requires_grad=with_gradient,
            shape=self.u_test.space_partition.node_count(),
        )

        with rhs_tape:
            # add inertia and gravity terms
            fem.integrate(
                displacement_rhs_form,
                fields={"u": self.du_field, "u_prev": self.du_prev, "v": self.u_test},
                values={"rho": self.density, "dt": self._step_dt(), "gravity": gravity},
                output=rhs,
                quadrature=self.displacement_quadrature,
                kernel_options={"enable_backward": True},
            )
            # for discontinuous elements, add penalty terms for forces to resist unwanted displacement jumps
            if self.side_quadrature is not None and self.dg_jump_pen > 0.0:
                # add discontinuous penalty terms
                fem.integrate(
                    dg_penalty_form,
                    fields={"u": self.u_field.trace(), "v": self.u_side_test},
                    values={"k": -self.typical_stiffness * self.dg_jump_pen},
                    quadrature=self.side_quadrature,
                    output=rhs,
                    add=True,
                    kernel_options={"enable_backward": True},
                )

        if with_external_forces: # if external forces are included, add potential forces
            for potential in self.energy_potentials:
                potential.add_forces(rhs, rhs_tape)

        return rhs

    def constraint_free_lhs(self):
        """
        builds base newton matrix before fixed point constraints/elasticity are applied
        """
        # copy inertia matrix
        lhs = sp.bsr_copy(self.A)

        # add each potentials hessian to the base newton matrix 
        for potential in self.energy_potentials:
            potential.add_hessian(lhs)

        return lhs

    def run_frame(self):
        """
        Runs one frame of simulation
        """
        # essentially saving the old frame and resetting du_field for next frame
        (self.du_field, self.du_prev) = (self.du_prev, self.du_field)

        # in quasi-quasistatic mode, reset du_prev to zero - this removes the effect of the previous frame
        if self.quasi_quasistatic:
            self.du_prev.dof_values.zero_()

        self.prepare_frame()  # computes initial guess for next frame (including potentials)

        tol = self.newton_tol**2  # sets tolerance for newton's method

        def host_read(tup):
            """
            Helper function to read values from wp.array to numpy array on CPU
            """
            return (x[:1].numpy()[0] if isinstance(x, wp.array) else x for x in tup)

        E_cur, C_cur = host_read(self.evaluate_energy())  # evaluates current total energy and constraint residual
        cumulative_time = 0.0

        # prints initial guess for energy and constraint residual if not quiet
        if not self.quiet:
            print(f"Newton initial guess: E={E_cur}, Cr={C_cur}")
        if self.log:
            mean_displ = np.mean(np.linalg.norm(self.du_field.dof_values.numpy(), axis=1))
            print(
                "\t".join(str(x) for x in (0, E_cur, C_cur, mean_displ, 0.0, 0.0, cumulative_time)),
                file=self.log,
            )

        # runs for n_newton iterations
        for k in range(self.n_newton):
            with wp.ScopedTimer(f"Iter {k}", print=False) as timer:
                E_ref, C_ref = E_cur, C_cur  # stores our current guesses as references
                self.checkpoint_newton_values()  # saves a snapshot of current state

                self.prepare_newton_step()
                rhs = self.newton_rhs()
                lhs = self.newton_lhs()
                delta_fields = self.solve_newton_system(lhs, rhs)

                self.apply_newton_deltas(delta_fields)  # applies the displacement deltas
                E_cur, C_cur = host_read(self.evaluate_energy())

                ddu = delta_fields[0]  # displacement delta storage
                step_size = wp.utils.array_inner(ddu, ddu) / (1 + ddu.shape[0])  # average correction magnitude

                # linear model
                self._ls.build_linear_model(lhs, rhs, delta_fields)  # builds model of energy change

                # Line search
                alpha = 1.0
                for _j in range(self.n_backtrack):
                    if self._ls.accept(alpha, E_cur, C_cur, E_ref, C_ref):  # if we accept the step
                        break

                    alpha = 0.5 * alpha  # try again with a smaller step
                    self.apply_newton_deltas(delta_fields, alpha=alpha)
                    E_cur, C_cur = host_read(self.evaluate_energy())

                if not self.quiet:
                    print(f"Newton iter {k}: E={E_cur}, Cr={C_cur}, step size {np.sqrt(step_size)}, alpha={alpha}")

            cumulative_time += timer.elapsed
            if self.log:
                mean_displ = np.mean(np.linalg.norm(self.du_field.dof_values.numpy(), axis=1))
                print(
                    "\t".join(
                        str(x)
                        for x in (
                            k + 1,
                            E_cur,
                            C_cur,
                            mean_displ,
                            step_size,
                            alpha,
                            cumulative_time,
                        )
                    ),
                    file=self.log,
                )

            if step_size < tol:
                break

    def prepare_newton_step(self, tape=None):
        # calls prepare_newton_step for each potential
        for potential in self.energy_potentials:
            potential.prepare_newton_step(self.dt, tape)

    def prepare_frame(self):
        self.compute_initial_guess()
        for potential in self.energy_potentials:
            potential.prepare_frame(self.dt)

    def checkpoint_newton_values(self):
        self._u_cur = wp.clone(self.u_field.dof_values)
        self._du_cur = wp.clone(self.du_field.dof_values)

    def apply_newton_deltas(self, delta_fields, alpha=1.0):
        # Restore checkpoint
        wp.copy(src=self._u_cur, dest=self.u_field.dof_values)
        wp.copy(src=self._du_cur, dest=self.du_field.dof_values)

        # Add to total displacement
        if alpha == 0.0:
            return

        delta_du = delta_fields[0]
        array_axpy(x=delta_du, y=self.u_field.dof_values, alpha=alpha)
        array_axpy(x=delta_du, y=self.du_field.dof_values, alpha=alpha)

    def _step_dt(self):
        # In fake quasistatic mode, use a large timestep for the rhs computation
        # Note that self.dt is still use to compute lhs (inertia matrix)
        return 1.0e6 if self.quasi_quasistatic else self.dt

    def evaluate_energy(self, E_u=None, cr=None):
        """Evaluates the energy of the system."""
        if E_u is None:
            E_u = wp.zeros(shape=(1,), dtype=float)  # create a storage array

        E_u = fem.integrate(  # integrate the kinetic and potential energy terms over the domain
            kinetic_potential_energy,
            quadrature=self.displacement_quadrature,
            fields={"u": self.du_field, "v": self.du_prev},
            values={
                "rho": self.density,
                "dt": self._step_dt(),
                "gravity": self.gravity,
            },
            output=E_u,
            add=True,
        )
        # if we have discontinuities, add penalty forces
        if self.side_quadrature is not None and self.dg_jump_pen > 0.0:
            fem.integrate(
                dg_penalty_form,
                fields={"u": self.u_field.trace(), "v": self.u_field.trace()},
                values={"k": 0.5 * self.typical_stiffness * self.dg_jump_pen},
                quadrature=self.side_quadrature,
                output=E_u,
                add=True,
            )

        for potential in self.energy_potentials: # add energy from each potential
            potential.add_energy(E_u)

        return E_u, cr # return the total energy and constraint residual 

    def _filter_forces(self, u_rhs, tape, temporary_store=None):
        """
        Filters forces through the collision projector by removing components in constrained directions.
        """

        if self._collision_projector_form is not None:
            # update collision projector, if required
            self.force_field.dof_values = u_rhs
            # produces a diagonal matrix of projection coeff
            self.v_bd_matrix = fem.integrate(
                self._collision_projector_form,
                fields={
                    "u": self.u_trial,
                    "v": self.u_test,
                    "u_cur": self.u_field,
                    "f": self.force_field,
                },
                values=self._collision_projector_args,
                assembly="nodal",
                output_dtype=float,
            )

            fem.normalize_dirichlet_projector(self.v_bd_matrix)
            self.project_constant_forms()

        # temp storage 
        orig_rhs = fem.borrow_temporary_like(u_rhs, temporary_store)
        orig_rhs.array.assign(u_rhs)

        # tape for adjoint computation 
        proj_tape = wp.Tape() if tape is None else tape
        # perform projection
        with proj_tape:
            diff_bsr_mv(
                A=self.v_bd_matrix,
                x=orig_rhs.array,
                y=u_rhs,
                alpha=-1.0,
                beta=1.0,
                self_adjoint=True,
            )

    def compute_initial_guess(self):
        # Start from last frame pose, known good state
        self.du_field.dof_values.zero_()

    def scale_lame_field(self, stiffness_scale_array: wp.array):
        wp.launch(
            scale_lame,
            dim=self.lame_field.dof_values.shape[0],
            inputs=[
                self.lame_field.dof_values,
                self.lame_ref,
                stiffness_scale_array,
            ],
        )

    def reset_fields(self):
        self.u_field.dof_values.zero_()


class ClassicFEM(Deformable):
    """
    Extends the Deformable class to implement models that include actual 
    elastic Newton physics. It contains models for Neo-Hookean and Corotational elasticity, 
    including energy, forces, Hessian and stress. 
    """
    def __init__(self, geo: fem.Geometry, active_cells: wp.array | None = None, **kwargs):
        super().__init__(geo, active_cells, **kwargs)

        self._ls = LineSearchUnconstrainedArmijoCriterion(self)

        self._make_elasticity_forms()

    def _make_elasticity_forms(self):
        if self.neo_hookean:
            # neo-hookean elasticity
            self.elastic_energy = ClassicFEM.nh_elastic_energy
            self.elastic_forces = ClassicFEM.nh_elastic_forces
            self.elasticity_hessian = ClassicFEM.nh_elasticity_hessian
            self.stress_field = ClassicFEM.nh_stress_field
        else:
            # corotational elasticity
            self.elastic_energy = ClassicFEM.cr_elastic_energy
            self.elastic_forces = ClassicFEM.cr_elastic_forces
            self.elasticity_hessian = ClassicFEM.cr_elasticity_hessian
            self.elastic_energy_dg_sip = ClassicFEM.cr_dg_elastic_energy
            self.elastic_forces_dg_sip = ClassicFEM.cr_dg_elastic_forces
            self.elasticity_hessian_dg_sip = ClassicFEM.cr_dg_elasticity_hessian
            self.stress_field = ClassicFEM.cr_stress_field

        # cached polar decomposition
        self._svd_U = None
        self._svd_sig = None
        self._svd_V = None
        self._svd_sides_U = None
        self._svd_sides_sig = None
        self._svd_sides_V = None

    def _elasticity_form_arguments(self):
        if self.neo_hookean:
            return {}

        # cached polar decomposition
        return {"Us": self._svd_U, "Vs": self._svd_V, "sigs": self._svd_sig}

    def _sides_elasticity_form_arguments(self):
        if self.neo_hookean:
            return {}

        # cached polar decomposition
        return {
            "Us": self._svd_sides_U,
            "Vs": self._svd_sides_V,
            "sigs": self._svd_sides_sig,
        }

    def supports_discontinuities(self) -> bool:
        return not self.neo_hookean

    def supports_matrix_free(self) -> bool:
        return not self.has_discontinuities()

    def init_strain_spaces(self):
        """
        Initializes space for strain calculations.
        """
        self.elasticity_quadrature = self.displacement_quadrature
        self.strain_quadrature = self.displacement_quadrature
        self.constraint_field = self.interpolated_constraint_field
        self._constraint_field_restriction = fem.make_restriction(
            self.constraint_field, space_restriction=self.u_test.space_restriction
        )

    def set_strain_basis(self, strain_basis: fem.BasisSpace):
        del strain_basis

    def prepare_newton_step(self, tape: wp.Tape = None):
        """
        newton iteration preparation
        """
        # updates every registered potential
        super().prepare_newton_step(tape=tape)

        # cache polar decomp for non-neohookean materials
        if not self.neo_hookean:
            if tape is not None:
                with tape:
                    self._cache_polar_decomposition()
            else:
                self._cache_polar_decomposition()

    def evaluate_energy(self, E_u=None, cr=None):
        """
        Evaluates the energy of the system.
        """
        E_u, c_r = super().evaluate_energy(E_u=E_u) # evaluate kinetic and potential energy

        # evaluate elastic energy
        fem.integrate(
            self.elastic_energy,
            quadrature=self.elasticity_quadrature,
            fields={"u_cur": self.u_field, "lame": self.lame_field},
            output=E_u,
            add=True,
        )

        # evaluate elastic energy on side quadrature points if discontinuities are present
        if self.side_quadrature is not None:
            fem.integrate(
                self.elastic_energy_dg_sip,
                quadrature=self.side_quadrature,
                fields={
                    "u_cur": self.u_field.trace(),
                    "lame": self.lame_field.trace(),
                },
                add=True,
                output=E_u,
            )

        return E_u, c_r

    def newton_lhs(self):
        """
        Constructs the matrix on the LHS of the newton system. 
        """
        if self.matrix_free:
            return None

        # integrate elasticity hessian to get matrix
        # output is a sparse BSR stiffness matrix
        u_matrix = fem.integrate(
            self.elasticity_hessian,
            quadrature=self.elasticity_quadrature,
            fields={
                "u_cur": self.u_field,
                "u": self.u_trial,
                "v": self.u_test,
                "lame": self.lame_field,
            },
            values=self._elasticity_form_arguments(),
            output_dtype=float,
        )
        # if we have discontinuities, add elasticity hessian 
        if self.side_quadrature is not None:
            fem.integrate(
                self.elasticity_hessian_dg_sip,
                quadrature=self.side_quadrature,
                fields={
                    "u_cur": self.u_field.trace(),
                    "u": self.u_side_trial,
                    "v": self.u_side_test,
                    "lame": self.lame_field.trace(),
                },
                values=self._sides_elasticity_form_arguments(),
                add=True,
                output=u_matrix,
            )

        # constraint_free_lhs is the matrix of the constraint free system
        u_matrix += self.constraint_free_lhs()
        fem.dirichlet.project_system_matrix(u_matrix, self.v_bd_matrix) # projcet to enforce boundary conditions

        return u_matrix


    def newton_rhs(self, tape: wp.Tape = None):
        """
        Builds the force vector on the RHS for classic FEM newton solve
        """
        # force vector for the constraint free system 
        u_rhs = self.constraint_free_rhs(tape=tape)

        # tape for adjoint computation 
        rhs_tape = wp.Tape() if tape is None else tape
        with rhs_tape: # evaluate elastic forces 
            fem.integrate(
                self.elastic_forces,
                quadrature=self.elasticity_quadrature,
                fields={
                    "u_cur": self.u_field,
                    "v": self.u_test,
                    "lame": self.lame_field,
                },
                values=self._elasticity_form_arguments(),
                output=u_rhs,
                add=True,
                kernel_options={"enable_backward": True},
            )

            if self.side_quadrature is not None: # if we have discontinuities, add elastic forces on side quadrature points
                fem.integrate(
                    self.elastic_forces_dg_sip,
                    quadrature=self.side_quadrature,
                    fields={
                        "u_cur": self.u_field.trace(),
                        "v": self.u_side_test,
                        "lame": self.lame_field.trace(),
                    },
                    values=self._sides_elasticity_form_arguments(),
                    add=True,
                    output=u_rhs,
                )
        # save a copy of u_rhs before boundary conditions are applied 
        self._minus_dE_du = wp.clone(u_rhs, requires_grad=False) 
        # remove forces in constrained directions 
        self._filter_forces(u_rhs, tape=tape) 

        return u_rhs

    def _cache_polar_decomposition(self, requires_grad: bool = False):
        """
        Precomputes the SVD of the deformation gradient (F = U * diag(sig) * V^T) at every
        elasticity quadrature point, storing the results in self._svd_U/_svd_sig/_svd_V.

        Args:
            requires_grad: whether to allocate the cached arrays with gradients enabled
                (needed when recording the computation for autodiff)
        """
        # precompute polar decomposition
        qp_count = self.elasticity_quadrature.total_point_count()
        self._svd_U = wp.empty(dtype=wp.mat33, shape=qp_count, requires_grad=requires_grad)
        self._svd_sig = wp.empty(dtype=wp.vec3, shape=qp_count, requires_grad=requires_grad)
        self._svd_V = wp.empty(dtype=wp.mat33, shape=qp_count, requires_grad=requires_grad)

        fem.interpolate(
            ClassicFEM._polar_decomposition,
            quadrature=self.elasticity_quadrature,
            fields={"u": self.u_field},
            values={"Us": self._svd_U, "Vs": self._svd_V, "sigs": self._svd_sig},
        )

        if self.side_quadrature is not None:
            # polar decompostion on side quadrature points
            qp_count = self.side_quadrature.total_point_count()
            self._svd_sides_U = wp.empty(dtype=wp.mat33, shape=qp_count, requires_grad=requires_grad)
            self._svd_sides_sig = wp.empty(dtype=wp.vec3, shape=qp_count, requires_grad=requires_grad)
            self._svd_sides_V = wp.empty(dtype=wp.mat33, shape=qp_count, requires_grad=requires_grad)

            fem.interpolate(
                ClassicFEM._polar_decomposition,
                quadrature=self.side_quadrature,
                fields={"u": self.u_field.trace()},
                values={
                    "Us": self._svd_sides_U,
                    "Vs": self._svd_sides_V,
                    "sigs": self._svd_sides_sig,
                },
            )

    @staticmethod
    def _solve_fp64(lhs, rhs, res, maxiters, tol=None):
        """
        Solves the linear system using 64 bit precision. 
        """
        lhs64 = sp.bsr_copy(lhs, scalar_type=wp.float64)
        rhs64 = wp.empty(shape=rhs.shape, dtype=wp.vec3d, device=rhs.device)
        wp.utils.array_cast(in_array=rhs, out_array=rhs64)

        res64 = wp.zeros_like(rhs64)
        bsr_cg(
            A=lhs64,
            b=rhs64,
            x=res64,
            quiet=True,
            tol=tol,
            max_iters=maxiters,
        )

        wp.utils.array_cast(in_array=res64, out_array=res)

        return res
    
    def solve_newton_system(self, lhs, rhs):
        """
        solves the linear system assembled by newton_lhs and newton_Rhs
        """
        if self.fp64:
            res = wp.empty_like(rhs)
            ClassicFEM._solve_fp64(lhs, rhs, res, maxiters=self.cg_iters, tol=self.cg_tol)
            return (res,)

        if lhs is None:
            lhs = self._make_matrix_free_newton_linear_operator()
            use_diag_precond = False
        else:
            use_diag_precond = True

        res = wp.zeros_like(rhs)
        bsr_cg(
            A=lhs,
            b=rhs,
            x=res,
            quiet=True,
            tol=self.cg_tol,
            max_iters=self.cg_iters,
            use_diag_precond=use_diag_precond,
        )
        return (res,)
    
    def record_adjoint(self, tape):
        """perform and record the adjoint computation using stress interpolation for loss
        """
        # The forward Newton is finding a root of rhs(q, p) = 0 with q = (u, S, R, lambda)
        # so drhs/dp = drhs/dq dq/dp + drhs/dp = 0
        # [- drhs/dq] dq/dp = drhs/dp
        # lhs dq/dp = drhs/dp
        # rebuild the system
        self.prepare_newton_step(tape=tape)
        rhs = self.newton_rhs(tape=tape)
        lhs = self.newton_lhs()

        # solve the system
        def solve_backward():
            adj_res = self.u_field.dof_values.grad
            ClassicFEM._solve_fp64(lhs, adj_res, rhs.grad, maxiters=self.cg_iters)

        # record the stress interpolation 
        tape.record_func(
            solve_backward,
            arrays=[
                self.u_field.dof_values,
                rhs,
            ],
        )

        # So we can compute stress-based losses
        with tape:
            self.interpolate_constraint_field()

    def interpolate_constraint_field(self, strain=False):
        """
        Interpolates the constraint field 
        """
        field = self.strain_field if strain else self.stress_field

        fem.interpolate(
            field,
            fields={
                "u_cur": self.u_field,
                "lame": self.lame_field,
            },
            kernel_options={"enable_backward": True},
            dest=self._constraint_field_restriction,
        )

    @fem.integrand
    def nh_elastic_energy(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        return snh_energy(F, lame(s))

    @fem.integrand
    def nh_elastic_forces(s: Sample, u_cur: Field, v: Field, lame: Field):
        F = defgrad(u_cur, s)
        tau = fem.grad(v, s)

        return -wp.ddot(tau, snh_stress(F, lame(s)))

    @fem.integrand
    def nh_stress_field(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        return snh_stress(F, lame(s))

    @fem.integrand
    def nh_elasticity_hessian(s: Sample, u_cur: Field, u: Field, v: Field, lame: Field):
        F_s = defgrad(u_cur, s)
        tau_s = fem.grad(v, s)
        sig_s = fem.grad(u, s)
        lame_s = lame(s)

        return snh_hessian_proj_analytic(F_s, tau_s, sig_s, lame_s)

    @fem.integrand
    def cr_elastic_energy(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        S = symmetric_strain(F)
        return hooke_energy(S, lame(s))

    @fem.integrand
    def cr_elastic_forces(
        s: Sample,
        u_cur: Field,
        v: Field,
        lame: Field,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        U = Us[s.qp_index]
        sig = sigs[s.qp_index]
        V = Vs[s.qp_index]

        S = symmetric_strain(sig, V)
        tau = symmetric_strain_delta(U, sig, V, fem.grad(v, s))
        return -wp.ddot(tau, hooke_stress(S, lame(s)))

    @fem.integrand
    def cr_stress_field(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        S = symmetric_strain(F)
        return hooke_stress(S, lame(s))

    @fem.integrand
    def cr_elasticity_hessian(
        s: Sample,
        u_cur: Field,
        u: Field,
        v: Field,
        lame: Field,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        U = Us[s.qp_index]
        sig = sigs[s.qp_index]
        V = Vs[s.qp_index]

        S_s = symmetric_strain(sig, V)
        tau_s = symmetric_strain_delta(U, sig, V, fem.grad(v, s))
        sig_s = symmetric_strain_delta(U, sig, V, fem.grad(u, s))
        lame_s = lame(s)

        return hooke_hessian(S_s, tau_s, sig_s, lame_s)

    @fem.integrand
    def strain_field(s: Sample, u_cur: Field, lame: Field):
        F = defgrad(u_cur, s)
        return symmetric_strain(F)

    @fem.integrand
    def _polar_decomposition(
        s: fem.Sample,
        u: fem.Field,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        F = defgrad_avg(u, s)

        U = wp.mat33()
        D = wp.vec3()
        V = wp.mat33()
        wp.svd3(F, U, D, V)

        Us[s.qp_index] = U
        sigs[s.qp_index] = D
        Vs[s.qp_index] = V

    def _make_matrix_free_newton_linear_operator(self):
        x_field = self.u_field.space.make_field(space_partition=self.u_field.space_partition)

        temporary_store = fem.TemporaryStore()

        if self.neo_hookean:

            def matvec(x: wp.array, y: wp.array, z: wp.array, alpha: float, beta: float):
                """Compute z = alpha * A @ x + beta * y"""
                wp.copy(src=x, dest=x_field.dof_values)
                fem.integrate(
                    self._nh_matrix_free_lhs_form,
                    quadrature=self.elasticity_quadrature,
                    fields={
                        "u_cur": self.u_field,
                        "u": x_field,
                        "v": self.u_test,
                        "lame": self.lame_field,
                    },
                    values={"rho": self.density, "dt": self.dt},
                    output=z,
                    temporary_store=temporary_store,
                )

                self._filter_forces(z, tape=None, temporary_store=temporary_store)
                fem.linalg.array_axpy(x=y, y=z, alpha=beta, beta=alpha)

        else:

            def matvec(x: wp.array, y: wp.array, z: wp.array, alpha: float, beta: float):
                """Compute z = alpha * A @ x + beta * y"""
                wp.copy(src=x, dest=x_field.dof_values)

                fem.integrate(
                    self._cr_matrix_free_lhs_form,
                    quadrature=self.elasticity_quadrature,
                    fields={
                        "u": x_field,
                        "v": self.u_test,
                        "lame": self.lame_field,
                    },
                    values={
                        "rho": self.density,
                        "dt": self.dt,
                        **self._elasticity_form_arguments(),
                    },
                    output=z,
                    temporary_store=temporary_store,
                )

                self._filter_forces(z, tape=None, temporary_store=temporary_store)
                fem.linalg.array_axpy(x=y, y=z, alpha=beta, beta=alpha)

        # dry run to make sure temporary_store is populated
        matvec(
            x=wp.zeros_like(x_field.dof_values),
            y=wp.zeros_like(x_field.dof_values),
            z=wp.zeros_like(x_field.dof_values),
            alpha=0.0,
            beta=0.0,
        )

        n = x_field.dof_values.shape[0] * 3
        linop = LinearOperator(
            shape=(n, n),
            dtype=float,
            device=x_field.dof_values.device,
            matvec=matvec,
        )
        linop._field = x_field  # prevent garbage collection
        return linop

    @fem.integrand
    def _cr_matrix_free_lhs_form(
        s: Sample,
        domain: Domain,
        u: Field,
        v: Field,
        lame: Field,
        rho: float,
        dt: float,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        U = Us[s.qp_index]
        sig = sigs[s.qp_index]
        V = Vs[s.qp_index]

        S_s = symmetric_strain(sig, V)
        tau_s = symmetric_strain_delta(U, sig, V, fem.grad(u, s))
        sig_s = symmetric_strain_delta(U, sig, V, fem.grad(v, s))
        lame_s = lame(s)

        return hooke_hessian(S_s, tau_s, sig_s, lame_s) + inertia_form(s, domain, u, v, rho, dt)

    @fem.integrand
    def _nh_matrix_free_lhs_form(
        s: Sample,
        domain: Domain,
        u_cur: Field,
        u: Field,
        v: Field,
        lame: Field,
        rho: float,
        dt: float,
    ):
        F_s = defgrad(u_cur, s)
        tau_s = fem.grad(u, s)
        sig_s = fem.grad(v, s)
        lame_s = lame(s)

        return snh_hessian_proj_analytic(F_s, tau_s, sig_s, lame_s) + inertia_form(s, domain, u, v, rho, dt)

    @fem.integrand
    def cr_dg_elastic_energy(
        s: Sample,
        domain: Domain,
        u_cur: Field,
        lame: Field,
    ):
        F_s = defgrad_avg(u_cur, s)

        U = wp.mat33()
        sig = wp.vec3()
        V = wp.mat33()
        wp.svd3(F_s, U, sig, V)

        S_s = symmetric_strain(sig, V)
        lame_s = lame(s)

        normal = fem.normal(domain, s)
        jump_u = fem.outer(fem.jump(u_cur, s), normal)
        R_jump_u = symmetric_strain_delta(U, sig, V, jump_u)

        return 0.5 * fem.measure_ratio(domain, s) * wp.ddot(jump_u, jump_u) * lame_s[0] - wp.ddot(
            R_jump_u, hooke_stress(S_s, lame_s)
        )

    @fem.integrand
    def cr_dg_elastic_forces(
        s: Sample,
        domain: Domain,
        u_cur: Field,
        v: Field,
        lame: Field,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        U = Us[s.qp_index]
        sig = sigs[s.qp_index]
        V = Vs[s.qp_index]

        S_s = symmetric_strain(sig, V)
        tau_s = symmetric_strain_delta(U, sig, V, fem.grad_average(v, s))
        lame_s = lame(s)

        normal = fem.normal(domain, s)
        jump_u = fem.outer(fem.jump(u_cur, s), normal)
        jump_v = fem.outer(fem.jump(v, s), normal)
        R_jump_u = symmetric_strain_delta(U, sig, V, jump_u)
        R_jump_v = symmetric_strain_delta(U, sig, V, jump_v)

        return -(
            fem.measure_ratio(domain, s) * wp.ddot(jump_u, jump_v) * lame_s[0]
            - hooke_hessian(S_s, R_jump_u, tau_s, lame_s)
            - wp.ddot(R_jump_v, hooke_stress(S_s, lame_s))
        )

    @fem.integrand
    def cr_dg_elasticity_hessian(
        s: Sample,
        domain: Domain,
        u_cur: Field,
        u: Field,
        v: Field,
        lame: Field,
        Us: wp.array(dtype=wp.mat33),
        sigs: wp.array(dtype=wp.vec3),
        Vs: wp.array(dtype=wp.mat33),
    ):
        U = Us[s.qp_index]
        sig = sigs[s.qp_index]
        V = Vs[s.qp_index]

        S_s = wp.mat33()
        tau_s = symmetric_strain_delta(U, sig, V, fem.grad_average(v, s))
        sig_s = symmetric_strain_delta(U, sig, V, fem.grad_average(u, s))
        lame_s = lame(s)

        normal = fem.normal(domain, s)
        jump_u = fem.outer(fem.jump(u, s), normal)
        jump_v = fem.outer(fem.jump(v, s), normal)
        R_jump_u = symmetric_strain_delta(U, sig, V, jump_u)
        R_jump_v = symmetric_strain_delta(U, sig, V, jump_v)

        return (
            fem.measure_ratio(domain, s) * wp.ddot(jump_u, jump_v) * lame_s[0]
            - hooke_hessian(S_s, R_jump_u, tau_s, lame_s)
            - hooke_hessian(S_s, R_jump_v, sig_s, lame_s)
        )