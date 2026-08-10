# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import warp as wp
import warp.fem as fem
from warp.fem import Domain, Field, Sample

from .deformable_model import DisplacementPotential, SoftbodySim


@wp.struct
class VolumetricForces:
    """
    A struct that stores a collection of smooth, localized forces distributed through
    a spherical region of the soft body. 

    Params:
        count: the number of active forces
        centers: the center of the spherical application region
        radii: radius where the force is applied 
        forces: total force vector
        tot_weight: the intergrated spatial weight, used for normalization
    """
    count: int 
    centers: wp.array(dtype=wp.vec3)
    radii: wp.array(dtype=float)
    forces: wp.array(dtype=wp.vec3)
    tot_weight: wp.array(dtype=float)


@wp.func
# indicates this is a low-level warp device function that other kernels and integrands can call
def force_weight(x: wp.vec3, forces: VolumetricForces, force_index: int):
    """
    Calculates how strongly force_index affects world position x. 
    """
    # calculate the normalized distance from the force centre to the world position
    r = wp.min(
        wp.length(x - forces.centers[force_index]) / (forces.radii[force_index] + 1.0e-7),
        1.0,
    )
    # calculate the cubic spline weight function
    # this tells you how much of the force is applied at position r from the force centre
    r2 = r * r
    return 2.0 * r * r2 - 3.0 * r2 + 1.0  # cubic spline


@fem.integrand
def force_weight_form(s: Sample, domain: Domain, forces: VolumetricForces, force_index: int):
    """
    A FEM wrapper for force_weight function. It is used to return the force weight for a given force. 

    Args:
        s: the sample point
        domain: domain of sample point
        forces: forces struct
        force_index: the index of the force to calculate the weight for

    Returns:
        the force weight for the given force at the sample point
    """
    # connects the force_weight function to the FEM framework
    return force_weight(domain(s), forces, force_index)


@fem.integrand
def force_action(x: wp.vec3, forces: VolumetricForces, force_index: int, vec: wp.vec3):
    """
    Calculates the action of one force on a vector.

    Args:
        x: the world position where force is evaluated
        forces: forces struct
        force_index: select which force is being evaluated
        vec: the vector to apply the force to 
    
    Returns: 
        the action of a force over the world position vector x
    """
    # action of a force over a vector
    return wp.where(
        forces.tot_weight[force_index] >= 1.0e-6, # check that normalization is valid
        # project the force onto the test vector and apply spatial falloff
        wp.dot(forces.forces[force_index], vec) * force_weight(x, forces, force_index) / forces.tot_weight[force_index], 
        0.0,
    )


@fem.integrand
def external_forces_form(s: Sample, domain: Domain, v: Field, forces: VolumetricForces):
    """
    Calculate the action of all forces on the test vector at the sample point. 

    Args:
        s: the sample point
        domain: the domain of the sample point
        v: the FEM test field 
        forces: collection of volumetirc forces
    
    Returns
        A float: the total action of all forces on the test vector at the sample point
    """
    f = float(0.0) # initialize total force to 0
    x = domain(s) # get the world position of the sample point
    for fi in range(forces.count): # iterate over all forces
        f += force_action(x, forces, fi, v(s)) # add the action of the current force
    return f


@fem.integrand
def external_forces_potential_energy(
    s: Sample,
    domain: Domain,
    u: Field,
    forces: VolumetricForces,
):
    """
    computes the potential energy associated with those forces. 
    Similar to external_forces_form, but returns the negative of the force action. 

    Args: 
        s: the sample point
        domain: the domain of the sample point
        u: FEM displacement field
        forces: collection of volumetric forces

    Returns:
        A float: the potential energy associated with the forces at the sample point
    """
    # return the negative of the force action
    return -external_forces_form(s, domain, u, forces)


class VolumetricForcePotential(DisplacementPotential):
    """
    A potential that applies volumetric forces to the soft body. 

    Params:
        self.forces: a volumetric forces struct 
        self.reserve_count: the number of forces to reserve
    """
    def __init__(self, sim, reserve_count: int = 0):
        super().__init__(sim)

        self.forces = VolumetricForces()
        self.forces.count = 0
        self.reserve(reserve_count)

    def init_constant_forms(self):
        self.update_force_weight()

    def reserve(self, count: int):
        self.forces.forces = wp.empty(shape=(count,), dtype=wp.vec3)
        self.forces.radii = wp.empty(shape=(count,), dtype=float)
        self.forces.centers = wp.empty(shape=(count,), dtype=wp.vec3)
        self.forces.tot_weight = wp.empty(shape=(count,), dtype=float)

    def update_force_weight(self):
        """
        Computes the normalization constant for each active volumetric force. 
        """
        for fi in range(self.forces.count):
            wi = self.forces.tot_weight[fi : fi + 1] # take a slice of the array as the output target for this force
            fem.integrate( # this writes the entry into that slice 
                force_weight_form,
                quadrature=self.sim.vel_quadrature,
                values={
                    "force_index": fi,
                    "forces": self.forces,
                },
                output=wi,
                accumulate_dtype=wp.float32,
            )

    def add_forces(self, rhs, _tape):
        """
        Distributes all active volumetric forces onto FEM nodes and adds to existing 
        newton RHS. 

        Args:
            rhs: the newton RHS
            _tape: the tape to differentiate with respect to
        """
        if self.forces.count > 0:
            # NOT differentiating with respect to external forces
            # Those are assumed to not depend on the geometry
            fem.integrate(
                external_forces_form,
                fields={"v": self.sim.u_test}, # FEM test field 
                values={
                    "forces": self.forces, # collection of forces and their properties
                },
                output_dtype=wp.vec3,
                quadrature=self.sim.vel_quadrature, # evaluates and integrates the force weight
                kernel_options={"enable_backward": False},
                output=rhs,
                add=True,
            )

    def add_energy(self, E_u):
        """
        Adds potential energy associated with the forces to the existing energy
        """
        # integrate the potential energy over the domain of the soft body
        fem.integrate(
            external_forces_potential_energy,
            quadrature=self.sim.vel_quadrature,
            fields={"u": self.sim.du_field},
            values={
                "forces": self.forces,
            },
            output=E_u,
            add=True,
        )


@fem.integrand
def prescribed_position_lhs_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
    stiffness: fem.Field,
): 
    """
    Assembles the spring stiffness H = kI. It tells Newton how the force changes
    when the displacement changes.

    Args:
        s: the sample point
        domain: the domain of the sample point
        u: the trial function
        v: the test function
        stiffness: the stiffness of the spring

    Returns:
        The spring stiffness H = kI
    """
    # trial function
    u_displ = u(s)
    # test function
    v_displ = v(s)
    # returns spring stiffness 
    return stiffness(s) * wp.dot(u_displ, v_displ)


@fem.integrand
def prescribed_position_rhs_form(
    s: fem.Sample,
    domain: fem.Domain,
    u_cur: fem.Field,
    v: fem.Field,
    stiffness: fem.Field,
    target: fem.Field,
):
    """
    Returns the spring force applied to the body at the sample point.
    based on the current position and the target position. 

    Args:
        s: the sample point
        domain: the domain of the sample point
        u_cur: the current displacement field
        v: the test function
        stiffness: the stiffness of the spring
        target: the target position

    Returns:
        The spring force applied to the body at the sample point
    """
    # calculate the current position of the sample point
    pos = u_cur(s) + domain(s)
    # displacement
    v_displ = v(s)
    # target point
    target_pos = target(s)
    # return spring force 
    return stiffness(s) * wp.dot(target_pos - pos, v_displ)


@fem.integrand
def prescribed_position_energy_form(
    s: fem.Sample,
    domain: fem.Domain,
    u_cur: fem.Field,
    stiffness: fem.Field,
    target: fem.Field,
):
    """
    Returns the potential energy associated with the prescribed position. THis is 
    essentially a measure of how far the body is from the target position. 

    Args:
        s: the sample point
        domain: the domain of the sample point
        u_cur: the current displacement field
        stiffness: the stiffness of the spring
        target: the target position

    Returns:
        The potential energy associated with the prescribed position
    """
    # calculate the current position of the sample point
    pos = u_cur(s) + domain(s)
    # store target position
    target_pos = target(s)
    # return potential energy
    return 0.5 * stiffness(s) * wp.length_sq(pos - target_pos)
