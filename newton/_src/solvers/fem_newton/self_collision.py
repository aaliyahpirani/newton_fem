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

from __future__ import annotations

from typing import Any

import warp as wp
import warp.fem as fem
import warp.sparse as sp

try:
    from warp.fem.geometry.closest_point import project_on_tri_at_origin
except ModuleNotFoundError:
    # Warp 1.17+ relocated this helper under the private package path.
    from warp._src.fem.geometry.closest_point import project_on_tri_at_origin

from ...sim import Contacts
from .deformable_model import Deformable, DisplacementPotential

class CollisionHandler:
    def __init__(
        self,
        kinematic_meshes: list[wp.Mesh],
        cp_cell_indices,
        cp_cell_coords,
    ):
        # stores meshes
        self.warp_meshes = kinematic_meshes
        self.cp_cell_indices = cp_cell_indices
        self.cp_cell_coords = cp_cell_coords

        self.collision_quadrature = None
        self.n_contact = 0
        self._pipeline_contacts: Contacts | None = None
        self._pipeline_body_q: wp.array | None = None
        self._pipeline_shape_body: wp.array | None = None
        self._pipeline_particle_radius: wp.array | None = None
        self._pipeline_shape_margin: wp.array | None = None
        self._particle_quadrature = None
        self._n_particles = 0

    def init_collision_detector(self, sim: Deformable):
        """
        Initializes the collision detector.

        Args:
            sim: the softbody simulation

        Returns:
            None
        """
        self.sim = sim

        # initialize all the arrays to work with
        n_cp = self.cp_cell_indices.shape[0]
        collision_quadrature = fem.PicQuadrature(
            domain=sim.vel_quadrature.domain,
            positions=(self.cp_cell_indices, self.cp_cell_coords),
            measures=wp.ones(n_cp, dtype=float),
        )
        self.set_collision_quadrature(collision_quadrature)
        # Frozen copy of the original surface-vertex PIC. Self-collision may replace
        # ``collision_quadrature`` with extra B samples; Newton particle ids still
        # index this original set.
        self._particle_quadrature = collision_quadrature
        self._n_particles = n_cp
        self.n_contact = 0

        # Self-collision can report many face hits per quadrature point.
        max_contacts = 128 * self.cp_cell_indices.shape[0]
        self.collision_indices_a = wp.empty(max_contacts, dtype=int)
        self.collision_indices_b = wp.empty(max_contacts, dtype=int)
        self.collision_normals = wp.empty(max_contacts, dtype=wp.vec3)
        self.collision_kinematic_gaps = wp.empty(max_contacts, dtype=wp.vec3)
        self._warned_contact_overflow = False

        jac_cols = sim.u_field.space_partition.node_count()
        self._collision_jacobian_a = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)
        self._collision_jacobian_b = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)

        self._collision_jacobian = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)
        self._collision_jacobian_t = sp.bsr_zeros(jac_cols, 0, block_type=wp.mat33)

        self._HtH_work_arrays = sp.bsr_mm_work_arrays()
        self._HbHa_work_arrays = sp.bsr_axpy_work_arrays()

        self._collision_stiffness = sim.collision_stiffness * sim.density / n_cp

    def set_collision_quadrature(self, quadrature: fem.PicQuadrature):
        self.collision_quadrature = quadrature

    def set_pipeline_contacts(
        self,
        contacts: Contacts | None,
        *,
        body_q: wp.array[wp.transform] | None,
        shape_body: wp.array[int] | None,
        particle_radius: wp.array[float] | None,
        shape_margin: wp.array[float] | None,
    ) -> None:
        """Store :class:`~newton.Contacts` from :class:`~newton.CollisionPipeline` for this step.

        The pipeline reports geometry only (particle, shape, body-frame point,
        world normal, body velocity). Penalty magnitude stays the FEM
        ``collision_stiffness`` / ``collision_radius`` law.

        Args:
            contacts: Soft-rigid contact buffer, or ``None`` to skip pipeline contacts.
            body_q: Body transforms [m, unitless], shape [body_count]. Unused for
                static shapes (``shape_body == -1``).
            shape_body: Parent body index per shape, shape [shape_count].
            particle_radius: Particle contact radii [m], shape [particle_count].
            shape_margin: Per-shape contact margins [m], shape [shape_count].
        """
        self._pipeline_contacts = contacts
        self._pipeline_body_q = body_q
        self._pipeline_shape_body = shape_body
        self._pipeline_particle_radius = particle_radius
        self._pipeline_shape_margin = shape_margin

    def _sample_particle_world_and_du(self):
        """Interpolate deformed position and Newton increment at original surface verts."""
        device = self.collision_indices_a.device
        pos_cur = wp.empty(self._n_particles, dtype=wp.vec3, device=device)
        fem.interpolate(
            world_position,
            fields={"u": self.sim.u_field},
            dest=pos_cur,
            at=self._particle_quadrature,
        )
        du_cur = wp.empty(self._n_particles, dtype=wp.vec3, device=device)
        fem.interpolate(
            self.sim.du_field,
            dest=du_cur,
            at=self._particle_quadrature,
        )
        return pos_cur, du_cur

    def harvest_pipeline_contact_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_q: wp.array[wp.transform] | None,
        body_com: wp.array[wp.vec3] | None,
        shape_body: wp.array[int] | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Accumulate equal-and-opposite FEM penalty wrenches on proxy bodies.

        Iterates :class:`~newton.Contacts` particle-shape records rather than the
        mixed FEM contact buffer, so self-collision and the optional FEM ground
        plane are not harvested. Static shapes (``shape_body == -1``) are skipped.

        Args:
            body_local_to_proxy_global: Destination-local body id to global proxy
                id, or ``-1`` if the body is not a proxy. Shape [body_count].
            out_body_f: Coupling wrenches [N, N·m], shape [global proxy count].
            body_q: Body transforms [m, unitless], shape [body_count].
            body_com: Body COM in the body frame [m], shape [body_count].
            shape_body: Parent body index per shape, shape [shape_count].
            contacts: Soft-rigid contact buffer, or ``None``.
            dt: Time step size [s].
        """
        out_body_f.zero_()
        if (
            contacts is None
            or body_q is None
            or body_com is None
            or shape_body is None
            or self._particle_quadrature is None
            or self._n_particles <= 0
            or contacts.soft_contact_max <= 0
        ):
            return

        pos_cur, du_cur = self._sample_particle_world_and_du()
        wp.launch(
            harvest_pipeline_particle_contact_wrenches,
            dim=contacts.soft_contact_max,
            inputs=[
                dt,
                dt * self.sim.friction_reg,
                self.sim.friction_fluid * self.sim.friction_reg,
                self.sim.collision_radius,
                self.sim.friction,
                self._collision_stiffness,
                self._n_particles,
                pos_cur,
                du_cur,
                body_q,
                body_com,
                shape_body,
                body_local_to_proxy_global,
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                out_body_f,
            ],
        )

    def add_collision_energy(self, E: wp.array):
        # if no contacts, return current energy
        if self.n_contact == 0:
            return E

        # get the displacement of the collision quadrature points
        cp_du = self._sample_cp_displacement(self.sim.du_field)
        # store empty space for each of the collision energies
        col_energies = wp.empty(self.n_contact, dtype=float)
        # launch collision energy kernel
        wp.launch(
            collision_energy,
            dim=self.n_contact,
            inputs=[
                self.sim.collision_radius,
                self.sim.friction,
                self.sim.dt * self.sim.friction_reg,
                self.sim.friction_fluid * self.sim.friction_reg,
                cp_du,
                self.collision_kinematic_gaps,
                self.collision_normals,
                self.collision_indices_a,
                self.collision_indices_b,
                col_energies,
            ],
        )
        # sum the collision energies
        Ec = wp.empty_like(E)
        wp.utils.array_sum(col_energies, out=Ec)

        # add collision energy to total energy (multiplied by collision stiffness)
        fem.linalg.array_axpy(x=Ec, y=E, alpha=self._collision_stiffness, beta=1.0)

        # return the total energy
        return E

    def add_collision_hessian(self, lhs: wp.array):
        # if no contacts, return current hessian
        if self.n_contact == 0:
            return lhs

        # get collision jacobian and transpose 
        H = self._collision_jacobian
        Ht = self._collision_jacobian_t

        # launch kernel to multiply BSR matrix by diagonal matrix
        wp.launch(
            bsr_mul_diag,
            dim=(Ht.nnz_sync(), Ht.block_shape[0]),
            inputs=[Ht.scalar_values, Ht.columns, self._col_energy_hessian],
        )

        # adds collision stiffness to the hessian
        sp.bsr_mm(
            x=Ht,
            y=H,
            z=lhs,
            alpha=self._collision_stiffness,
            beta=1.0,
            work_arrays=self._HtH_work_arrays,
        )

        # return the updated hessian
        return lhs

    def add_collision_forces(self, rhs: wp.array):
        """
        Converts per-contact collision gradients into nodal collision forces and adds them to rhs. 

        Args: 
            rhs: the rhs of the linear system

        Returns:
            None. Updates rhs with collision forces
        """
        # if no contacts, return current forces
        if self.n_contact == 0:
            return rhs

        # contacts
        sp.bsr_mv(
            A=self._collision_jacobian_t,
            x=self._col_energy_gradients,
            y=rhs,
            alpha=-self._collision_stiffness,
            beta=1.0,
        )

        return rhs

    def prepare_newton_step(self, dt: float):
        """
        
        """
        # runs collision detection, builds collision jacobian 
        # the goal is to find the currently active contacts and their hessian
        self.detect_collisions(dt)
        self.build_collision_jacobian()

        # compute per-contact forces and hessian
        n_contact = self.n_contact
        if n_contact > 0:
            # store empty space for per-contact collision gradients and hessian
            self._col_energy_gradients = wp.empty(n_contact, dtype=wp.vec3)
            self._col_energy_hessian = wp.empty(n_contact, dtype=wp.mat33)
            # store the displacement of the collision quadrature points
            cp_du = self._sample_cp_displacement(self.sim.du_field)

            # launch kernel to compute per-contact collision gradients and hessian
            wp.launch(
                collision_gradient_and_hessian,
                dim=n_contact,
                inputs=[
                    self.sim.collision_radius,
                    self.sim.friction,
                    dt * self.sim.friction_reg,
                    self.sim.friction_fluid * self.sim.friction_reg,
                    cp_du,
                    self.collision_kinematic_gaps,
                    self.collision_normals,
                    self.collision_indices_a,
                    self.collision_indices_b,
                    self._col_energy_gradients,
                    self._col_energy_hessian,
                ],
            )

    def cp_world_position(self, dest=None):
        """
        Interpolates the world position of each collision quadrature point. There is one collision quadrautre
        point for each sample location where we compute collision forces.

        Args: 
            dest: optional destination array to store the world position of the collision quadrature points

        Returns: 
            The destination array containing the world position of the collision quadrature points
        """
        cp_pic = self.collision_quadrature

        # if no collision quadrature, return an empty array
        if cp_pic is None:
            if dest is None:
                dest = wp.array([], dtype=wp.vec3)
            else:
                dest.assign([])
            return dest

        # if no destination array, create a new one
        if dest is None:
            dest = wp.empty(cp_pic.total_point_count(), dtype=wp.vec3)
        # interpolate the world position of the collision quadrature points (domain + displacement)
        fem.interpolate(
            world_position,
            fields={"u": self.sim.u_field},
            dest=dest,
            at=cp_pic,
        )
        # return the destination array
        return dest

    def _sample_cp_displacement(self, du_field, dest=None):
        """
        Compute the displacement of each collision quadrature points. 

        Args:
            du_field: the displacement field to sample
            dest: optional destination array 
        
        Returns:
            None. Stores displacement of each collision quadrautre point in the destination array
        """
        # collision quadrautre points 
        cp_pic = self.collision_quadrature
        # if no destination array, create a new one 
        if dest is None:
            dest = wp.empty(cp_pic.total_point_count(), dtype=wp.vec3)
        # interpolate the displacement of the collision quadrature points and store in destination array
        fem.interpolate(
            du_field,
            dest=dest,
            at=cp_pic,
        )

        return dest

    def detect_collisions(self, dt):
        """
        Writes contact data that is later used to compute collision forces and hessian.
        These contacts are refreshed when the mesh deforms.

        Args:
            dt: Time step

        Returns: 

        """
        # buffer capacity
        max_contacts = self.collision_normals.shape[0]

        count = wp.zeros(1, dtype=int) # current number of contacts
        indices_a = self.collision_indices_a # collision quadrature point index on soft body
        indices_b = self.collision_indices_b # for ground/kinematic meshes
        normals = self.collision_normals # contact normal
        kinematic_gaps = self.collision_kinematic_gaps # gap between soft body and ground/kinematic meshes

        self.run_collision_detectors( # runs the actual collision detection kernels
            dt,
            count,
            indices_a,
            indices_b,
            normals,
            kinematic_gaps,
        )
        self._ingest_pipeline_contacts(
            dt,
            count,
            indices_a,
            indices_b,
            normals,
            kinematic_gaps,
        )

        self.n_contact = int(count.numpy()[0]) # update the number of contacts

        if self.n_contact > max_contacts:  # if above buffer capacity, some contacts will be ignored
            if not self._warned_contact_overflow:
                print("Warning: contact buffer size exceeded, some have been ignored")
                self._warned_contact_overflow = True
            self.n_contact = max_contacts

    def run_collision_detectors(
        self,
        dt,
        count,
        indices_a,
        indices_b,
        normals,
        kinematic_gaps,
    ):
        """
        Runs the collision detection kernels. 

        Args: 
            dt: Time step
            count: current number of contacts
            indices_a: stores one index for each potential contact point. 
            indices_b: stores one index for each potential contact point
            normals: stores the contact normal for each contact point
            kinematic_gaps: stores the gap between soft body and ground/kinematic meshes

        Returns: 
            None
        """
        # collision quadrature, stores the quadrature points
        cp_pic = self.collision_quadrature
        # number of collision quadrature points
        n_cp = cp_pic.total_point_count()
        # max num contacts
        max_contacts = self.collision_normals.shape[0]

        # current position of collision quadrature points
        cp_cur_pos = self.cp_world_position()
        # store the displacement of the collision quadrature points 
        cp_du = self._sample_cp_displacement(self.sim.du_field)

        # calculate collision radius
        collision_radius = self.sim.collision_radius * self.sim.collision_detection_ratio

        # if we have a ground height, detect ground collisions
        if self.sim.ground:
            # ground height
            ground_height = self.sim.ground_height
            # launch the ground collision detection kernel
            wp.launch(
                detect_ground_collisions,
                dim=n_cp,
                inputs=[
                    max_contacts,
                    self.sim.up_axis_index,
                    cp_cur_pos,
                    cp_du,
                    collision_radius,
                    ground_height,
                    count,
                    normals,
                    kinematic_gaps,
                    indices_a,
                    indices_b,
                ],
            )

        # if we have rigid meshes (kinematic meshes), detect collisions with them
        # this is unused by example_cutting
        if self.warp_meshes:
            mesh_ids = wp.array([mesh.id for mesh in self.warp_meshes], dtype=wp.uint64)
            wp.launch(
                detect_mesh_collisions,
                dim=(len(mesh_ids), n_cp),
                inputs=[
                    max_contacts,
                    dt,
                    mesh_ids,
                    cp_cur_pos,
                    cp_du,
                    collision_radius,
                    count,
                    normals,
                    kinematic_gaps,
                    indices_a,
                    indices_b,
                ],
            )

    def _ingest_pipeline_contacts(
        self,
        dt,
        count,
        indices_a,
        indices_b,
        normals,
        kinematic_gaps,
    ):
        """Append CollisionPipeline particle-shape contacts onto the FEM contact buffers.

        Skips edge/face records (``soft_contact_particle < 0``). Particle ids index
        the original surface-vertex PIC stored in ``_particle_quadrature``.
        """
        contacts = self._pipeline_contacts
        shape_body = self._pipeline_shape_body
        if contacts is None or shape_body is None or self._particle_quadrature is None:
            return
        if self._n_particles <= 0:
            return

        n_contacts = int(contacts.soft_contact_count.numpy()[0])
        if n_contacts <= 0:
            return
        n_contacts = min(n_contacts, contacts.soft_contact_max)

        pos_cur, du_cur = self._sample_particle_world_and_du()

        body_q = self._pipeline_body_q
        if body_q is None:
            body_q = wp.zeros(1, dtype=wp.transform, device=pos_cur.device)

        wp.launch(
            ingest_pipeline_particle_contacts,
            dim=n_contacts,
            inputs=[
                self.collision_normals.shape[0],
                dt,
                n_contacts,
                self._n_particles,
                pos_cur,
                du_cur,
                body_q,
                shape_body,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                count,
                normals,
                kinematic_gaps,
                indices_a,
                indices_b,
            ],
        )

    def build_collision_quadratures(self):
        n_contact = self.n_contact
        a_cells = wp.empty(n_contact, dtype=int)
        a_coords = wp.empty(n_contact, dtype=wp.vec3)
        b_cells = wp.empty(n_contact, dtype=int)
        b_coords = wp.empty(n_contact, dtype=wp.vec3)
        wp.launch(
            gather_cell_coordinates,
            dim=n_contact,
            inputs=[
                self.collision_quadrature.cell_indices,
                self.collision_quadrature.particle_coords,
                self.collision_indices_a,
                a_cells,
                a_coords,
            ],
        )
        wp.launch(
            gather_cell_coordinates,
            dim=n_contact,
            inputs=[
                self.collision_quadrature.cell_indices,
                self.collision_quadrature.particle_coords,
                self.collision_indices_b,
                b_cells,
                b_coords,
            ],
        )

        measures = wp.ones(n_contact, dtype=float)

        a_contact_pic = fem.PicQuadrature(
            self.collision_quadrature.domain,
            positions=(a_cells, a_coords),
            measures=measures,
        )
        b_contact_pic = fem.PicQuadrature(
            self.collision_quadrature.domain,
            positions=(b_cells, b_coords),
            measures=measures,
        )

        return a_contact_pic, b_contact_pic

    def build_collision_jacobian(self):
        n_contact = self.n_contact

        # Build collision jacobian
        # (derivative of collision gap `pos_a - pos_b` w.r.t. degrees of freedom)

        if n_contact == 0:
            return

        a_contact_pic, b_contact_pic = self.build_collision_quadratures()

        u_trial = fem.make_trial(self.sim.u_field.space, space_partition=self.sim.u_field.space_partition)

        sp.bsr_set_zero(
            self._collision_jacobian_a,
            n_contact,
            self.sim.u_field.space_partition.node_count(),
        )
        fem.interpolate(
            u_trial,
            at=a_contact_pic,
            dest=self._collision_jacobian_a,
            kernel_options={"enable_backward": False},
        )

        sp.bsr_set_zero(
            self._collision_jacobian_b,
            n_contact,
            self.sim.u_field.space_partition.node_count(),
        )
        fem.interpolate(
            u_trial,
            at=b_contact_pic,
            dest=self._collision_jacobian_b,
            kernel_options={"enable_backward": False},
        )

        self._collision_jacobian_a.nnz_sync()
        self._collision_jacobian_b.nnz_sync()

        sp.bsr_assign(self._collision_jacobian, src=self._collision_jacobian_a)
        sp.bsr_axpy(
            x=self._collision_jacobian_b,
            y=self._collision_jacobian,
            alpha=-1,
            beta=1,
            work_arrays=self._HbHa_work_arrays,
        )

        sp.bsr_set_transpose(dest=self._collision_jacobian_t, src=self._collision_jacobian)


class MeshSelfCollisionHandler(CollisionHandler):
    """
    A collision handler that detects self collisions in addition to the ground/kinetmatic mesh collisions.
    """
    def __init__(
        self,
        vtx_quadrature: fem.PicQuadrature,
        tri_mesh: wp.Mesh,
    ):
        super().__init__([], vtx_quadrature.cell_indices, vtx_quadrature.particle_coords)
        
        # store quadrature points for the vertices of the mesh 
        self.tri_vtx_quadrature = vtx_quadrature
        # store rest positions of vertices
        self.vtx_rest_pos = wp.clone(tri_mesh.points)
        # store the mesh
        self.tri_mesh = tri_mesh


    def run_collision_detectors(
        self,
        dt,
        count,
        indices_a,
        indices_b,
        normals,
        kinematic_gaps,
    ):
        """
        Runs collision detection kernels and adds any additional self collision quadrature points.

        Args:
            dt: time step
            count: the current number of contacts
            indices_a: stores one index for each potential contact point. 
            indices_b: stores one index for each potential contact point
            normals: stores the contact normal for each contact point
            kinematic_gaps: stores the gap between soft body and ground/kinematic meshes

        Returns:
            None
        """
        # set collision quadrature points  
        self.set_collision_quadrature(self.tri_vtx_quadrature)

        # run collision detection kernels (these are for ground and kinematic mesh collisions)
        super().run_collision_detectors(
            dt,
            count,
            indices_a,
            indices_b,
            normals,
            kinematic_gaps,
        )
        
        # get world positions of quadrature points
        self.cp_world_position(dest=self.tri_mesh.points)
        # refit the mesh so the changed positions are reflected in the mesh data structure 
        self.tri_mesh.refit()

        # store the displacement of the quadrature points
        cp_du = self._sample_cp_displacement(self.sim.du_field)

        # number of collision quadrature points
        n_cp = cp_du.shape[0]
        # max contacts
        max_contacts = self.collision_normals.shape[0]

        # collision radius
        collision_radius = self.sim.collision_radius * self.sim.collision_detection_ratio

        # initial number of contacts
        start_contacts = count.numpy()[0]

        # store the world positions of the quadrature points
        pos_b = wp.empty(indices_b.shape, dtype=wp.vec3)

        # launch self collision detection kernel
        wp.launch(
            detect_mesh_self_collisions,
            dim=(n_cp),
            inputs=[
                start_contacts,
                max_contacts,
                dt,
                self.sim.self_immunity_radius_ratio,
                self.tri_mesh.id,
                self.vtx_rest_pos,
                cp_du,
                collision_radius,
                count,
                normals,
                kinematic_gaps,
                indices_a,
                indices_b,
                pos_b,
            ],
        )
        # calculate the number of self contacts
        self_contacts = int(min(max_contacts, count.numpy()[0]) - start_contacts)

        # if there are self contacts, create new quadrature points for them
        if self_contacts > 0:
            # update quadrature points 
            contact_points = wp.empty(n_cp + self_contacts, dtype=wp.vec3)
            # copy over the rest positions of the vertices and self contact positions
            wp.copy(contact_points[:n_cp], self.vtx_rest_pos)
            wp.copy(contact_points[n_cp:], pos_b[:self_contacts])

            # create new quadrature points including self contacts
            quadrature = fem.PicQuadrature(
                fem.Cells(self.sim.geo),
                contact_points,
                max_dist=self.sim.typical_length,
            )
            # set 
            self.set_collision_quadrature(quadrature)


class CollisionPotential(DisplacementPotential):
    """
    """
    def __init__(self, sim: Deformable, collision_handler: CollisionHandler):
        super().__init__(sim)
        # store the collision handler
        self.collision_handler = collision_handler

    def prepare_frame(self, dt):
        self.collision_handler.detect_collisions(dt)

    def add_energy(self, E_u):
        self.collision_handler.add_collision_energy(E_u)

    def add_hessian(self, lhs: sp.BsrMatrix):
        self.collision_handler.add_collision_hessian(lhs)

    def add_forces(self, rhs: wp.array, tape: wp.Tape = None):
        self.collision_handler.add_collision_forces(rhs)

    def prepare_newton_step(self, dt, tape=None):
        """
        Prepares the collision handler for the next Newton step
        """
        self.collision_handler.prepare_newton_step(dt)

    def init_constant_forms(self):
        self.collision_handler.init_collision_detector(self.sim)


@wp.kernel
def bsr_mul_diag(
    Bt_values: wp.array3d(dtype=float),
    Bt_columns: wp.array(dtype=int),
    C_values: wp.array(dtype=Any),
):
    """
    Kernel to multiply a BSR matrix by a diagonal matrix 

    Args:
        Bt_values: the values of the transp
    """
    # thread index
    i, r = wp.tid()
    # get the column index
    col = Bt_columns[i]

    # get the value of the diagonal matrix at the column index
    C = C_values[col]

    # get value of BSR matrix
    Btr = Bt_values[i, r]
    # multiply BSR matrix by diagonal matrix
    BtC = wp.vec3(Btr[0], Btr[1], Btr[2]) @ C
    # store result
    for k in range(3):
        Btr[k] = BtC[k]


@wp.kernel
def detect_ground_collisions(
    max_contacts: int,
    up_axis: int,
    pos_cur: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    radius: float,
    ground_height: float,
    count: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    i = wp.tid()
    x = pos_cur[i]

    if x[up_axis] < ground_height + radius:
        idx = wp.atomic_add(count, 0, 1)
        if idx >= max_contacts:
            return

        nor = wp.vec3()
        nor[up_axis] = 1.0

        normals[idx] = nor
        kinematic_gaps[idx] = (wp.dot(x - du_cur[i], nor) - ground_height) * nor
        indices_a[idx] = i
        indices_b[idx] = fem.NULL_QP_INDEX


@wp.kernel
def detect_mesh_collisions(
    max_contacts: int,
    dt: float,
    mesh_ids: wp.array(dtype=wp.uint64),
    pos_cur: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    radius: float,
    count: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    m, tid = wp.tid()
    mesh_id = mesh_ids[m]

    x = pos_cur[tid]

    query = wp.mesh_query_point(mesh_id, x, radius)

    if query.result:
        cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)

        delta = x - cp
        dist = wp.length(delta) * query.sign

        if dist < radius:
            idx = wp.atomic_add(count, 0, 1)
            if idx >= max_contacts:
                return

            if dist < 0.00001:
                n = wp.mesh_eval_face_normal(mesh_id, query.face)
            else:
                n = wp.normalize(delta) * query.sign
            normals[idx] = n

            v = wp.mesh_eval_velocity(mesh_id, query.face, query.u, query.v)

            kinematic_gap = (dist - wp.dot(du_cur[tid], n)) * n - v * dt
            kinematic_gaps[idx] = kinematic_gap
            indices_a[idx] = tid
            indices_b[idx] = fem.NULL_QP_INDEX


@wp.kernel
def ingest_pipeline_particle_contacts(
    max_contacts: int,
    dt: float,
    n_contacts: int,
    n_particles: int,
    pos_cur: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    k = wp.tid()
    if k >= n_contacts:
        return

    particle = soft_contact_particle[k]
    if particle < 0 or particle >= n_particles:
        return

    shape = soft_contact_shape[k]
    if shape < 0:
        return

    n = soft_contact_normal[k]
    body = shape_body[shape]
    X_wb = wp.transform_identity()
    if body >= 0:
        X_wb = body_q[body]

    bx = wp.transform_point(X_wb, soft_contact_body_pos[k])
    x = pos_cur[particle]
    dist = wp.dot(n, x - bx)
    bv = wp.transform_vector(X_wb, soft_contact_body_vel[k])

    idx = wp.atomic_add(count, 0, 1)
    if idx >= max_contacts:
        return

    # Same kinematic-gap convention as detect_mesh_collisions: rest-space
    # separation along n, minus rigid motion over this step.
    kinematic_gaps[idx] = (dist - wp.dot(du_cur[particle], n)) * n - bv * dt
    normals[idx] = n
    indices_a[idx] = particle
    indices_b[idx] = fem.NULL_QP_INDEX


@wp.func
def collision_energy_gradient(
    offset: wp.vec3,
    nor: wp.vec3,
    rc: float,
    mu: float,
    dt: float,
    nu: float,
):
    """World-space gradient of one kinematic contact energy wrt the gap offset."""
    d = wp.dot(offset, nor)
    d_hat = d / rc
    stick = wp.where(d_hat < 1.0, 1.0, 0.0)
    dE_d_hat = d_hat - 1.0
    g = dE_d_hat * stick / rc * nor

    vt = (offset - d * nor) / dt
    vt_norm = wp.length(vt)
    mu_fn = -mu * wp.min(0.0, dE_d_hat) / rc
    f1_over_vt_norm = wp.where(vt_norm < 1.0, 2.0 - vt_norm, 1.0 / vt_norm)
    return g + mu_fn * (f1_over_vt_norm + nu) * vt


@wp.kernel
def harvest_pipeline_particle_contact_wrenches(
    dt: float,
    friction_dt: float,
    nu: float,
    radius: float,
    mu: float,
    stiffness: float,
    n_particles: int,
    pos_cur: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    body_local_to_proxy_global: wp.array(dtype=int),
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    out_body_f: wp.array(dtype=wp.spatial_vector),
):
    k = wp.tid()
    if k >= soft_contact_count[0]:
        return

    particle = soft_contact_particle[k]
    if particle < 0 or particle >= n_particles:
        return

    shape = soft_contact_shape[k]
    if shape < 0:
        return

    body = shape_body[shape]
    if body < 0 or body >= body_local_to_proxy_global.shape[0]:
        return

    proxy_global = body_local_to_proxy_global[body]
    if proxy_global < 0 or proxy_global >= out_body_f.shape[0]:
        return

    n = soft_contact_normal[k]
    X_wb = body_q[body]
    bx = wp.transform_point(X_wb, soft_contact_body_pos[k])
    bv = wp.transform_vector(X_wb, soft_contact_body_vel[k])
    du = du_cur[particle]
    dist = wp.dot(n, pos_cur[particle] - bx)
    # Same kinematic gap as ingest / detect_mesh_collisions.
    kinematic_gap = (dist - wp.dot(du, n)) * n - bv * dt
    offset = du + kinematic_gap
    gradient = collision_energy_gradient(offset, n, radius, mu, friction_dt, nu)

    force_on_body = stiffness * gradient
    com_world = wp.transform_point(X_wb, body_com[body])
    torque_on_body = wp.cross(bx - com_world, force_on_body)
    wp.atomic_add(out_body_f, proxy_global, wp.spatial_vector(force_on_body, torque_on_body))


@wp.kernel
def detect_mesh_self_collisions(
    cur_contacts: int,
    max_contacts: int,
    dt: float,
    self_immunity_ratio: float,
    mesh_id: wp.uint64,
    mesh_rest_pos: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    radius: float,
    count: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    pos_b: wp.array(dtype=wp.vec3),
):
    """
    A warp kernel that detects self collisions between the vertices of a mesh. 

    """
    # thread index
    tid = wp.tid()
    # get the mesh
    mesh = wp.mesh_get(mesh_id)

    # store the world position of the vertex
    x = mesh.points[tid]

    # store the lower and upper bounds of the query
    lower = x - wp.vec3(radius)
    upper = x + wp.vec3(radius)

    # query result returns a box
    query = wp.mesh_query_aabb(mesh_id, lower, upper)

    # iterate over the faces in the query result 
    face_index = wp.int32(0)
    while wp.mesh_query_aabb_next(query, face_index):
        t0 = mesh.indices[3 * face_index + 0]
        t1 = mesh.indices[3 * face_index + 1]
        t2 = mesh.indices[3 * face_index + 2]
        if tid == t0 or tid == t1 or tid == t2:
            # ignore self collisions
            continue

        # store the world positions of the vertices of the face
        u1 = mesh.points[t0]
        u2 = mesh.points[t1]
        u3 = mesh.points[t2]

        # return the barycentric coordinates
        d, bary = project_on_tri_at_origin(x - u1, u2 - u1, u3 - u1)
        # if the point is not in the interior, ignore it
        if wp.max(bary) >= 1.0 or wp.min(bary) <= 0.0:
            continue

        # store the interpolated position of the vertex and calculate the distance to the original point
        cp = bary[0] * u1 + bary[1] * u2 + bary[2] * u3
        delta = x - cp

        # store the face normal
        face_nor = wp.mesh_eval_face_normal(mesh_id, face_index)
        # store the sign of the distance to the face normal
        sign = wp.where(wp.dot(delta, face_nor) > 0.0, 1.0, -1.0)

        # distance to original point
        dist = wp.length(delta) * sign

        # if the distance is less than the collision radius, we have a collision
        if dist < radius:
            # discard self-collisions of points that were very close at rest
            rp0 = mesh_rest_pos[t0]
            rp1 = mesh_rest_pos[t1]
            rp2 = mesh_rest_pos[t2]
            xb_rest = bary[0] * rp0 + bary[1] * rp1 + bary[2] * rp2 # rest position of the colliding vertex
            xa_rest = mesh_rest_pos[tid] # rest position of the original vertex
            if wp.length(xb_rest - xa_rest) < self_immunity_ratio * radius: # if too close, ignore
                continue

            # add the contact
            idx = wp.atomic_add(count, 0, 1)
            if idx >= max_contacts: # if we have too many contacts, ignore
                return

            if dist < 0.00001:
                n = face_nor
            else:
                n = wp.normalize(delta) * sign
            normals[idx] = n # store the normal of the contact

            du0 = du_cur[t0] # current displacement
            du1 = du_cur[t1]
            du2 = du_cur[t2]
            # calculate the displacement of the original vertex relative to contact point
            du = du_cur[tid] - du0 * bary[0] - du1 * bary[1] - du2 * bary[2] 

            # calculate the gap between the original vertex and the contact point
            kinematic_gap = (dist - wp.dot(du, n)) * n 
            kinematic_gaps[idx] = kinematic_gap # store the gap
            indices_a[idx] = tid # store index of original vertex
            indices_b[idx] = mesh.points.shape[0] + idx - cur_contacts # store index of contact point
            pos_b[idx - cur_contacts] = xb_rest # store the rest position of the colliding vertex


@wp.func
def collision_offset(
    c: int,
    du_cur: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    idx_a = indices_a[c]
    idx_b = indices_b[c]

    offset = du_cur[idx_a] + kinematic_gaps[c]
    if idx_b != fem.NULL_QP_INDEX:
        offset -= du_cur[idx_b]

    return offset


@wp.func
def collision_target_distance(
    c: int,
    radius: float,
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    return wp.where(indices_b[c] == fem.NULL_ELEMENT_INDEX, 1.0, 2.0) * radius


@wp.kernel
def collision_energy(
    radius: float,
    mu: float,
    dt: float,
    nu: float,
    du_cur: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    energies: wp.array(dtype=float),
):
    c = wp.tid()

    offset = collision_offset(c, du_cur, kinematic_gaps, indices_a, indices_b)
    rc = collision_target_distance(c, radius, indices_a, indices_b)

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    stick = wp.where(d_hat < 1.0, 1.0, 0.0)
    gap = d_hat - 1.0
    E = 0.5 * stick * gap * gap

    vt = (offset - d * nor) / dt  # tangential velocity
    vt_norm = wp.length(vt)

    mu_fn = -mu * wp.min(0.0, gap) / rc  # yield force

    E += (
        mu_fn
        * dt
        * (
            0.5 * nu * vt_norm * vt_norm
            + wp.where(
                vt_norm < 1.0,
                vt_norm * vt_norm * (1.0 - vt_norm / 3.0),
                vt_norm - 1.0 / 3.0,
            )
        )
    )

    energies[c] = E


@wp.kernel
def collision_gradient_and_hessian(
    radius: float,
    mu: float,
    dt: float,
    nu: float,
    du_cur: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    gradient: wp.array(dtype=wp.vec3),
    hessian: wp.array(dtype=wp.mat33),
):
    c = wp.tid()

    offset = collision_offset(c, du_cur, kinematic_gaps, indices_a, indices_b)
    rc = collision_target_distance(c, radius, indices_a, indices_b)

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    stick = wp.where(d_hat < 1.0, 1.0, 0.0)

    dE_d_hat = d_hat - 1.0
    gradient[c] = collision_energy_gradient(offset, nor, rc, mu, dt, nu)
    hessian[c] = wp.outer(nor, nor) * stick / (rc * rc)

    vt = (offset - d * nor) / dt  # tangential velocity
    vt_norm = wp.length(vt)
    vt_dir = wp.normalize(vt)  # avoids dealing with 0

    mu_fn = -mu * wp.min(0.0, dE_d_hat) / rc  # yield force
    f1_over_vt_norm = wp.where(vt_norm < 1.0, 2.0 - vt_norm, 1.0 / vt_norm)

    # regularization such that f / H dt <= k v (penalizes friction switching dir)
    friction_slip_reg = 0.1
    df1_d_vtn = wp.max(
        2.0 * (1.0 - vt_norm),
        friction_slip_reg / (0.5 * friction_slip_reg + vt_norm),
    )

    vt_perp = wp.cross(vt_dir, nor)
    hessian[c] += (
        mu_fn / dt * ((df1_d_vtn + nu) * wp.outer(vt_dir, vt_dir) + (f1_over_vt_norm + nu) * wp.outer(vt_perp, vt_perp))
    )


@wp.kernel
def gather_cell_coordinates(
    qp_cells: wp.array(dtype=int),
    qp_coords: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    cells: wp.array(dtype=int),
    coords: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    qp = indices[i]

    if qp == fem.NULL_QP_INDEX:
        cells[i] = fem.NULL_ELEMENT_INDEX
    else:
        cells[i] = qp_cells[qp]
        coords[i] = qp_coords[qp]


@fem.integrand
def world_position(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    return domain(s) + u(s)
