# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFEMNewton pipeline particle-shape contact ingest."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_fem_newton_pipeline_particle_contacts(test, device):
    """Step SolverFEMNewton with CollisionPipeline particle-shape contacts without NaNs."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_soft_grid(
        pos=wp.vec3(-0.1, -0.1, 0.12),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=1.0,
        k_mu=1.0e2,
        k_lambda=1.0e2,
        k_damp=0.0,
        particle_radius=0.05,
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
        hx=0.5,
        hy=0.5,
        hz=0.05,
    )
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverFEMNewton(
        model=model,
        resolution=8,
        up_axis=2,
        gravity=10.0,
        young_modulus=10.0,
        poisson_ratio=0.1,
        density=1.0,
        dt=1.0 / 30.0,
        n_newton=1,
        cg_iters=20,
        y_min=-2.0,
        y_max=2.0,
        quiet=True,
        ground=False,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = pipeline.contacts()

    pipeline.collide(state_0, contacts)
    n_soft = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(n_soft, 0)

    solver.step(state_0, state_1, control, contacts, 1.0 / 30.0)
    q = state_1.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertGreater(solver.collision_handler.n_contact, 0)


class TestSolverFEMNewton(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
add_function_test(
    TestSolverFEMNewton,
    "test_fem_newton_pipeline_particle_contacts",
    test_fem_newton_pipeline_particle_contacts,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
