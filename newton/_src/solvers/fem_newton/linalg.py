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

from typing import Any, Tuple

import warp as wp
import warp.sparse as sp
from warp.fem.utils import array_axpy

wp.set_module_options({"enable_backward": False})
wp.set_module_options({"fast_math": True})


def diff_bsr_mv(
    A: sp.BsrMatrix,
    x: wp.array,
    y: wp.array,
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose: bool = False,
    self_adjoint: bool = False,
):
    """Performs y = alpha*A*x + beta*y and records the adjoint on the tape"""

    from warp.context import runtime

    tape = runtime.tape
    if tape is not None and (x.requires_grad or y.requires_grad):

        def backward():
            # adj_x += adj_y * alpha
            # adj_y = adj_y * beta

            sp.bsr_mv(
                A=A,
                x=y.grad,
                y=x.grad,
                alpha=alpha,
                beta=1.0,
                transpose=(not transpose) and (not self_adjoint),
            )
            if beta != 1.0:
                array_axpy(x=y.grad, y=y.grad, alpha=0.0, beta=beta)

        runtime.tape.record_func(backward, arrays=[x, y])

    runtime.tape = None
    # in case array_axpy eventually records its own stuff
    sp.bsr_mv(A, x, y, alpha, beta, transpose)
    runtime.tape = tape