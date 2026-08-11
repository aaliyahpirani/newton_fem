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

import weakref
from typing import Any

import warp as wp
import warp.sparse as sp


class LineSearch:
    """
    Base class for line search acceptance criteria. 
    """
    def __init__(self, sim):
        self.sim = weakref.proxy(sim)

    def build_linear_model(self, sim, lhs, rhs, delta_fields):
        pass

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        pass


class LineSearchNaiveCriterion(LineSearch):
    """
    Naive criterion for line search. 
    """
    def __init__(self, sim):
        super().__init__(sim)
        self.penalty = sim.young_modulus  # default penalty for the constraint 

    def build_linear_model(self, sim, lhs, rhs, delta_fields):
        pass

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        """
        Acceptance criterion for the naive line search. A trial step is accepted if a single
        penalized cost does not increase. 

        Args:
            alpha: the step size 
            E_cur: the current elastic energy
            C_cur: the current constraint violation
            E_ref: the elastic energy before this newton attempt
            C_ref: the constraint before this newton attempt

        """
        f_cur = E_cur + self.penalty * C_cur # build the current cost 
        f_ref = E_ref + self.penalty * C_ref # build the previous cost
        return f_cur <= f_ref # accept the step if the current cost is lower 


class LineSearchUnconstrainedArmijoCriterion(LineSearch):
    """
    Unconstrained Armijo criterion for line search. 
    """
    def __init__(self, sim):
        super().__init__(sim)
        self.armijo_coeff = 0.0001

    def build_linear_model(self, lhs, rhs, delta_fields):
        """
        Computes the first order prediction of how energy changes along the newton direction. 
        """
        # unpack the search direction in displacement space 
        (delta_u,) = delta_fields

        # compute the direction derivative of the energy along the delta 
        m = -wp.utils.array_inner(delta_u, self.sim._minus_dE_du.view(delta_u.dtype))
        # store the linear prediction 
        self.m = m

    def accept(self, alpha, E_cur, C_cur, E_ref, C_ref):
        """
        Acceptance criterion for the unconstrained Armijo line search. 
        Accept if actual energy is at most the linear prediction plus a small constant. 

        Args:
            alpha: step size
            E_cur: energy after trial step
            C_cur: constraint violation after trial step
            E_ref: energy before trial step
            C_ref: constraint violation before trial step
        """
        # return if smaller than the linear prediction plus a small constant 
        return E_cur <= E_ref + self.armijo_coeff * alpha * self.m

