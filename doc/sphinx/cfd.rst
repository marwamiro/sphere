Fluid simulation by CFD
=======================
Fluid flow is governed by the Navier-Stokes continuity and momentum equations,
assuming that the fluid is incompressible. In a single phase fluid without
grains, the continuity equation is:

.. math::
    \nabla \cdot \boldsymbol{v}_f = 0

and the momentum equation:

.. math::
    \frac{\partial \boldsymbol{v}_f}{\partial t}
    + \boldsymbol{v}_f \cdot \nabla \boldsymbol{v}_f =
    - \frac{1}{\rho_f} \nabla p_f + \nu \nabla^2 \boldsymbol{v}_f
    + \boldsymbol{f}_g

Here, :math:`\boldsymbol{v}_f` is the fluid velocity,
and :math:`p_f` is the fluid pressure. :math:`\nu` is the fluid
viscosity, and :math:`\boldsymbol{f}_g` is the gravitational force.  The
`Laplacian`_ (:math:`\nabla^2`) is the `divergence`_ (:math:`\nabla \cdot`) of
the `gradient`_ (:math:`\nabla`):

.. math::
    \nabla^2 = \nabla \cdot \nabla = 
    \frac{\partial^2}{\partial x^2} +
    \frac{\partial^2}{\partial y^2} +
    \frac{\partial^2}{\partial z^2}

In an averaged discretization the continuity equation becomes:

.. math::
    \nabla \cdot \bar{\boldsymbol{v}}_f = 0

The bar symbol denotes that the value is averaged in the cell. The momentum
equation, assuming that that the fluid is inviscid (zero
viscosity), becomes:

.. math::
    \frac{\partial \bar{\boldsymbol{v}}_f}{\partial t}
    + \bar{\boldsymbol{v}}_f \cdot \nabla \bar{\boldsymbol{v}}_f =
    - \frac{1}{\rho_f} \nabla \bar{p}_f
    + \boldsymbol{f}_g

When the fluid flow happens in a porous medium, the equations are modified to
take account for the porosity in the cell (:math:`\phi`) and the averaged drag
from the particles (:math:`\boldsymbol{\bar{f}}_i`) (Shamy and Zeghal, 2005, and
model A in Zhu et al. 2007). The continuity equation becomes:

.. math::
    \frac{\partial \phi}{\partial t}
    + \nabla \cdot (\phi \bar{\boldsymbol{v}}_f) = 0

and the momentum equation becomes:

.. math::
    \frac{\partial (\phi \bar{\boldsymbol{v}}_f)}{\partial t}
    + \nabla \cdot (\phi \bar{\boldsymbol{v}}_f \otimes \bar{\boldsymbol{v}}_f) =
    - \frac{1}{\rho_f} \phi \nabla \bar{p}_f
    - \frac{1}{\rho_f} \boldsymbol{\bar{f}}_i
    + \phi \boldsymbol{f}_g

The outer product :math:`\bar{\boldsymbol{v}}_f \otimes \bar{\boldsymbol{v}}_f`
is equivalent to a matrix multiplication :math:`\bar{\boldsymbol{v}}_f
\bar{\boldsymbol{v}}_f^T`, and results in a 3-by-3 matrix. The divergence of a
matrix yields a vector field.
The solution of the above equations is performed by operator splitting methods.
The methodology presented by Langtangen et al. (2002) is for a viscous fluid
without particles. A velocity prediction after a forward step in time
(:math:`\Delta t`) in the momentum equation is found using an explicit scheme:

.. math::
    \bar{\boldsymbol{v}}^*_f = \bar{\boldsymbol{v}}^t_f
    - \Delta t \bar{\boldsymbol{v}}^t_f \cdot \nabla \bar{\boldsymbol{v}}^t_f
    - \Delta t \frac{\beta}{\rho_f} \nabla \bar{p}_f^t
    + \Delta t \nu \nabla^2 \bar{\boldsymbol{v}}_f^t
    + \Delta t \boldsymbol{f}_g^t

This predicted velocity does not account for the incompressibility condition.
The parameter :math:`\beta` is an adjustable, dimensionless relaxation
parameter. The above velocity prediction is modified to account for the presence
of particles and the fluid inviscidity:

.. math::
    \bar{\boldsymbol{v}}^*_f = \bar{\boldsymbol{v}}^t_f 
    - \Delta t \nabla \cdot (\phi^t \bar{\boldsymbol{v}}_f^t \otimes \bar{\boldsymbol{v}}_f^t)
    - \Delta t \frac{\beta}{\rho_f} \phi^t \nabla \bar{p}_f^t
    - \frac{\Delta t}{\rho_f} \boldsymbol{\bar{f}}_i^t
    + \Delta t \phi^t \boldsymbol{f}_g^t

The new velocities should fulfill the continuity (here incompressibility)
equation:

.. math::
    \frac{\Delta \phi^t}{\Delta t} + \nabla \cdot (\phi^t
    \bar{\boldsymbol{v}}_f^{t+\Delta t}) = 0

The divergence of a scalar and vector can be `split`_:

.. math::
    \phi^t \nabla \cdot \bar{\boldsymbol{v}}_f^{t+\Delta t} +
    \bar{\boldsymbol{v}}_f^{t+\Delta t} \cdot \nabla \phi^t
    + \frac{\Delta \phi^t}{\Delta t} = 0

The predicted velocity is corrected using the new pressure (Langtangen et al.
2002):

.. math::
    \bar{\boldsymbol{v}}_f^{t+\Delta t} = \bar{\boldsymbol{v}}_f^*
    - \frac{\Delta t}{\rho} \nabla \epsilon
    \quad \text{where} \quad
    \epsilon = p_f^{t+\Delta t} - \beta p_f^t

The above formulation of the future velocity is put into the continuity
equation:

.. math::
    \Rightarrow
    \phi^t \nabla \cdot
    \left( \bar{\boldsymbol{v}}^*_f - \frac{\Delta t}{\rho_f} \nabla \epsilon \right)
    +
    \left( \bar{\boldsymbol{v}}^*_f - \frac{\Delta t}{\rho_f} \nabla \epsilon \right)
    \cdot \nabla \phi^t + \frac{\Delta \phi^t}{\Delta t} = 0

.. math::
    \Rightarrow
    \phi^t \nabla \cdot
    \bar{\boldsymbol{v}}^*_f - \frac{\Delta t}{\rho_f} \phi^t \nabla^2 \epsilon
    + \nabla \phi^t \cdot \bar{\boldsymbol{v}}^*_f
    - \nabla \phi^t \cdot \nabla \epsilon \frac{\Delta t}{\rho}
    + \frac{\Delta \phi^t}{\Delta t} = 0

.. math::
    \Rightarrow
    \frac{\Delta t}{\rho} \phi^t \nabla^2 \epsilon
    = \phi^t \nabla \cdot \bar{\boldsymbol{v}}^*_f
    + \nabla \phi^t \cdot \bar{\boldsymbol{v}}^*_f
    - \nabla \phi^t \cdot \nabla \epsilon \frac{\Delta t}{\rho}
    + \frac{\Delta \phi^t}{\Delta t}

The pressure difference in time becomes a `Poisson equation`_ with added terms:

.. math::
    \Rightarrow
    \nabla^2 \epsilon
    = \frac{\nabla \cdot \bar{\boldsymbol{v}}^*_f \rho}{\Delta t}
    + \frac{\nabla \phi^t \cdot \bar{\boldsymbol{v}}^*_f \rho}{\Delta t \phi^t}
    - \frac{\nabla \phi^t \cdot \nabla \epsilon}{\phi^t}
    + \frac{\Delta \phi^t \rho}{\Delta t^2 \phi^t}

The right hand side of the above equation is termed the *forcing function*
:math:`f`.  See the `Jacobi iterative solution procedure of a Poisson
equation`_.  The value of :math:`\epsilon` is found `iteratively`_ by using the
discrete Laplacian previously mentioned. The value of :math:`\epsilon(x,y,z)` is
the solution sought, and the right hand side of the above equation is the
forcing function.  Using second-order finite difference approximations of the
Laplace operator second-order partial derivatives, the differential equations
become a system of equations that is solved using Jacobi iterations. The total
number of unknowns is :math:`(n_x - 1)(n_y - 1)(n_z - 1)`.

The discrete Laplacian (approximation of the Laplace operator) can be obtained
by a finite-difference seven-point stencil in a three-dimensional, cubic
grid with cell spacing :math:`\Delta x, \Delta y, \Delta z`, considering the 6 face neighbors:

.. math::
    \nabla^2 \epsilon_{i_x,i_y,i_z}  \approx 
    \frac{\epsilon_{i_x-1,i_y,i_z} - 2 \epsilon_{i_x,i_y,i_z}
    + \epsilon_{i_x+1,i_y,i_z}}{\Delta x^2}
    + \frac{\epsilon_{i_x,i_y-1,i_z} - 2 \epsilon_{i_x,i_y,i_z}
    + \epsilon_{i_x,i_y+1,i_z}}{\Delta y^2}

    + \frac{\epsilon_{i_x,i_y,i_z-1} - 2 \epsilon_{i_x,i_y,i_z}
    + \epsilon_{i_x,i_y,i_z+1}}{\Delta z^2}
    \approx f_{i_x,i_y,i_z}

Within a Jacobi iteration, the value of the unknowns (:math:`\epsilon^n`) is
used to find an updated solution estimate (:math:`\epsilon^{n+1}`).
The solution for the updated value takes the form:

.. math::
    \epsilon^{n+1}_{i_x,i_y,i_z}
    = \frac{-\Delta x^2 \Delta y^2 \Delta z^2 f_{i_x,i_y,i_z}
    + \Delta y^2 \Delta z^2 (\epsilon^n_{i_x-1,i_y,i_z} +
      \epsilon^n_{i_x+1,i_y,i_z})
    + \Delta x^2 \Delta z^2 (\epsilon^n_{i_x,i_y-1,i_z} +
      \epsilon^n_{i_x,i_y+1,i_z})
    + \Delta x^2 \Delta y^2 (\epsilon^n_{i_x,i_y,i_z-1} +
      \epsilon^n_{i_x,i_y,i_z+1})}
      {2 (\Delta x^2 \Delta y^2
      + \Delta x^2 \Delta z^2
      + \Delta y^2 \Delta z^2) }

The difference between the current and updated value is termed the *normalized residual*:

.. math::
    r_{i_x,i_y,i_z} = \frac{(\epsilon^{n+1}_{i_x,i_y,i_z} - \epsilon^n_{i_x,i_y,i_z})^2}{(\epsilon^{n+1}_{i_x,i_y,i_z})^2}

Note that the :math:`\epsilon` values cannot be 0 due to the above normalization
of the residual.

The updated values are at the end of the iteration stored as the current values,
and the maximal value of the normalized residual is found. If this value is
larger than a tolerance criteria, the procedure is repeated. The iterative
procedure is ended if the number of iterations exceeds a defined limit. 

After the values of :math:`\epsilon` are found, they are used to find the new
pressures and velocities:

.. math::
    \bar{p}_f^{t+\Delta t} = \beta \bar{p}^t + \epsilon

.. math::
    \bar{\boldsymbol{v}}_f^{t+\Delta t} =
    \bar{\boldsymbol{v}}^*_f - \frac{\Delta t}{\rho} \nabla \epsilon




.. _Laplacian: https://en.wikipedia.org/wiki/Laplace_operator 
.. _divergence: https://en.wikipedia.org/wiki/Divergence
.. _gradient: https://en.wikipedia.org/wiki/Gradient
.. _split: http://www.wolframalpha.com/input/?i=div(p+v)
.. _Poisson equation: https://en.wikipedia.org/wiki/Poisson's_equation
.. _`Jacobi iterative solution procedure of a Poisson equation`: http://www.rsmas.miami.edu/personal/miskandarani/Courses/MSC321/Projects/prjpoisson.pdf
.. _iteratively: https://en.wikipedia.org/wiki/Relaxation_(iterative_method)

