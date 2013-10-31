Fluid dynamics by CFD
=====================
Fluid flow is governed by the Navier-Stokes continuity and momentum equations,
assuming that the fluid is incompressible. In a single phase fluid without
grains, the continuity equation is:

.. math::
    \nabla \cdot \boldsymbol{v}_f = 0

and the momentum equation:

.. math::
    \frac{\partial \boldsymbol{v}_f}{\partial t}
    + \bar{\boldsymbol{v}}_f \cdot \nabla \bar{\boldsymbol{v}}_f =
    - \frac{1}{\rho_f} \nabla p_f + \nu \nabla^2 \bar{\boldsymbol{v}}_f
    + \boldsymbol{f}_g

Here, :math:`\bar{\boldsymbol{v}}_f` is the averaged fluid velocity,
and :math:`\bar{p}_f` is the averaged fluid pressure. :math:`\nu` is the fluid
viscosity, and :math:`\boldsymbol{f}_g` is the gravitational force.  The
`Laplacian`_ (:math:`\nabla^2`) is the `divergence`_ (:math:`\nabla \cdot`) of
the `gradient`_ (:math:`\nabla`):

.. math::
    \nabla^2 = \nabla \cdot \nabla = 
    \frac{\partial^2}{\partial x^2} +
    \frac{\partial^2}{\partial y^2} +
    \frac{\partial^2}{\partial z^2}

The discrete Laplacian (approximation of the Laplace operator) can be obtained
by a finite-difference seven-point stencil in a three-dimensional regular, cubic
grid with cell spacing :math:`h`, considering the 6 face neighbors (O'Reilly and
Beck 2006):

.. math::
    % OReilly and Beck 2006 A Family of Large-Stencil Discrete Laplacian
    % Approximations in Three Dimensions
    \nabla^2 f(x,y,z) \approx \frac{1}{h^2} \left(
    f(x-h,y,z) + f(x+h,y,z) + f(x,y-h,z) +
    f(x,y+h,z) + f(x,y,z-h) + f(x,y,z+h) - 6f(x,y) \right)

In an averaged discretization, assuming that that the fluid is inviscid (zero
viscosity), the continuity equation becomes:

.. math::
    \nabla \cdot \bar{\boldsymbol{v}}_f = 0

The bar symbol denotes that the value is averaged in the cell. The momentum
equation becomes:

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
    + \nabla \cdot (\phi \bar{\boldsymbol{v}}_f \bar{\boldsymbol{v}}_f) =
    - \frac{1}{\rho_f} \phi \nabla \bar{p}_f
    - \frac{1}{\rho_f} \boldsymbol{\bar{f}}_i
    + \phi \boldsymbol{f}_g

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
    - \Delta t \nabla \cdot (\phi^t \bar{\boldsymbol{v}}_f^t \bar{\boldsymbol{v}}_f^t)
    - \Delta t \frac{\beta}{\rho_f} \phi \nabla \bar{p}_f^t
    - \frac{\Delta t}{\rho_f} \boldsymbol{\bar{f}}_i^t
    + \Delta t \phi^t \boldsymbol{f}_g^t

The new velocities should fulfill the continuity (here incompressibility)
equation:

.. math::
    \frac{\Delta \phi}{\Delta t} + \nabla \cdot (\phi
    \bar{\boldsymbol{v}}_f^{t+\Delta t}) = 0

The divergence of a scalar and vector can be `split`_:

.. math::
    \phi \nabla \cdot \bar{\boldsymbol{v}}_f^{t+\Delta t} +
    \bar{\boldsymbol{v}}_f^{t+\Delta t} \cdot \nabla \phi
    + \frac{\Delta \phi}{\Delta t} = 0

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
    \phi \nabla \cdot
    \left( \bar{\boldsymbol{v}}^*_f - \frac{\Delta t}{\rho_f} \nabla \epsilon \right)
    +
    \left( \bar{\boldsymbol{v}}^*_f - \frac{\Delta t}{\rho_f} \nabla \epsilon \right)
    \cdot \nabla \phi + \frac{\Delta \phi}{\Delta t} = 0

.. math::
    \Rightarrow
    \phi \nabla \cdot
    \bar{\boldsymbol{v}}^*_f - \frac{\Delta t}{\rho_f} \phi \nabla^2 \epsilon
    + \nabla \phi \cdot \bar{\boldsymbol{v}}^*_f
    - \nabla \phi \cdot \nabla \epsilon \frac{\Delta t}{\rho}
    + \frac{\Delta \phi}{\Delta t} = 0

.. math::
    \Rightarrow
    \frac{\Delta t}{\rho} \phi \nabla^2 \epsilon
    = \phi \nabla \cdot \bar{\boldsymbol{v}}^*_f
    + \nabla \phi \cdot \bar{\boldsymbol{v}}^*_f
    - \nabla \phi \cdot \nabla \epsilon \frac{\Delta t}{\rho}
    + \frac{\Delta \phi}{\Delta t}

The pressure difference in time becomes a `Poisson equation`_ with added terms:

.. math::
    \Rightarrow
    \nabla^2 \epsilon
    = \frac{\nabla \cdot \bar{\boldsymbol{v}}^*_f \rho}{\Delta t}
    + \frac{\nabla \phi \cdot \bar{\boldsymbol{v}}^*_f \rho}{\Delta t \phi}
    - \frac{\nabla \phi \cdot \nabla \epsilon}{\phi}
    + \frac{\Delta \phi \rho}{\Delta t^2 \phi}

The value of :math:`\epsilon` is found `iteratively`_ by using the discrete
Laplacian previously mentioned. When the solution is found, the value is used to
find the new pressure and velocity:

.. math::
    \bar{p}_f^{t+\Delta t} = \beta \bar{p}^t + \epsilon

.. math::
    \bar{\boldsymbol{v}}_f^{t+\Delta t} =
    \bar{\boldsymbol{v}}^*_f - \frac{\Delta t}{\rho} \nabla \phi




.. _Laplacian: https://en.wikipedia.org/wiki/Laplace_operator 
.. _divergence: https://en.wikipedia.org/wiki/Divergence
.. _gradient: https://en.wikipedia.org/wiki/Gradient
.. _split: http://www.wolframalpha.com/input/?i=div(p+v)
.. _Poisson equation: https://en.wikipedia.org/wiki/Poisson's_equation
.. _iteratively: https://en.wikipedia.org/wiki/Relaxation_(iterative_method)

