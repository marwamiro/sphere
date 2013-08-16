Discrete element method
=======================
The discrete element method (or distinct element method) was initially
formulated by Cundall and Strack (1979). It simulates the physical behavior and
interaction of discrete, unbreakable particles, with their own mass and inertia,
under the influence of e.g. gravity and boundary conditions such as moving
walls. By discretizing time into small time steps, explicit integration of
Newton's second law of motion is used to predict the new position and kinematic
values for each particle from the previous sums of forces. This Lagrangian
approach is ideal for simulating discontinuous materials, such as granular
matter.
The complexity of the computations is kept low by representing the particles as
spheres, which keeps contact-searching algorithms simple.



