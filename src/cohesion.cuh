#ifndef COHESION_CUH_
#define COHESION_CUH_

// cohesion.cuh
// Functions governing attractive forces between contacts

// Check bond pair list, apply linear contact model to pairs
__global__ void bondsLinear(
        uint2*  dev_bonds,
        Float4* dev_bonds_delta, // Contact displacement
        Float4* dev_bonds_omega, // Contact rotational displacement
        Float4* dev_x,
        Float4* dev_vel,
        Float4* dev_angvel,
        Float4* dev_force,
        Float4* dev_torque)
{
    // Find thread index
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < devC_params.nb0) {

        //// Read values

        // Read bond data
        __syncthreads();
        const uint2 bond = dev_bonds[idx]; // Particle indexes in bond pair

        // Check if the bond has been erased
        if (bond.x < devC_np) {

            const Float4 delta0_4 = dev_bonds_delta[idx];
            const Float4 omega0_4 = dev_bonds_omega[idx];

            // Convert tangential vectors to Float3's
            // Uncorrected tangential component of displacement
            Float3 delta0_t = MAKE_FLOAT3(
                    delta0_4.x,
                    delta0_4.y,
                    delta0_4.z);
            const Float delta0_n = delta0_4.w;

            // Uncorrected tangential component of rotation
            Float3 omega0_t = MAKE_FLOAT3(
                    omega0_4.x,
                    omega0_4.y,
                    omega0_4.z);
            const Float omega0_n = omega0_4.w;

            // Read particle data
            const Float4 x_i = dev_x[bond.x];
            const Float4 x_j = dev_x[bond.y];
            const Float4 vel_i = dev_vel[bond.x];
            const Float4 vel_j = dev_vel[bond.y];
            const Float4 angvel4_i = dev_angvel[bond.x];
            const Float4 angvel4_j = dev_angvel[bond.y];

            const Float3 angvel_i = MAKE_FLOAT3(angvel4_i.x, angvel4_i.y, angvel4_i.z);
            const Float3 angvel_j = MAKE_FLOAT3(angvel4_j.x, angvel4_j.y, angvel4_j.z);


            //// Bond geometry and inertia

            // Parallel-bond radius (Potyondy and Cundall 2004, eq. 12)
            const Float R_bar = devC_params.lambda_bar * fmin(x_i.w, x_j.w);

            // Bond cross section area (Potyondy and Cundall 2004, eq. 15)
            const Float A = PI * R_bar*R_bar;

            // Bond moment of inertia (Potyondy and Cundall 2004, eq. 15)
            const Float I = 0.25 * PI * R_bar*R_bar*R_bar*R_bar;

            // Bond polar moment of inertia (Potyondy and Cundall 2004, eq. 15)
            //const Float J = 0.50 * PI * R_bar*R_bar*R_bar*R_bar;
            const Float J = I*2.0;

            // Inter-particle vector
            const Float3 x = MAKE_FLOAT3(
                    x_i.x - x_j.x,
                    x_i.y - x_j.y,
                    x_i.z - x_j.z);
            const Float x_length = length(x);

            // Find overlap (negative value if overlapping)
            const Float overlap = fmin(0.0, x_length - (x_i.w + x_j.w));

            // Normal vector of contact (points from i to j)
            Float3 n = x/x_length;


            //// Force

            // Correct tangential displacement vector for rotation of the contact plane
            //const Float3 delta_t0 = delta_t0_uncor - dot(delta_t0_uncor, n);
            delta0_t = delta0_t - (n * dot(n, delta0_t));

            // Contact displacement, Luding 2008 eq. 10
            const Float3 ddelta = (
                    MAKE_FLOAT3(
                        vel_i.x - vel_j.x,
                        vel_i.y - vel_j.y,
                        vel_i.z - vel_j.z) 
                    + (x_i.w + overlap/2.0) * cross(n, angvel_i)
                    + (x_j.w + overlap/2.0) * cross(n, angvel_j)
                    ) * devC_dt;

            // Normal component of the displacement increment
            //const Float ddelta_n = dot(ddelta, n);
            const Float ddelta_n = -dot(ddelta, n);

            // Normal component of the total displacement
            const Float delta_n = delta0_n + ddelta_n;

            // Tangential component of the displacement increment
            // Luding 2008, eq. 9
            const Float3 ddelta_t = ddelta - n * dot(n, ddelta);

            // Tangential component of the total displacement
            const Float3 delta_t = delta0_t + ddelta_t;

            // Normal force: Visco-elastic contact model
            // The elastic component caused by the overlap is subtracted.
            //f_n = devC_params.k_n * A * delta_n * n;
            const Float3 f_n = (devC_params.k_n * A * delta_n + devC_params.gamma_n * ddelta_n/devC_dt) * n;
            //f_n += devC_params.k_n * overlap * n;


            // Tangential force: Visco-elastic contact model
            //f_t = -devC_params.k_t * A * delta_t;
            const Float3 f_t = -devC_params.k_t * A * delta_t - devC_params.gamma_t * ddelta_t/devC_dt;

            // Force vector
            const Float3 f = f_n + f_t;


            //// Torque

            // Correct tangential rotational vector for rotation of the contact plane
            omega0_t = omega0_t - (-n) * dot(omega0_t, -n);
            //omega0_t = omega0_t - (n * dot(n, omega0_t));

            // Contact rotational velocity
            Float3 domega = MAKE_FLOAT3(
                    angvel_j.x - angvel_i.x,
                    angvel_j.y - angvel_i.y,
                    angvel_j.z - angvel_i.z) * devC_dt;
            /*const Float3 domega = MAKE_FLOAT3(
              angvel_i.x - angvel_j.x,
              angvel_i.y - angvel_j.y,
              angvel_i.z - angvel_j.z) * devC_dt;*/

            // Normal component of the rotational increment
            const Float domega_n = dot(domega, -n);
            //const Float domega_n = dot(-n, domega);
            //const Float domega_n = -dot(n, domega);

            // Normal component of the total rotation
            const Float omega_n = omega0_n + domega_n;

            // Tangential component of the rotational increment
            //const Float3 domega_t = domega - (-n) * dot(domega, -n);
            const Float3 domega_t = domega - domega_n * (-n);
            //const Float3 domega_t = domega - n * dot(n, domega);

            // Tangential component of the total rotation
            const Float3 omega_t = omega0_t + domega_t;

            // Twisting torque: Visco-elastic contact model
            //const Float3 t_n = -devC_params.k_t * J * omega_n * n;
            const Float3 t_n = -devC_params.k_t * J * omega_n * -n;
            //t_n = devC_params.k_t * J * omega_n * n;
            //t_n = (devC_params.k_t * J * omega_n + devC_params.gamma_t * domega_n/devC_dt) * n;

            // Bending torque: Visco-elastic contact model
            //t_t = -devC_params.k_n * I * omega_t;
            //const Float3 t_t = devC_params.k_n * I * omega_t;
            const Float3 t_t = -devC_params.k_n * I * omega_t;
            //t_t = -devC_params.k_n * I * omega_t - devC_params.gamma_n * domega_t/devC_dt;
            //t_t = devC_params.k_n * I * omega_t - devC_params.gamma_n * domega_t/devC_dt;

            // Torque vector
            //t = t_n + t_t;
            //Float3 t_i = t_n + cross(-(x_i.w + overlap*0.5) * n, t_t);
            //Float3 t_j = t_n + cross(-(x_j.w + overlap*0.5) * n, t_t);
            //const Float3 t_i = t_n + cross(-(x_i.w + overlap*0.5) * n, f_t + t_t);
            //const Float3 t_j = t_n + cross(-(x_j.w + overlap*0.5) * n, f_t - t_t);
            const Float3 t_j = t_n + t_t;
            //const Float3 t_i = t_n - t_t; //t_n - t_t;
            //const Float3 t_j = t_n + t_t;


            //// Bond strength (Potyondy & Cundall 2004)
            // Extensions of Euler-Bernoulli beam bending theory
            // Max. stresses in bond periphery

            // Tensile stress
            const Float sigma_max = length(f_n) / A + length(t_t) * R_bar / I;

            // Shear stress
            const Float tau_max = length(f_t) / A + length(t_n) * R_bar / J;

            // Break bond if tensile and shear stresses exceed strengths
            if (sigma_max >= devC_params.sigma_b || tau_max >= devC_params.tau_b) {
                __syncthreads();
                dev_bonds[idx].x = devC_params.nb0;
                return;
            }



            //// Save values
            __syncthreads();

            // Save updated displacements in global memory
            dev_bonds_delta[idx] = MAKE_FLOAT4(delta_t.x, delta_t.y, delta_t.z, delta_n);
            dev_bonds_omega[idx] = MAKE_FLOAT4(omega_t.x, omega_t.y, omega_t.z, omega_n);

            // Save forces and torques to the particle pairs
            // !!! This is probably wrong, see Obermayer et al. 2013, C & GT (49)
            dev_force[bond.x] += MAKE_FLOAT4(f.x, f.y, f.z, 0.0);
            dev_force[bond.y] -= MAKE_FLOAT4(f.x, f.y, f.z, 0.0);
            //dev_torque[bond.x] += MAKE_FLOAT4(t.x, t.y, t.z, 0.0);
            //dev_torque[bond.y] += MAKE_FLOAT4(t.x, t.y, t.z, 0.0);
            //dev_torque[bond.x] += MAKE_FLOAT4(t_i.x, t_i.y, t_i.z, 0.0);
            dev_torque[bond.x] -= MAKE_FLOAT4(t_j.x, t_j.y, t_j.z, 0.0);
            dev_torque[bond.y] += MAKE_FLOAT4(t_j.x, t_j.y, t_j.z, 0.0);
            //dev_torque[bond.y] += MAKE_FLOAT4(t_j.x, t_j.y, t_j.z, 0.0);
            //dev_torque[bond.y] -= MAKE_FLOAT4(t_j.x, t_j.y, t_j.z, 0.0);
            //dev_torque[bond.y] -= MAKE_FLOAT4(t.x, t.y, t.z, 0.0);
            // make sure to remove write conflicts
        }
    }
}

// Linear-elastic bond: Attractive force with normal- and shear components
// acting upon particle A in a bonded particle pair
__device__ void bondLinear_old(Float3* N, Float3* T, Float* es_dot, Float* p,
        unsigned int idx_a, unsigned int idx_b, 
        Float4* dev_x_sorted, Float4* dev_vel_sorted, 
        Float4* dev_angvel_sorted,
        Float radius_a, Float radius_b, 
        Float3 x_ab, Float x_ab_length, 
        Float delta_ab) 
{

    // If particles are not overlapping, apply bond force
    if (delta_ab > 0.0f) {

        // Allocate variables and fetch missing time=t values for particle A and B
        Float4 vel_a     = dev_vel_sorted[idx_a];
        Float4 vel_b     = dev_vel_sorted[idx_b];
        Float4 angvel4_a = dev_angvel_sorted[idx_a];
        Float4 angvel4_b = dev_angvel_sorted[idx_b];

        // Convert to Float3's
        Float3 angvel_a = MAKE_FLOAT3(angvel4_a.x, angvel4_a.y, angvel4_a.z);
        Float3 angvel_b = MAKE_FLOAT3(angvel4_b.x, angvel4_b.y, angvel4_b.z);

        // Normal vector of contact
        Float3 n_ab = x_ab/x_ab_length;

        // Relative contact interface velocity, w/o rolling
        Float3 vel_ab_linear = MAKE_FLOAT3(vel_a.x - vel_b.x, 
                vel_a.y - vel_b.y, 
                vel_a.z - vel_b.z);

        // Relative contact interface velocity of particle surfaces at
        // the contact, with rolling (Hinrichsen and Wolf 2004, eq. 13.10)
        Float3 vel_ab = vel_ab_linear
            + radius_a * cross(n_ab, angvel_a)
            + radius_b * cross(n_ab, angvel_b);

        // Relative contact interface rolling velocity
        //Float3 angvel_ab = angvel_a - angvel_b;
        //Float  angvel_ab_length = length(angvel_ab);

        // Normal component of the relative contact interface velocity
        //Float vel_n_ab = dot(vel_ab_linear, n_ab);

        // Tangential component of the relative contact interface velocity
        // Hinrichsen and Wolf 2004, eq. 13.9
        Float3 vel_t_ab = vel_ab - (n_ab * dot(vel_ab, n_ab));
        //Float  vel_t_ab_length = length(vel_t_ab);

        Float3 f_n = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
        Float3 f_t = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);

        // Mean radius
        Float R_bar = (radius_a + radius_b)/2.0f;

        // Normal force component: Elastic
        f_n = devC_params.k_n * delta_ab * n_ab;

        if (length(vel_t_ab) > 0.f) {
            // Shear force component: Viscous
            f_t = -1.0f * devC_params.gamma_t * vel_t_ab;

            // Shear friction production rate [W]
            //*es_dot += -dot(vel_t_ab, f_t);
        }

        // Add force components from this bond to total force for particle
        *N += f_n + f_t;
        *T += -R_bar * cross(n_ab, f_t);

        // Pressure excerted onto the particle from this bond
        *p += length(f_n) / (4.0f * PI * radius_a*radius_a);

    }
} // End of bondLinear()


// Capillary cohesion after Richefeu et al. (2006)
__device__ void capillaryCohesion_exp(Float3* N, Float radius_a, 
        Float radius_b, Float delta_ab,
        Float3 x_ab, Float x_ab_length, 
        Float kappa)
{

    // Normal vector 
    Float3 n_ab = x_ab/x_ab_length;

    Float3 f_c;
    Float lambda, R_geo, R_har, r, h;

    // Determine the ratio; r = max{Ri/Rj;Rj/Ri}
    if ((radius_a/radius_b) > (radius_b/radius_a))
        r = radius_a/radius_b;
    else
        r = radius_b/radius_a;

    // Exponential decay function
    h = -sqrtf(r);

    // The harmonic mean
    R_har = (2.0f * radius_a * radius_b) / (radius_a + radius_b);

    // The geometrical mean
    R_geo = sqrtf(radius_a * radius_b);

    // The exponential falloff of the capillary force with distance
    lambda = 0.9f * h * sqrtf(devC_params.V_b/R_har);

    // Calculate cohesional force
    f_c = -kappa * R_geo * expf(-delta_ab/lambda) * n_ab;

    // Add force components from this collision to total force for particle
    *N += f_c;

} // End of capillaryCohesion_exp


#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
