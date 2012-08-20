#ifndef COHESION_CUH_
#define COHESION_CUH_

// cohesion.cuh
// Functions governing attractive forces between contacts


// Linear-elastic bond: Attractive force with normal- and shear components
// acting upon particle A in a bonded particle pair
__device__ void bondLinear(Float3* N, Float3* T, Float* es_dot, Float* p,
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
    f_n = devC_k_n * delta_ab * n_ab;

    if (length(vel_t_ab) > 0.f) {
      // Shear force component: Viscous
      f_t = -1.0f * devC_gamma_s * vel_t_ab;

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
  lambda = 0.9f * h * sqrtf(devC_V_b/R_har);

  // Calculate cohesional force
  f_c = -kappa * R_geo * expf(-delta_ab/lambda) * n_ab;

  // Add force components from this collision to total force for particle
  *N += f_c;

} // End of capillaryCohesion_exp


#endif
