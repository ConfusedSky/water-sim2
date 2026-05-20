#version 450 core
in float v_density;
in float v_eye_z;

uniform float u_particle_radius;
uniform mat4  u_proj;
uniform float u_depth_near;
uniform float u_depth_far;

out vec4 frag;

void main() {
    vec2  uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    float sphere_z = sqrt(1.0 - r2);

    // Sphere-imposter depth correction: push eye_z to the sphere surface.
    float eye_z = v_eye_z + u_particle_radius * sphere_z;

    // Linear depth in [0, 1] over [u_depth_near, u_depth_far]. Guarded against
    // far <= near so the controls can be edited freely without producing NaNs.
    float span      = max(u_depth_far - u_depth_near, 1e-6);
    float linear_z  = clamp((-eye_z - u_depth_near) / span, 0.0, 1.0);

    gl_FragDepth = linear_z;
    frag         = vec4(vec3(linear_z), 1.0);
}
