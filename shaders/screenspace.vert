#version 450 core
layout(location = 0) in vec4 in_particle; // xyz=pos, w=density01

uniform mat4  u_mv;
uniform mat4  u_proj;
uniform float u_particle_radius;
uniform float u_viewport_h;

out float v_density;
out float v_eye_z;

void main() {
    vec4 pos_eye   = u_mv * vec4(in_particle.xyz, 1.0);
    v_eye_z        = pos_eye.z;
    v_density      = in_particle.w;
    float eye_neg  = -pos_eye.z;
    gl_PointSize   = max(1.0, u_particle_radius * u_proj[1][1] * u_viewport_h / eye_neg);
    gl_Position    = u_proj * pos_eye;
}
