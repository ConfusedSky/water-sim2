#version 450 core
in float v_density;
in float v_eye_z;
in float v_r_norm;

uniform float u_particle_radius;
uniform mat4  u_proj;
uniform vec3  u_base_color_lo;
uniform vec3  u_base_color_hi;

out vec4 frag;

void main() {
    // Re-normalize the sprite's [-1, 1] coords into the *actual* sphere's
    // [-1, 1] so the disc boundary tracks the continuous projected radius
    // (v_r_norm), not the rasterized sprite footprint.
    vec2  uv   = (gl_PointCoord * 2.0 - 1.0) / v_r_norm;
    float r2   = dot(uv, uv);
    if (r2 > 1.0) discard;

    float sphere_z = sqrt(1.0 - r2);
    vec3  n        = vec3(uv.x, -uv.y, sphere_z);   // view-space sphere normal

    // Correct fragment depth to sphere surface
    float eye_z  = v_eye_z + u_particle_radius * sphere_z;
    float ndc_z  = (u_proj[2][2] * eye_z + u_proj[3][2]) / (-eye_z);
    gl_FragDepth = ndc_z * 0.5 + 0.5;

    // Phong
    vec3  light = normalize(vec3(0.45, 0.75, 0.55));
    float diff  = max(dot(n, light), 0.0);
    vec3  H     = normalize(light + vec3(0.0, 0.0, 1.0));
    float spec  = pow(max(dot(n, H), 0.0), 64.0);
    float fres  = pow(1.0 - sphere_z, 3.0);

    float d    = clamp((v_density - 0.8) / 0.4, 0.0, 1.0);
    vec3  base = mix(u_base_color_lo, u_base_color_hi, d);
    vec3  col  = base * (0.15 + 0.85 * diff);
    col = mix(col, vec3(0.85, 0.95, 1.00), fres * 0.4);
    col += vec3(spec * 0.55);
    frag = vec4(col, 1.0);
}
