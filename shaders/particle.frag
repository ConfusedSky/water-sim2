#version 450 core
in float v_density;
in float v_eye_z;

uniform float u_sphere_radius;
uniform float u_proj22;   // proj[2][2]
uniform float u_proj32;   // proj[3][2]

out vec4 frag;

void main() {
    vec2  uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    float sphere_z = sqrt(1.0 - r2);
    vec3  n        = vec3(uv.x, -uv.y, sphere_z);   // view-space sphere normal

    // Correct fragment depth to sphere surface
    float eye_z  = v_eye_z + u_sphere_radius * sphere_z;
    float ndc_z  = (u_proj22 * eye_z + u_proj32) / (-eye_z);
    gl_FragDepth = ndc_z * 0.5 + 0.5;

    // Phong
    vec3  light = normalize(vec3(0.45, 0.75, 0.55));
    float diff  = max(dot(n, light), 0.0);
    vec3  H     = normalize(light + vec3(0.0, 0.0, 1.0));
    float spec  = pow(max(dot(n, H), 0.0), 64.0);
    float fres  = pow(1.0 - sphere_z, 3.0);

    float d    = clamp((v_density - 0.8) / 0.4, 0.0, 1.0);
    vec3  base = mix(vec3(0.06, 0.28, 0.65), vec3(0.50, 0.82, 1.00), d);
    vec3  col  = base * (0.15 + 0.85 * diff);
    col = mix(col, vec3(0.85, 0.95, 1.00), fres * 0.4);
    col += vec3(spec * 0.55);
    frag = vec4(col, 1.0);
}
