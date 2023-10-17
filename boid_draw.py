from Draw import Draw

# Draw this Boid's “body” -- currently an irregular tetrahedron.
def draw_boid(position, forward, side, up, body_radius, color):
    center = position
    nose = center + forward * body_radius
    tail = center - forward * body_radius
    bd = body_radius * 2  # body diameter (defaults to 1)
    apex = tail + up * 0.25 * bd + forward * 0.1 * bd
    wingtip0 = tail + side * 0.3 * bd
    wingtip1 = tail - side * 0.3 * bd
    # Draw the 4 triangles of a boid's body.
    def draw_tri(a, b, c, color):
        Draw.add_colored_triangle(a, b, c, color)
    draw_tri(nose, apex,     wingtip1, color * 1.00)
    draw_tri(nose, wingtip0, apex,     color * 0.95)
    draw_tri(apex, wingtip0, wingtip1, color * 0.90)
    draw_tri(nose, wingtip1, wingtip0, color * 0.70)
