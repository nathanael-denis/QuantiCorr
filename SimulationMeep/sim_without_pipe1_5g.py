import meep as mp
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as patches

# === Output directory ===
output_dir = "SimFiles/OpenAir_3Oxide_Backscatter"
os.makedirs(output_dir, exist_ok=True)

# === Simulation parameters ===
resolution = 50
sx, sy, dpml = 10, 5, 1.0
frequency = 1.0

# === Material ===
oxide = mp.Medium(epsilon=5)

# === Oxide positions (three sources) ===
num_oxides = 3
oxide_radius = 0.1
vertical_spacing = 0.6
total_height = (num_oxides - 1) * vertical_spacing
y_start = -total_height / 2

oxide_positions = [(0, y_start + i * vertical_spacing) for i in range(num_oxides)]

# === Geometry: oxide cylinders (no pipe) ===
geometry = []
for pos in oxide_positions:
    geometry.append(mp.Cylinder(radius=oxide_radius,
                                center=mp.Vector3(pos[0], pos[1]),
                                height=mp.inf,
                                material=oxide))

# === Oxides as Gaussian sources ===
sources = []
for pos in oxide_positions:
    sources.append(
        mp.Source(
            src=mp.GaussianSource(frequency=frequency, fwidth=0.5),
            component=mp.Ez,
            center=mp.Vector3(pos[0], pos[1]),
            size=mp.Vector3(0, 2*oxide_radius)
        )
    )

# === Simulation setup ===
sim = mp.Simulation(
    cell_size=mp.Vector3(sx, sy),
    boundary_layers=[mp.PML(dpml)],
    geometry=geometry,
    sources=sources,
    resolution=resolution
)

# --- Antenna / receiver monitor (to measure backscatter) ---
recv_center = mp.Vector3(-sx/2 + 1, 0)
recv_monitor = sim.add_flux(frequency, 0.5, 50, mp.FluxRegion(center=recv_center, size=mp.Vector3(0, sy)))

# === Run simulation ===
sim.run(until=50)

# === Save Ez field ===
ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sx, sy), component=mp.Ez)
np.save(os.path.join(output_dir, "Ez_field.npy"), ez_data)

# === Plot Ez field and oxide positions ===
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(ez_data.T, origin='lower', interpolation='spline36', cmap='RdBu',
               extent=[-sx/2, sx/2, -sy/2, sy/2])
plt.colorbar(im, ax=ax, label='Ez field')

# --- Overlay: Oxide sources and antenna ---
oxide_size_display = 150  # same size as previous scripts

# Oxide sources
ax.scatter([pos[0] for pos in oxide_positions],
           [pos[1] for pos in oxide_positions],
           s=oxide_size_display,
           facecolors='black',
           edgecolors='black',
           linewidths=0.8,
           label='Oxide Sources')

# Antenna receiver (same size as oxide sources)
ax.scatter(recv_center.x, recv_center.y,
           s=oxide_size_display,
           c='red',
           marker='o',
           label='Antenna Receiver')

# --- Labels and formatting ---
ax.set_xlabel('x (a.u.)')
ax.set_ylabel('y (a.u.)')
ax.set_title('2D: Three Oxide Sources Backscatter (Open Air, No Pipe)')
ax.legend(loc='upper right', frameon=False)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "backscatter_field_3_oxide_open_air.png"), dpi=300)
plt.show()

# === Analyze reflected flux at antenna ===
refl_data = sim.get_fluxes(recv_monitor)
print("Reflected flux at antenna:", refl_data)
