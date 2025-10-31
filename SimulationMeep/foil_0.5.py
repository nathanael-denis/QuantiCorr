import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

# === Directories ===
output_dir = "SimFiles/Pipe_2D_OxideSource_Backscatter"
os.makedirs(output_dir, exist_ok=True)

# === Simulation parameters ===
resolution = 60
sx = 16
sy = 10
dpml = 1.0

# === Materials ===
air = mp.Medium(epsilon=1)
metal = mp.metal
oxide = mp.Medium(epsilon=5)

# === Pipe / walls geometry ===
wall_thickness = 0.2
gap_between_walls = 1.5
oxide_radius = 0.1

geometry = []

# --- Adjusted wall segments with 20% extra height ---
wall_height = (sy/2 - 3.5) * 1.4  # 20% up and 20% down

# Left metallic wall
geometry.append(mp.Block(size=mp.Vector3(wall_thickness, wall_height),
                         center=mp.Vector3(-gap_between_walls/2, -wall_height/2),
                         material=metal))
geometry.append(mp.Block(size=mp.Vector3(wall_thickness, wall_height),
                         center=mp.Vector3(-gap_between_walls/2, wall_height/2),
                         material=metal))

# Right metallic wall
geometry.append(mp.Block(size=mp.Vector3(wall_thickness, wall_height),
                         center=mp.Vector3(gap_between_walls/2, -wall_height/2),
                         material=metal))
geometry.append(mp.Block(size=mp.Vector3(wall_thickness, wall_height),
                         center=mp.Vector3(gap_between_walls/2, wall_height/2),
                         material=metal))

# --- Single oxide “point” source in the middle ---
oxide_center = (0, 0)
geometry.append(mp.Block(size=mp.Vector3(2*oxide_radius, 2*oxide_radius),
                         center=mp.Vector3(*oxide_center),
                         material=oxide))

# === Oxide as Gaussian source ===
sources = [mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.5),
                     component=mp.Ez,
                     center=mp.Vector3(*oxide_center),
                     size=mp.Vector3(0, 2.25*oxide_radius))]

# === Simulation setup ===
sim = mp.Simulation(
    cell_size=mp.Vector3(sx, sy),
    boundary_layers=[mp.PML(dpml)],
    geometry=geometry,
    sources=sources,
    resolution=resolution
)

# --- Antenna / receiver monitor (backscatter) ---
recv_center = mp.Vector3(-sx/2 + 1, 0)
recv_monitor = sim.add_flux(1.0, 0.5, 50, mp.FluxRegion(center=recv_center, size=1.25*mp.Vector3(0, sy)))

# === Run simulation ===
sim.run(until=100)

# === Save Ez field ===
ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sx, sy), component=mp.Ez)
np.save(os.path.join(output_dir, "Ez_field.npy"), ez_data)

# === Plot Ez field and geometry ===
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(ez_data.T, origin='lower', extent=[-sx/2, sx/2, -sy/2, sy/2],
               interpolation='spline36', cmap='RdBu')
plt.colorbar(im, ax=ax, label='Ez field')

# --- Oxide source overlay ---
oxide_size_display = 150
ax.scatter(oxide_center[0], oxide_center[1],
           s=oxide_size_display, facecolors='black', edgecolors='black', linewidths=0.8,
           label='Oxide Source')

# --- Antenna receiver overlay (match oxide size) ---
ax.scatter(recv_center.x, recv_center.y,
           s=oxide_size_display, c='red', marker='o', label='Antenna Receiver')

# --- Pipe walls overlay ---
# Left wall
ax.add_patch(patches.Rectangle((-gap_between_walls/2 - wall_thickness/2, -wall_height),
                               wall_thickness, wall_height, edgecolor='black', facecolor='none', lw=2))
ax.add_patch(patches.Rectangle((-gap_between_walls/2 - wall_thickness/2, 0),
                               wall_thickness, wall_height, edgecolor='black', facecolor='none', lw=2))
# Right wall
ax.add_patch(patches.Rectangle((gap_between_walls/2 - wall_thickness/2, -wall_height),
                               wall_thickness, wall_height, edgecolor='black', facecolor='none', lw=2))
ax.add_patch(patches.Rectangle((gap_between_walls/2 - wall_thickness/2, 0),
                               wall_thickness, wall_height, edgecolor='black', facecolor='none', lw=2))

# --- Labels and formatting ---
ax.set_xlabel('x (a.u.)')
ax.set_ylabel('y (a.u.)')
ax.set_title('2D: Oxide Source Backscatter in Pipe with Antenna Receiver')
ax.legend(loc='upper right', frameon=False)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "backscatter_field.png"), dpi=300)
plt.show()

# === Analyze reflected flux at antenna ===
refl_data = sim.get_fluxes(recv_monitor)
print("Reflected flux at antenna:", refl_data)
