import meep as mp
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerPatch

# === Output directory ===
output_dir = "SimFiles/NoPipe_OxideSource_Backscatter"
os.makedirs(output_dir, exist_ok=True)

# === Simulation parameters ===
resolution = 50
sx, sy, dpml = 10, 5, 1.0
frequency = 1.0

# === Material ===
oxide = mp.Medium(epsilon=5)

# === Geometry: single oxide stack ===
oxide_center = mp.Vector3(0, 0)
oxide_radius = 0.1
geometry = [
    mp.Cylinder(radius=oxide_radius, center=oxide_center, height=mp.inf, material=oxide)
]

# === Oxide as Gaussian source ===
sources = [
    mp.Source(
        src=mp.GaussianSource(frequency=frequency, fwidth=0.5),
        component=mp.Ez,
        center=oxide_center,
        size=mp.Vector3(0, 2*oxide_radius)
    )
]

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
recv_monitor = sim.add_flux(frequency, 0.5, 50, mp.FluxRegion(center=recv_center, size=mp.Vector3(0, sy)))

# === Run simulation ===
sim.run(mp.at_every(0.1, mp.output_efield_z), until=20)

# === Save Ez field ===
ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sx, sy), component=mp.Ez)
np.save(os.path.join(output_dir, "Ez_field.npy"), ez_data)

# === Plot Ez field ===
fig, ax = plt.subplots(figsize=(8, 4))

# --- Harmonize color scale: symmetric around zero ---
max_abs = np.max(np.abs(ez_data))
im = ax.imshow(ez_data.T, origin='lower', interpolation='spline36', cmap='RdBu',
               extent=[-sx/2, sx/2, -sy/2, sy/2],
               vmin=-max_abs, vmax=max_abs)
plt.colorbar(im, ax=ax, label='Ez field')

# --- Overlay: oxide source (solid circle) ---
oxide_patch = patches.Circle((oxide_center.x, oxide_center.y), oxide_radius,
                             edgecolor='black', facecolor='black', lw=2, label='Oxide Source')
ax.add_patch(oxide_patch)

# --- Overlay: antenna receiver ---
antenna_patch = patches.Circle((recv_center.x, recv_center.y), oxide_radius,
                               edgecolor='red', facecolor='red', lw=2, label='Antenna Receiver')
ax.add_patch(antenna_patch)

# --- Custom legend handler for smaller circles ---
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        scale = 0.5  # smaller legend circle
        center = (width / 2 - xdescent, height / 2 - ydescent)
        p = patches.Circle(xy=center, radius=width/2 * scale)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

# --- Add legend with circular markers ---
ax.legend(loc='upper right', frameon=False, handler_map={patches.Circle: HandlerCircle()})

# --- Labels and formatting ---
ax.set_xlabel('x (a.u.)')
ax.set_ylabel('y (a.u.)')
ax.set_title('Backscatter Field with Oxide Source and Antenna Receiver (No Pipe)')
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "backscatter_field_harmonized.png"), dpi=300)
plt.show()

# === Analyze reflected flux at antenna ===
refl_data = sim.get_fluxes(recv_monitor)
print("Reflected flux at antenna:", refl_data)
