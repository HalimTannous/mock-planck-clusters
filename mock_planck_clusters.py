"""
Mock Planck galaxy-cluster catalogue from LambdaCDM + halo mass function.

Pipeline
LambdaCDM (Planck18)
  -> Tinker (2008) halo mass function, M500c            [Colossus]
  -> dN / (dz dlnM500 dOmega)                            [per steradian]
  -> dN / (dz dlnM500)                                   [per deg^2]
  -> N = Omega_survey * dN/(dz dlnM500) * dz * dlnM500   [expected counts]
  -> Poisson realisation -> mock (z, M500) list
  -> random sky positions inside the Planck survey mask
  -> catalogue (RA, DEC, REDSHIFT, M500) + projection / comparison plots

Scope
This script only builds the underlying halo population predicted by
LambdaCDM + the halo mass function over the Planck survey footprint. It does
NOT apply the Planck SZ selection function (completeness as a function of
M500, z and local noise), so the resulting catalogue is much larger than the
real PSZ2 catalogue. See the README for how a selection function could be
added on top of this.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3d projection on old mpl)

from colossus.cosmology import cosmology
from colossus.lss import mass_function
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

# 1. Cosmology
cosmology.setCosmology("planck18")
cosmo = cosmology.getCurrent()

h = cosmo.H0 / 100.0

print("Cosmology:")
print(f"  Cosmology = {cosmo.name}")
print(f"  H0        = {cosmo.H0:.3f} km/s/Mpc")
print(f"  h         = {h:.5f}")
print(f"  Omega_m   = {cosmo.Om0:.5f}")
print(f"  sigma8    = {cosmo.sigma8:.5f}")
print(f"  ns        = {cosmo.ns:.5f}")


# 2. Redshift grid
z_min = 0.01
z_max = 3.0
delta_z = 0.01

z_edges = np.arange(z_min, z_max + delta_z, delta_z)
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])


# 3. Mass grid (uniform in ln M500)
M_min = 5e14
M_max = 5e15
delta_lnM = 0.01

lnM_edges = np.arange(np.log(M_min), np.log(M_max) + delta_lnM, delta_lnM)
lnM_centers = 0.5 * (lnM_edges[:-1] + lnM_edges[1:])

M_centers = np.exp(lnM_centers)

# Colossus works in Msun/h:  M[Msun/h] = M[Msun] * h
M_colossus = M_centers * h


# 4. Planck survey mask -> survey solid angle
def load_healpix_mask(filename):
    mask = hp.read_map(filename, field=0)
    return np.asarray(mask, dtype=float)


def calculate_survey_area(mask1_file, mask2_file):
    mask1 = load_healpix_mask(mask1_file)
    mask2 = load_healpix_mask(mask2_file)

    if len(mask1) != len(mask2):
        raise ValueError("The two Planck masks have different NSIDE values.")

    nside = hp.npix2nside(len(mask1))

    # keep a pixel only if it is valid (finite, positive) in BOTH masks
    survey_mask = (
        np.isfinite(mask1) & np.isfinite(mask2)
        & (mask1 > 0) & (mask2 > 0)
    )

    npix_survey = np.sum(survey_mask)
    npix_total = hp.nside2npix(nside)

    sky_fraction = npix_survey / npix_total

    pixel_area_sr = hp.nside2pixarea(nside)
    omega_survey_sr = npix_survey * pixel_area_sr
    omega_survey_deg2 = omega_survey_sr * (180.0 / np.pi) ** 2

    return omega_survey_deg2, omega_survey_sr, sky_fraction, survey_mask, nside


MASK1_FILE = "catalogs/HFI_PCCS_SZ-selfunc-union-survey_2.02.fits"
MASK2_FILE = "catalogs/HFI_PCCS_SZ-selfunc-union-survey_R2.08.fits"

(
    omega_survey_deg2,
    omega_survey_sr,
    sky_fraction,
    survey_mask,
    mask_nside,
) = calculate_survey_area(MASK1_FILE, MASK2_FILE)

print()
print("Planck survey:")
print(f"  NSIDE        = {mask_nside}")
print(f"  Sky fraction = {sky_fraction:.6f}")
print(f"  Area         = {omega_survey_deg2:.2f} deg^2")
print(f"  Solid angle  = {omega_survey_sr:.6f} sr")


# 5. Comoving volume element  dV / (dz dOmega)   [Mpc^3 / sr], flat universe
def dV_dz_dOmega(z):
    c_km_s = 299792.458                       # speed of light [km/s]

    Hz = cosmo.Hz(z)                          # H(z) [km/s/Mpc]

    DM_hmpc = cosmo.comovingDistance(z_min=0.0, z_max=z)   # [Mpc/h]
    DM_mpc = DM_hmpc / h                                   # [Mpc]

    return c_km_s / Hz * DM_mpc ** 2          # [Mpc^3 / sr] per unit z


# 6. Differential halo abundance  dN / (dz dlnM500)  [deg^-2]
dN_dz_dlnM_deg2 = np.zeros((len(z_centers), len(M_centers)))

sr_to_deg2 = (180.0 / np.pi) ** 2            # 1 sr = 3282.806... deg^2

for iz, z in enumerate(z_centers):

    # Tinker 2008 mass function for the M500c definition
    dn_dlnM = mass_function.massFunction(
        M_colossus, z,
        q_in="M", q_out="dndlnM",
        mdef="500c", model="tinker08",
    )                                        # [ (Mpc/h)^-3 ] = [ h^3 Mpc^-3 ]

    dn_dlnM_mpc3 = dn_dlnM * h ** 3          # -> [ Mpc^-3 ]

    volume_element = dV_dz_dOmega(z)         # [ Mpc^3 / sr ]

    dN_dz_dlnM_sr = dn_dlnM_mpc3 * volume_element      # [ sr^-1 ]  per dz per dlnM
    dN_dz_dlnM_deg2[iz] = dN_dz_dlnM_sr / sr_to_deg2   # [ deg^-2 ] per dz per dlnM


# 7. Expected number of haloes
N_expected_bin = omega_survey_deg2 * dN_dz_dlnM_deg2 * delta_z * delta_lnM
N_expected_total = np.sum(N_expected_bin)

print()
print("Expected number of haloes:")
print(f"  N = {N_expected_total:.1f}")


# 8. Poisson realisation
rng = np.random.default_rng(12345)

N_mock_bin = rng.poisson(N_expected_bin)
N_mock_total = np.sum(N_mock_bin)

print()
print("Mock catalogue:")
print(f"  N = {N_mock_total}")


# 9. Assign (z, M500) to every mock object
z_list = []
M_list = []

for iz, z_center in enumerate(z_centers):
    for im, lnM_center in enumerate(lnM_centers):

        N = N_mock_bin[iz, im]
        if N == 0:
            continue

        # uniform inside the redshift bin
        z_values = z_center + (rng.random(N) - 0.5) * delta_z

        # uniform inside the ln-mass bin
        lnM_values = lnM_center + (rng.random(N) - 0.5) * delta_lnM

        z_list.append(z_values)
        M_list.append(np.exp(lnM_values))

z_mock = np.concatenate(z_list)
M_mock = np.concatenate(M_list)

print(f"  Generated = {len(z_mock)}")


# 10. Random sky positions inside the Planck mask (rejection sampling)
N_needed = len(z_mock)

ra_inside = []
dec_inside = []
N_collected = 0

while N_collected < N_needed:

    N_remaining = N_needed - N_collected
    batch_size = max(10000, int(N_remaining / max(sky_fraction, 0.01) * 1.2))

    # uniform on the sphere: uniform RA, uniform in sin(dec)
    sin_dec = rng.uniform(-1.0, 1.0, batch_size)          # renamed from `u`
    ra_batch = rng.uniform(0.0, 360.0, batch_size)
    dec_batch = np.degrees(np.arcsin(sin_dec))

    # Planck masks are in Galactic coordinates -> convert candidates before the lookup
    gal_batch = SkyCoord(ra=ra_batch * u.deg, dec=dec_batch * u.deg, frame="icrs").galactic
    theta_batch = np.radians(90.0 - gal_batch.b.deg)
    phi_batch = np.radians(gal_batch.l.deg)
    pix_batch = hp.ang2pix(mask_nside, theta_batch, phi_batch, nest=False)

    valid_batch = survey_mask[pix_batch]
    ra_inside.append(ra_batch[valid_batch])
    dec_inside.append(dec_batch[valid_batch])
    N_collected += int(np.sum(valid_batch))


ra_mock = np.concatenate(ra_inside)[:N_needed]
dec_mock = np.concatenate(dec_inside)[:N_needed]

print()
print("Planck mask:")
print(f"  Final positions = {len(ra_mock)}")


# 11. Build, sort and save the catalogue
mock_catalogue = pd.DataFrame(
    {
        "RA": ra_mock,
        "DEC": dec_mock,
        "REDSHIFT": z_mock,
        "M500": M_mock,
    }
)

mock_catalogue = mock_catalogue.sort_values("REDSHIFT").reset_index(drop=True)

print()
print("Final Planck-masked catalogue:")
print(f"  Number of clusters = {len(mock_catalogue)}")
print()
print(mock_catalogue.head())

mock_catalogue.to_csv("mock_planck_clusters.csv", index=False)

mock_table = Table.from_pandas(mock_catalogue)
mock_table.write("mock_planck_clusters.fits", overwrite=True)

print()
print("Saved:")
print("  mock_planck_clusters.csv")
print("  mock_planck_clusters.fits")


# 12. Projections of the model and comparison with the mock

# 12a. dN/(dz dlnM) vs ln M500, one curve per redshift
z_plot_values = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00]

plt.figure(figsize=(9, 6))
for z_plot in z_plot_values:
    iz = np.argmin(np.abs(z_centers - z_plot))
    plt.plot(lnM_centers, dN_dz_dlnM_deg2[iz], label=f"z = {z_centers[iz]:.2f}")
plt.yscale("log")
plt.xlabel(r"$\ln(M_{500}/M_\odot)$", fontsize=13)
plt.ylabel(r"$dN/(dz\,d\ln M_{500})$ [deg$^{-2}$]", fontsize=13)
plt.title(r"$M_{500c}$ halo abundance", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.show()


# 12b. integrate over ln M -> dN/dz   (model)
dN_dz_model = omega_survey_deg2 * np.sum(dN_dz_dlnM_deg2, axis=1) * delta_lnM

plt.figure(figsize=(9, 6))
plt.plot(z_centers, dN_dz_model, linewidth=2)
plt.yscale("log")
plt.xlabel(r"$z$", fontsize=13)
plt.ylabel(r"$dN/dz$", fontsize=13)
plt.title("Predicted cluster redshift distribution", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 12c. mock redshift histogram
plt.figure(figsize=(9, 6))
plt.hist(mock_catalogue["REDSHIFT"], bins=z_edges, histtype="step")
plt.xlabel(r"$z$", fontsize=13)
plt.ylabel("Number of clusters", fontsize=13)
plt.title("Mock Planck cluster redshift distribution", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 12d. mock mass histogram
plt.figure(figsize=(9, 6))
plt.hist(np.log10(mock_catalogue["M500"]), bins=50, histtype="step")
plt.xlabel(r"$\log_{10}(M_{500}/M_\odot)$", fontsize=13)
plt.ylabel("Number of clusters", fontsize=13)
plt.title("Mock Planck cluster mass distribution", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 12e. mass vs redshift
plt.figure(figsize=(9, 6))
plt.scatter(mock_catalogue["REDSHIFT"], mock_catalogue["M500"], s=2, alpha=0.25)
plt.yscale("log")
plt.xlabel(r"$z$", fontsize=13)
plt.ylabel(r"$M_{500}\ [M_\odot]$", fontsize=13)
plt.title("Mock Planck cluster mass-redshift distribution", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 12f. sky distribution
plt.figure(figsize=(10, 5))
plt.scatter(mock_catalogue["RA"], mock_catalogue["DEC"], s=1, alpha=0.25)
plt.xlabel("RA [deg]", fontsize=13)
plt.ylabel("DEC [deg]", fontsize=13)
plt.title("Mock Planck cluster sky distribution", fontsize=14)
plt.xlim(0, 360)
plt.ylim(-90, 90)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 12f-bis. sky distribution, Mollweide projection with the survey mask
hp.mollview(
    survey_mask.astype(float),
    title="Planck mask + mock clusters (Mollweide)",
    cbar=False, cmap="Greys", min=0, max=1,
)
hp.graticule(dpar=30, dmer=30, alpha=0.3)
theta = np.radians(90.0 - mock_catalogue["DEC"].values)
phi = np.radians(mock_catalogue["RA"].values)
hp.projscatter(theta, phi, s=1, alpha=0.2, color="C0")
plt.show()


# 12g. dN/dz : model vs mock
plt.figure(figsize=(9, 6))
plt.plot(z_centers, dN_dz_model, linewidth=2, label="HMF model")
plt.hist(
    mock_catalogue["REDSHIFT"], bins=z_edges,
    weights=np.ones(len(mock_catalogue)) / delta_z,
    histtype="step", linewidth=1.5, label="Mock catalogue",
)
plt.yscale("log")
plt.xlabel(r"$z$", fontsize=13)
plt.ylabel(r"$dN/dz$", fontsize=13)
plt.title("Cluster redshift distribution: model vs mock", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# 12h. integrate over z -> dN/dlnM   (model), then compare with the mock
dN_dlnM_model = omega_survey_deg2 * np.sum(dN_dz_dlnM_deg2, axis=0) * delta_z

M_edges = np.exp(lnM_edges)                 # bins uniform in ln M

plt.figure(figsize=(9, 6))
plt.plot(M_centers, dN_dlnM_model, linewidth=2, label="HMF model")
plt.hist(
    mock_catalogue["M500"], bins=M_edges,
    weights=np.ones(len(mock_catalogue)) / delta_lnM,
    histtype="step", linewidth=1.5, label="Mock catalogue",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$M_{500}\ [M_\odot]$", fontsize=13)
plt.ylabel(r"$dN/d\ln M_{500}$", fontsize=13)
plt.title("Cluster mass distribution: model vs mock", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# 12i. cumulative abundance N(>M) : model vs mock
mass_thresholds = np.logspace(np.log10(M_min), np.log10(M_max), 100)

N_cumulative_model = []
for mass_threshold in mass_thresholds:
    valid = M_centers >= mass_threshold
    N = (
        omega_survey_deg2
        * np.sum(dN_dz_dlnM_deg2[:, valid])
        * delta_z * delta_lnM
    )
    N_cumulative_model.append(N)
N_cumulative_model = np.asarray(N_cumulative_model)

N_cumulative_mock = np.array(
    [np.sum(mock_catalogue["M500"] >= m) for m in mass_thresholds]
)

plt.figure(figsize=(9, 6))
plt.plot(mass_thresholds, N_cumulative_model, linewidth=2, label="HMF model")
plt.step(mass_thresholds, N_cumulative_mock, where="post", linewidth=1.5,
         label="Mock catalogue")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$M_{500}\ [M_\odot]$", fontsize=13)
plt.ylabel(r"$N(>M_{500})$", fontsize=13)
plt.title("Cumulative cluster abundance: model vs mock", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# 12j. 2d map of dN/(dz dlnM)
plt.figure(figsize=(9, 6))
plt.pcolormesh(z_edges, M_edges, dN_dz_dlnM_deg2.T, shading="auto")
plt.yscale("log")
plt.xlabel(r"$z$", fontsize=13)
plt.ylabel(r"$M_{500}\ [M_\odot]$", fontsize=13)
plt.title(r"$dN/(dz\,d\ln M_{500})$ per deg$^2$", fontsize=14)
plt.colorbar(label=r"$dN/(dz\,d\ln M_{500})$ [deg$^{-2}$]")
plt.tight_layout()
plt.show()


# 12k. 3d surface of dN/(dz dlnM)
Z_3D, LN_M_3D = np.meshgrid(z_centers, lnM_centers, indexing="ij")
M_LOG10_3D = np.log10(np.exp(LN_M_3D))
HMF_3D = dN_dz_dlnM_deg2

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
surface = ax.plot_surface(
    Z_3D, M_LOG10_3D, HMF_3D,
    cmap="viridis", linewidth=0, antialiased=True,
)
ax.set_xlabel(r"$z$", fontsize=12, labelpad=10)
ax.set_ylabel(r"$\log_{10}(M_{500}/M_\odot)$", fontsize=12, labelpad=10)
ax.set_zlabel(r"$dN/(dz\,d\ln M_{500})$ [deg$^{-2}$]", fontsize=12, labelpad=10)
ax.set_title(r"Three-dimensional halo abundance $dN/(dz\,d\ln M_{500})$",
             fontsize=14, pad=20)
fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.1,
             label=r"$dN/(dz\,d\ln M_{500})$ [deg$^{-2}$]")
ax.view_init(elev=25, azim=-120)
plt.tight_layout()
plt.show()


# 13. Final statistics
relative_difference = (len(mock_catalogue) - N_expected_total) / N_expected_total * 100.0

print()
print("Final catalogue statistics:")
print(f"  Expected HMF number   = {N_expected_total:.1f}")
print(f"  Mock catalogue number = {len(mock_catalogue)}")
print(f"  Difference            = {len(mock_catalogue) - N_expected_total:.1f}")
print(f"  Relative difference   = {relative_difference:.2f}%   (Poisson scatter ~ {100.0 / np.sqrt(N_expected_total):.2f}%)")
print(f"  Minimum M500 = {mock_catalogue['M500'].min():.3e} Msun")
print(f"  Maximum M500 = {mock_catalogue['M500'].max():.3e} Msun")
print(f"  Minimum z    = {mock_catalogue['REDSHIFT'].min():.3f}")
print(f"  Maximum z    = {mock_catalogue['REDSHIFT'].max():.3f}")
