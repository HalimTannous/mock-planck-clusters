# mock-planck-clusters

Generate a mock galaxy-cluster catalogue over the Planck SZ survey footprint,
starting from LambdaCDM and the halo mass function — no observational
selection effects included (yet).

## What this does

1. Set a LambdaCDM cosmology (Planck 2018 parameters, via
   [Colossus](https://bdiemer.bitbucket.io/colossus/)).
2. Evaluate the Tinker (2008) halo mass function for the `M500c` mass
   definition on a `(z, ln M500)` grid.
3. Combine it with the comoving volume element to get the differential
   halo abundance on the sky, `dN / (dz dlnM500)`, in deg⁻².
4. Read the Planck survey mask (HEALPix) to get the survey solid angle,
   `Omega_survey`.
5. Compute the expected number of haloes per `(z, M500)` bin,
   `N = Omega_survey * dN/(dz dlnM500) * dz * dlnM500`, and draw a Poisson
   realisation of it.
6. Scatter each mock cluster's redshift and mass uniformly inside its bin,
   and its sky position uniformly inside the survey mask.
7. Save the resulting catalogue (`RA`, `DEC`, `REDSHIFT`, `M500`) to CSV and
   FITS, and produce a set of diagnostic plots: the abundance curves,
   1D/2D/3D projections of `dN/(dz dlnM500)`, and mock-vs-model comparisons.

## Scope — no selection function

This is the **underlying halo population** predicted by LambdaCDM over the
Planck footprint and mass/redshift range requested, not a mock of the
detected PSZ2 catalogue. It does not apply the Planck SZ selection function
(the completeness of the detection pipeline as a function of `M500`, `z`,
and the local map noise), so it will contain far more objects than the real
PSZ2 catalogue (~1650 confirmed clusters).

Adding a selection function later would mean, roughly:

- loading the completeness curves / noise map that ship with the Planck SZ
  selection-function products,
- for each mock cluster, looking up the local noise at its sky position and
  the completeness `Q(M500, z, noise)`,
- keeping the cluster with probability `Q` instead of keeping every object.

That step is intentionally left out for now.

## Requirements

See [requirements.txt](requirements.txt). Notably
[healpy](https://healpy.readthedocs.io/) and
[Colossus](https://bdiemer.bitbucket.io/colossus/).

```bash
pip install -r requirements.txt
```

## Data

The script expects two Planck HEALPix survey-mask FITS files in
`catalogs/`:

- `HFI_PCCS_SZ-selfunc-union-survey_2.02.fits` (PSZ1)
- `HFI_PCCS_SZ-selfunc-union-survey_R2.08.fits` (PSZ2)

Download them from the
[Planck Legacy Archive](https://pla.esac.esa.int/) and place them in
`catalogs/` (not tracked in this repo — see `.gitignore`).

## Usage

```bash
python mock_planck_clusters.py
```

This prints the cosmology, survey area, expected/realised cluster counts and
catalogue statistics to the terminal, saves `mock_planck_clusters.csv` and
`mock_planck_clusters.fits`, and shows the diagnostic plots.

Key parameters (top of the script):

| Variable | Meaning | Default |
|---|---|---|
| `z_min`, `z_max`, `delta_z` | redshift grid | `0.01`, `3.0`, `0.01` |
| `M_min`, `M_max`, `delta_lnM` | `M500` grid (uniform in `ln M`) | `5e14`, `5e15` Msun, `0.01` |

## Output

- `mock_planck_clusters.csv`, `mock_planck_clusters.fits` — the catalogue
  (`RA`, `DEC`, `REDSHIFT`, `M500`).
- Diagnostic plots: `dN/(dz dlnM500)` vs `ln M500` per redshift, `dN/dz` and
  `dN/dlnM500` (model vs mock), cumulative `N(>M500)` (model vs mock), a 2D
  map and a 3D surface of `dN/(dz dlnM500)`, and sky distributions
  (Cartesian and Mollweide).

## License

[MIT](LICENSE)
