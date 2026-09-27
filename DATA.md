# Data access

Large source files, derived Parquet tables, and saved model files are intentionally not stored in Git. They are public or reproducible inputs, but the full working directory is tens of gigabytes.

## Historical model table

The 1980–2020 weekly feature archive comes from the USGS ScienceBase release [Data and models for predicting drought from streamflow](https://doi.org/10.5066/P1X5VH96).

```powershell
.\.venv\Scripts\ingest-historical.exe
.\.venv\Scripts\build-weekly.exe --historical --delete-zip
.\.venv\Scripts\build-weekly.exe --matched-history
```

The drainage-area polygons used to average gridded weather come from ScienceBase item [`64510406d34eefd5da80a2c7`](https://www.sciencebase.gov/catalog/item/64510406d34eefd5da80a2c7):

```powershell
.\.venv\Scripts\ingest-basins.exe
```

## Live inputs

Each ingest command downloads directly from the named public provider and writes a file under `data/raw/`.

| Command | Provider |
|---|---|
| `ingest-flow` | [USGS Water Data API](https://api.waterdata.usgs.gov/) |
| `ingest-gridmet` | [gridMET](https://www.climatologylab.org/gridmet.html) |
| `ingest-nldas` | [NASA NLDAS-2 Noah](https://disc.gsfc.nasa.gov/datasets/NLDAS_NOAH0125_H_2.0/summary) |
| `ingest-smap` | [NASA SMAP SPL3SMP](https://nsidc.org/data/spl3smp/versions/9) |
| `ingest-swe` | [University of Arizona SWE](https://climate.arizona.edu/data/UA_SWE/) |
| `ingest-gefs` | [NOAA GEFS public archive](https://registry.opendata.aws/noaa-gefs/) |
| `ingest-nmme` | [NOAA NMME](https://www.cpc.ncep.noaa.gov/products/NMME/) |
| `ingest-climate` | NOAA CPC/NCEI and SILSO climate indexes |
| `ingest-reservoirs` | USGS, Reclamation RISE, USACE CWMS, and Water Data for Texas |
| `ingest-openet` | USGS SSEBop and the OpenET point API |

Use `COMMAND --help` to select a backfill window. NASA downloads require an Earthdata account; USGS Water Data and OpenET require their respective API keys. The expected environment-variable names are in `.env.example`.

After the source files are present:

```powershell
.\.venv\Scripts\build-weekly.exe --from-daily
```

The production desktop retains the 2019–2026 backfill used for the published monitor. Reproducing the exact published metrics requires the same date ranges and source versions; routine Monday runs download only recent data and merge it into those local archives.
