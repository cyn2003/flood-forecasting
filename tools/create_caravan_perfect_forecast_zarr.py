# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Caravan dynamic variables into local Zarr products for `multimet`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from googlehydrology.datasetzoo.caravan import load_caravan_timeseries_together


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caravan-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--variables', nargs='+', required=True)
    parser.add_argument('--basin-file', type=Path, default=None)
    parser.add_argument('--hindcast-product', type=str, default='caravan_obs')
    parser.add_argument(
        '--forecast-product', type=str, default='caravan_perfect'
    )
    parser.add_argument('--lead-time', type=int, required=True)
    parser.add_argument('--csv', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_basins(
    caravan_root: Path, basin_file: Path | None, csv: bool
) -> list[str]:
    if basin_file is not None:
        return [
            line.strip()
            for line in basin_file.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    kind = 'csv' if csv else 'netcdf'
    ext = '.csv' if csv else '.nc'
    timeseries_root = caravan_root / 'timeseries' / kind
    basins = sorted(path.stem for path in timeseries_root.rglob(f'*{ext}'))
    if not basins:
        raise FileNotFoundError(
            f'No Caravan timeseries files found under {timeseries_root}.'
        )
    return basins


def cast_float32(ds: xr.Dataset) -> xr.Dataset:
    for name in ds.data_vars:
        if np.issubdtype(ds[name].dtype, np.floating):
            ds[name] = ds[name].astype(np.float32)
    return ds


def rename_variables(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    for name in ds.data_vars:
        if name == 'total_precipitation_sum_mswep':
            rename_map[name] = 'mswep_total_precipitation_sum'
        elif name == 'temperature_2m_mean_mswx':
            rename_map[name] = 'mswx_temperature_2m_mean'
        else:
            rename_map[name] = f'era5land_{name}'
    return ds.rename(rename_map)


def build_hindcast_dataset(
    caravan_root: Path,
    basins: list[str],
    variables: list[str],
    csv: bool,
) -> xr.Dataset:
    ds = load_caravan_timeseries_together(
        data_dir=caravan_root,
        basins=basins,
        target_features=variables,
        csv=csv,
    )
    ds = rename_variables(ds)
    return cast_float32(ds.transpose('date', 'basin', ...))


def build_forecast_dataset(hindcast_ds: xr.Dataset, lead_time: int) -> xr.Dataset:
    lead_datasets = []
    for lead in range(1, lead_time + 1):
        shifted = hindcast_ds.shift(date=-lead)
        shifted = shifted.expand_dims(lead_time=[pd.Timedelta(days=lead)])
        lead_datasets.append(shifted)

    forecast_ds = xr.concat(lead_datasets, dim='lead_time')
    forecast_ds['lead_time'].attrs.pop('units', None)
    return cast_float32(
        forecast_ds.transpose('date', 'lead_time', 'basin', ...)
    )


def rechunk_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    chunk_map = {}
    if 'date' in ds.dims:
        chunk_map['date'] = min(365, ds.sizes['date'])
    if 'lead_time' in ds.dims:
        chunk_map['lead_time'] = ds.sizes['lead_time']
    if 'basin' in ds.dims:
        chunk_map['basin'] = min(256, ds.sizes['basin'])
    return ds.chunk(chunk_map)


def write_zarr(ds: xr.Dataset, store: Path, overwrite: bool) -> None:
    if overwrite and store.exists():
        import shutil

        shutil.rmtree(store)
    store.parent.mkdir(parents=True, exist_ok=True)
    ds = rechunk_for_zarr(ds)
    ds.to_zarr(store, mode='w', consolidated=False)


def main() -> None:
    args = parse_args()

    basins = load_basins(args.caravan_root, args.basin_file, args.csv)
    hindcast_ds = build_hindcast_dataset(
        args.caravan_root, basins, args.variables, args.csv
    )
    forecast_ds = build_forecast_dataset(hindcast_ds, args.lead_time)

    hindcast_store = args.output_root / args.hindcast_product / 'timeseries.zarr'
    forecast_store = args.output_root / args.forecast_product / 'timeseries.zarr'

    write_zarr(hindcast_ds, hindcast_store, args.overwrite)
    write_zarr(forecast_ds, forecast_store, args.overwrite)

    print(f'Wrote hindcast product to {hindcast_store}')
    print(f'Wrote forecast product to {forecast_store}')
    print(f'Basins: {len(basins)}')
    print(f'Variables: {", ".join(args.variables)}')
    print(f'Lead times: 1..{args.lead_time} days')


if __name__ == '__main__':
    main()
