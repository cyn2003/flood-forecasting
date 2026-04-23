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

import pandas as pd
import xarray as xr

from googlehydrology.datasetzoo.caravan import (
    load_caravan_attributes,
    load_caravan_timeseries_together,
)
from googlehydrology.datasetzoo.multimet import Multimet
from googlehydrology.utils.config import Config


class Caravan(Multimet):
    """Dataset class for local Caravan data.

    This class adapts the local Caravan NetCDF/CSV data layout to the current
    GoogleHydrology dataset interface. It supports historical dynamic inputs
    from Caravan time-series files, static attributes from Caravan attributes
    files, and streamflow targets.

    Forecast inputs are constructed as perfect forecasts from the same Caravan
    time-series files: lead time ``k`` at issue date ``t`` uses the value from
    ``t + k``.
    """

    def __init__(
        self,
        cfg: Config,
        is_train: bool,
        period: str,
        basins: list[str] | None = None,
        compute_scaler: bool = True,
    ):
        cfg.as_dict().setdefault('forecast_inputs', [])
        cfg.as_dict().setdefault('union_mapping', {})
        super().__init__(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basins=basins,
            compute_scaler=compute_scaler,
        )

    def _load_hindcast_features(self) -> list[xr.Dataset]:
        """Load hindcast inputs from local Caravan time-series files."""
        hindcast_only_features = [
            feature
            for feature in self._hindcast_features
            if feature not in self._forecast_features
        ]
        if not hindcast_only_features:
            return []
        ds = load_caravan_timeseries_together(
            data_dir=self._dynamics_data_path,
            basins=self._basins,
            target_features=hindcast_only_features,
            csv=self._cfg.load_as_csv,
        )
        return [ds]

    def _load_forecast_features(self) -> list[xr.Dataset]:
        """Load forecast inputs as perfect forecasts from Caravan time series."""
        if not self._forecast_features:
            return []

        ds = load_caravan_timeseries_together(
            data_dir=self._dynamics_data_path,
            basins=self._basins,
            target_features=self._forecast_features,
            csv=self._cfg.load_as_csv,
        )

        lead_time_datasets = []
        for lead_time in range(0, self.lead_time + 1):
            shifted = ds.shift(date=-lead_time)
            shifted = shifted.expand_dims(
                lead_time=[pd.Timedelta(days=lead_time)]
            )
            lead_time_datasets.append(shifted)

        forecast_ds = xr.concat(lead_time_datasets, dim='lead_time')
        forecast_ds['lead_time'].attrs['units'] = 'timedelta (days)'
        return [forecast_ds]

    def _load_static_features(self) -> xr.Dataset:
        """Load static attributes from local Caravan attribute files."""
        return load_caravan_attributes(
            data_dir=self._statics_data_path,
            basins=self._basins,
            features=self._static_features,
        )

    def _load_target_features(self) -> xr.Dataset:
        """Load target variables from local Caravan time-series files."""
        return load_caravan_timeseries_together(
            data_dir=self._targets_data_path,
            basins=self._basins,
            target_features=self._target_features,
            csv=self._cfg.load_as_csv,
        )
