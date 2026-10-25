# Data

The two large CSV files exceed GitHub's 100 MB file limit, so they are stored here as **zipped copies** (`*.csv.zip`, about 15 MB in total). The code reads the zips directly; if you unzip them in this folder, the uncompressed CSVs are used instead (faster to load) and are ignored by Git. The sensor registry is small and stored as it is.

| File | Size | Contents |
|---|---|---|
| `5occupancySensor_MayToDec2024_9MRows.csv.zip` | ~8.5 MB (171 MB unzipped) | Occupancy events: one row per headcount change, all rooms (`floorspaceid`, `headcount`, `collecteddate`, ...) |
| `5EnvSensor_MayToDec2024_180kRows.csv.zip` | ~6.6 MB (353 MB unzipped) | Environmental sensor records: readings stored as JSON in the `jsondata` column (CO2, temperature, humidity, pressure, formaldehyde, VOCs, particulate matter, battery) |
| `Sensor ID and Locations.xlsx` | ~19 KB | Registry mapping environmental sensor IDs to building, room and area |

The datasets were provided for the MMA3001 project. Contact the unit's teaching team for access.

**Known data issues** (handled in the code, see `src/occupancy_model.py`):

* The occupancy export was truncated at Excel's 1,048,575-row limit, so it ends on 6 May 2024 despite its file name.
* `createdate` in the environmental file is the *upload* time, which can be months after the measurement; the measurement time inside the JSON is used instead.
* Environmental sensors 227, 241 and 777 repeat a single frozen reading, and sensor 326 has no readings during the occupancy period, so only sensor 6012002000869 is used.
* The registry lists environmental sensors only; it does not map the occupancy rooms.
