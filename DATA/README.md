# Data

The raw data files are **not stored in this repository**: the two CSV files exceed GitHub's 100 MB file limit, and the building sensor data is not ours to publish.

To run the project, place these three files in this `DATA/` folder:

| File | Size | Contents |
|---|---|---|
| `5occupancySensor_MayToDec2024_9MRows.csv` | ~171 MB | Occupancy events: one row per headcount change, all rooms (`floorspaceid`, `headcount`, `collecteddate`, ...) |
| `5EnvSensor_MayToDec2024_180kRows.csv` | ~353 MB | Environmental sensor records: readings stored as JSON in the `jsondata` column (CO2, temperature, humidity, pressure, formaldehyde, VOCs, particulate matter, battery) |
| `Sensor ID and Locations.xlsx` | ~19 KB | Registry mapping environmental sensor IDs to building, room and area |

The datasets were provided for the MMA3001 project. Contact the unit's teaching team for access.

**Known data issues** (handled in the code, see `src/occupancy_model.py`):

* The occupancy export was truncated at Excel's 1,048,575-row limit, so it ends on 6 May 2024 despite its file name.
* `createdate` in the environmental file is the *upload* time, which can be months after the measurement; the measurement time inside the JSON is used instead.
* Environmental sensors 227, 241 and 777 repeat a single frozen reading, and sensor 326 has no readings during the occupancy period, so only sensor 6012002000869 is used.
* The registry lists environmental sensors only; it does not map the occupancy rooms.
