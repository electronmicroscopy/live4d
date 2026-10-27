# START header_appendix keys

The controller that arms the detector, typically the machine with access to the microscope and
scan generator, can describe each series in the Stream2 START `header_appendix`. Live4D reads
it once per series and logs every entry.

Every key is optional. A series without a `header_appendix`, or with one that is not a map, uses
the configured values. Set the appendix as a map (a JSON object, not a JSON-encoded string): a
string such as `'{"scan_width": 256}'` is not parsed and is ignored. Keys not listed here are
ignored. Values that are present must have the listed type; an invalid value rejects the series
or fails the run, as stated per key.

## Keys read by Live4D

| Key | Type | Meaning | When absent | Invalid value |
|---|---|---|---|---|
| `scan_width` | integer, 1 … 2³¹−1 | Probe positions per scan row | `scan.width` | START rejected |
| `scan_height` | integer, 1 … 2³¹−1 | Scan rows | `scan.height` | START rejected |
| `name_pattern` | string | HDF5 file name stem below `writer.output_dir`; `$id` is replaced by the series ID. An empty string keeps the configured pattern. | `writer.name_pattern` | Run fails (non-string, absolute path, `..`, trailing `/` or `.`) |
| `enable_h5_writer` | boolean | `false` skips writing this series | `true` | Run fails (non-boolean) |
| `electron_microscope` | map | EM metadata file content; see [below](#em-metadata-electron_microscope) | No metadata file from Live4D | Run fails |

`scan_width` and `scan_height` form a pair: give both or neither. A START with only one is
rejected. Booleans are not accepted as integers.

`enable_h5_writer` cannot turn writing on when `writer.enabled` is false, because the writer is
then not part of the graph. `name_pattern` and `enable_h5_writer` have no effect without the
writer. See [HDF5 output](holoscan_basic_ingest.md#hdf5-output).

## EM metadata (`electron_microscope`)

With `writer.format: em`, Live4D writes `<name>_em_metadata.h5` beside the master file from
this map when the series starts, and links it from the master file as
`/entry/instrument/electron_microscope`. The map is read only in the `em` format; `nxmx` ignores
it, even when invalid. A series without the map in the `em` format gets only the link, and the
acquisition controller writes the metadata file, as novena does. The `[h5-series]` record reports
`em_metadata_file`, or `null` when Live4D did not write it.

`electron_microscope` is a map whose nested structure equals the `/electron_microscope` group of
the EM metadata file written by novena (`em_metadata/EmMetadata.h`, format version 1.0). Every
dataset in that group may appear at the same path, with the value in the file's unit. Nested
groups are maps; every field is optional at every level.

The file always contains every group and dataset of the format, with novena's types, units and
`NX_class` attributes, so novena's reader, which opens each one, can read it. Omitted fields get
novena's defaults: 0, an empty string, `false`, a one-element `[0]` array, and `"regular"` and
`"raster"` for `scan_type` and `scan_pattern`. Live4D sets `format_version` itself.

A value of the wrong type, a `scan_type` or `scan_pattern` outside its listed values, or a group
that is not a map fails the run before any file of the series is written. Keys that are not
part of the format are not written; each is logged as a warning naming its path.

The file is written once, at START. Values that novena records at the end of an acquisition,
such as `experiment.persons`, `sample` and `description`, must be known when the series is armed.

Types: *string*, *bool* (stored as int8), *uint* (non-negative integer, stored as uint64), *float*
(integer or floating point, stored as float64), *float[]* (array of numbers, or a 1-D float32 or
float64 typed array; one element per scan position for the `irregular_scan` vectors). Booleans are
not accepted as numbers. The `aux_detector` `data` vectors are stored as float32, the
`irregular_scan` vectors as float64.

The whole START message, including the appendix, must fit within 16 KiB, which limits the
per-position arrays to small scans.

Novena's reader reads each vector into a one-element buffer. Arrays with more than one element
are written correctly but cannot be read by that version of novena.

### `electron_microscope`

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `instrument_type` | string | | For example `"TEM"` |
| `manufacturer` | string | | Microscope manufacturer |
| `model` | string | | Microscope model |
| `location` | string | | Site |

### `electron_microscope.experiment`

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `persons` | string | | Operators |
| `time` | string | | Acquisition time, `YYYY-MM-DD HH:MM:SS` |
| `sample` | string | | Sample name |
| `description` | string | | Free text |

### `electron_microscope.electron_source`

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `gun_type` | string | | For example `"Schottky FEG"` |
| `accelerating_voltage` | float | V | Beam energy in eV has the same value |
| `beam_current` | float | A | Probe current |
| `spot_size` | uint | | Spot size setting |

### `electron_microscope.illumination_system`

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `mode` | string | | `"STEM"` or `"TEM"` |
| `submode` | string | | `"scanning"` or `"fixed-beam"` |
| `scan_magnification` | float | | Scan magnification reported by the microscope |
| `scan_field_of_view_x`, `scan_field_of_view_y` | float | m | Scanned field of view |
| `working_distance` | float | m | |
| `beam_diameter` | float | m | Probe diameter |
| `semi_convergence_angle` | float | mrad | Convergence semi-angle |
| `defocus` | float | m | Probe defocus |
| `beam_defining_aperture_type` | string | | For example `"C2"` |
| `beam_defining_aperture_size` | float | m | Aperture diameter |

### `electron_microscope.imaging_system`

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `mode` | string | | `"diffraction"` or `"imaging"` |
| `image_magnification` | float | | Imaging mode |
| `pixel_size_x`, `pixel_size_y` | float | m | Detector pixel in the specimen plane (imaging mode) |
| `camera_length` | float | m | Diffraction mode |
| `reciprocal_pixel_size_x`, `reciprocal_pixel_size_y` | float | mrad | Diffraction-pattern sampling per detector pixel (diffraction mode) |
| `rotation` | float | degrees | Image rotation |
| `defocus` | float | m | |

The detector pixel sizes refer to the images as streamed: with 2×2 detector binning, use the
binned pixel (twice the physical pitch).

### `electron_microscope.scan_controller`

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `manufacturer` | string | | Scan generator manufacturer |
| `model` | string | | Scan generator model |
| `enabled` | bool | | Beam is scanned |
| `rotation` | float | degrees | Scan rotation set on the scan generator |
| `rotation_offset` | float | degrees | Scan rotation reported by the microscope |
| `scan_type` | string | | `"regular"` (default) or `"irregular"` |

### `electron_microscope.scan_controller.regular_scan`

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `scan_pattern` | string | | `"raster"` (default) or `"serpentine"` |
| `dwell_time` | float | s | Time per probe position |
| `flyback_time_line` | float | s | |
| `flyback_time_frame` | float | s | |
| `frame_interval` | float | s | |
| `n_frames` | uint | | Scan frames in the series |
| `n_pixels_x`, `n_pixels_y` | uint | | Scan size; normally equal to `scan_width`, `scan_height` |
| `pixel_size_x`, `pixel_size_y` | float | m | Scan step |

### `electron_microscope.scan_controller.irregular_scan`

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `n_scan_positions` | uint | | |
| `dwell_time` | float[] | s | Per position |
| `position_x`, `position_y` | float[] | m | Per position |

### `electron_microscope.scan_controller.aux_detector_1` … `aux_detector_4`

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `description` | string | | For example `"HAADF"` |
| `gain` | float | | |
| `offset` | float | | |
| `data` | float[] | | Signal per scan position |

### `electron_microscope.sample_stage`

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `manufacturer` | string | | |
| `model` | string | | |
| `position_x`, `position_y`, `position_z` | float | m | |
| `tilt_alpha`, `tilt_beta` | float | degrees | |

The format has no field for detected or incident dose.

## Example

```json
{
  "scan_width": 256,
  "scan_height": 256,
  "name_pattern": "session12/scan_$id",
  "electron_microscope": {
    "instrument_type": "TEM",
    "manufacturer": "FEI Co",
    "model": "Tecnai",
    "electron_source": {"accelerating_voltage": 200000.0},
    "illumination_system": {
      "mode": "STEM",
      "submode": "scanning",
      "scan_magnification": 225000.0,
      "scan_field_of_view_x": 2.56e-7,
      "scan_field_of_view_y": 2.56e-7,
      "semi_convergence_angle": 20.0
    },
    "imaging_system": {
      "mode": "diffraction",
      "camera_length": 0.12,
      "reciprocal_pixel_size_x": 0.46,
      "reciprocal_pixel_size_y": 0.46
    },
    "scan_controller": {
      "enabled": true,
      "scan_type": "regular",
      "regular_scan": {
        "scan_pattern": "raster",
        "dwell_time": 8.3e-6,
        "n_frames": 1,
        "n_pixels_x": 256,
        "n_pixels_y": 256,
        "pixel_size_x": 1.0e-9,
        "pixel_size_y": 1.0e-9
      }
    }
  }
}
```
