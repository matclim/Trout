// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// 2018 muon-flux test beam: find and fit the drift-tube tracks of the
// muflux_digits RNTuple (Shannon's workflows/muflux_digitise.jsonnet, or the
// 2018 data in the same layout) in the Goliath field, one event per entry,
// into the muflux_tracks RNTuple (src/detectors/muflux/).
//
// Top-level arguments (--tla-str / --tla-code):
//   infile      muflux_digits file (required)
//   events      entries to reconstruct, at most those of infile (required)
//   wire_table  dt_wires.txt exported from the legacy geometry (required)
//   field_map   Goliath map in the SHiP field-map format (required)
//   map         map name inside field_map (default 'goliath_2018')
//   out         output file (default 'muflux_tracks.root')
//   sigma       hit resolution in mm (default 0.27)
//   tube_wall_density  of the drift-tube walls in g/cm3 (default 2.699, aluminium; 0.9 for
//               simulations from the legacy geometry, whose walls are "CoilAluminium")
function(infile, events, wire_table, field_map, map='goliath_2018', out='muflux_tracks.root',
         sigma=0.27, tube_wall_density=2.699)
  {
    stage: 'reconstruction',
    driver: {
      cpp: 'generate_layers',
      layers: {
        event: { parent: 'job', total: events },
      },
    },
    sources: {
      muflux_digits: {
        cpp: 'read_muflux_digits',
        input_file: infile,
        layer: 'event',
      },
      spectrometer_field: {
        cpp: 'acts_field_provider',
        file: field_map,
        map: map,
        translation: [0, 0, 3507.5],  // mm: the map's origin at the legacy z = 350.75 cm
      },
    },
    modules: {
      muflux_tracking: {
        cpp: 'muflux_tracking',
        layer: 'event',
        wire_table: wire_table,
        sigma: sigma,
        tube_wall_density: tube_wall_density,
      },
      output: {
        cpp: 'muflux_tracks_output',
        layer: 'event',
        rntuple_file: out,
      },
    },
  }
