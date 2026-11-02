// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// MufluxEvent.hpp — products of the 2018 muon-flux reconstruction
//
// MufluxDigits: one event of Shannon's muflux_digits RNTuple (or of the 2018
// data converted to the same layout), read by read_muflux_digits.
// MufluxTracks: the tracks muflux_tracking finds and fits in it, written by
// muflux_tracks_output. Lengths in mm, momenta in GeV.

#pragma once

#include <cstdint>
#include <vector>

namespace SHiP::trout::muflux {

struct MufluxDigits {
    std::uint64_t entry = 0;             ///< event: the digits' `entry` field (simulation entry)
    std::vector<std::int32_t> dt_id;     ///< drift-tube channel id (legacy numbering)
    std::vector<float> dt_drift;         ///< drift distance [mm]
    std::vector<std::int32_t> dt_track;  ///< simulated track of the digit (empty for data)
    std::vector<std::int32_t> rpc_id;    ///< RPC strip id (legacy numbering)
};

/// The tracks of one event; the per-track vectors are parallel.
struct MufluxTracks {
    std::uint64_t entry = 0;
    std::int32_t n_digits = 0;         ///< drift-tube digits of the event
    std::int32_t n_unknown = 0;        ///< digits whose channel is not in the wire table
    std::int32_t n_upstream_only = 0;  ///< T1/T2 candidates without a T3/T4 match
    std::int32_t n_rpc = 0;            ///< RPC strips of the event

    std::vector<std::int32_t> q;       ///< charge from the fit
    std::vector<std::int32_t> fit_ok;  ///< 1 if the Kalman fit converged
    std::vector<float> p;              ///< fitted momentum at the first layer
    std::vector<float> sigma_p;
    std::vector<float> chi2;
    std::vector<std::int32_t> ndf;
    std::vector<std::int32_t> n_hits;                    ///< hits used by the fit
    std::vector<std::int32_t> n_x_up, n_stereo, n_down;  ///< hits found by the pattern recognition
    std::vector<float> p_patrec;           ///< momentum from the bend between the tracklets
    std::vector<float> match_dx;           ///< tracklet x difference at the kick plane
    std::vector<std::int32_t> lr_changes;  ///< sides changed by the refits
    std::vector<std::int32_t> n_outliers;  ///< hits left out by the outlier cut
    /// state at the z of the first drift-tube layer (straight from the first fitted point)
    std::vector<float> x0, y0, z0, px0, py0, pz0;
    /// state at the last fitted point
    std::vector<float> x1, y1, z1, px1, py1, pz1;
    /// simulation: most frequent track among the hits, and its fraction (-1, 0 for data)
    std::vector<std::int32_t> true_track;
    std::vector<float> true_frac;
    /// RPC matching (MufluxRpc.hpp; NaN without an RPC strip table): the track
    /// straight from its last state at RPC station 1, minus the nearest RPC track
    /// in x and in y, and its distance to the nearest fired strip per station
    std::vector<float> x_rpc1, y_rpc1, dx_rpc1, dy_rpc1;
    std::vector<float> rpc_dist1, rpc_dist2, rpc_dist3, rpc_dist4, rpc_dist5;
    std::vector<std::int32_t> n_rpc_stations;  ///< stations with a strip within rpc_station_dist

    /// the event's RPC tracks (not parallel to the tracks): coord = slope * z + intercept
    /// [mm], direction 0 measures y, 1 measures x
    std::vector<std::int32_t> rpc_track_direction;
    std::vector<float> rpc_track_slope, rpc_track_intercept;
};

}  // namespace SHiP::trout::muflux
