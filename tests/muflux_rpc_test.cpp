// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// muflux_rpc_test.cpp — RPC tracks and matching of MufluxRpc.hpp on synthetic
// events: strips laid out as in the 2018 geometry (5 stations, 116 horizontal
// strips of 10.6 mm from y = 801 mm down, 184 vertical strips of 10.6 mm from
// x = 973 mm down), straight muons firing the strip they cross and sometimes a
// neighbour, noise strips.

#include "detectors/muflux/MufluxRpc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

namespace mf = SHiP::trout::muflux;

int failures = 0;
void check(bool ok, std::string const& what) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++failures;
    }
}

constexpr double kPitch = 10.6;
constexpr std::array<double, 5> kZ{8784.3, 9733.9, 10287.8, 10839.0, 11387.1};

std::map<std::int32_t, mf::RpcStrip> strips() {
    std::map<std::int32_t, mf::RpcStrip> s;
    for (int st = 1; st <= 5; ++st) {
        for (int c = 1; c <= 116; ++c) {
            std::int32_t const id = st * 10000 + c;
            s[id] = {.id = id,
                     .station = st,
                     .direction = 0,
                     .x = 0,
                     .y = 801.2 - kPitch * (c - 1),
                     .z = kZ[st - 1]};
        }
        for (int c = 1; c <= 184; ++c) {
            std::int32_t const id = st * 10000 + 1000 + c;
            s[id] = {.id = id,
                     .station = st,
                     .direction = 1,
                     .x = 972.7 - kPitch * (c - 1),
                     .y = 194,
                     .z = kZ[st - 1] + 8};
        }
    }
    return s;
}

// the strip of a plane that a coordinate falls in (0 outside)
int channel(int direction, double coord) {
    double const top = direction == 0 ? 801.2 : 972.7;
    int const n = direction == 0 ? 116 : 184;
    int const c = static_cast<int>(std::lround((top - coord) / kPitch)) + 1;
    return c >= 1 && c <= n ? c : 0;
}

struct Muon {
    double x, y, tx, ty;  // at z = 0
};

void fire(Muon const& m, std::vector<std::int32_t>& ids, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0, 1);
    for (int st = 1; st <= 5; ++st) {
        for (int dir = 0; dir < 2; ++dir) {
            double const z = kZ[st - 1] + 8 * dir;
            double const coord = dir == 0 ? m.y + m.ty * z : m.x + m.tx * z;
            int const c = channel(dir, coord);
            if (c == 0) {
                continue;
            }
            ids.push_back(st * 10000 + dir * 1000 + c);
            if (u(rng) < 0.3 && channel(dir, coord + kPitch) > 0) {
                ids.push_back(st * 10000 + dir * 1000 + c - 1);  // a neighbour
            }
        }
    }
}

}  // namespace

int main() {
    std::mt19937_64 rng{20180703};
    std::uniform_real_distribution<double> u(0, 1);
    mf::RpcConfig config;
    config.dead.clear();
    mf::Rpc const rpc{strips(), config};
    check(std::abs(rpc.z_station1() - (kZ[0] + 4)) < 1e-6,
          "z of station 1 midway between its planes");

    // single muons with noise: one RPC track per direction, close to the muon
    int n_ok = 0, n_events = 500;
    double worst = 0;
    int n_matched = 0, n_close = 0, n_cut = 0;
    for (int ev = 0; ev < n_events; ++ev) {
        // inside the strips at every station: |x| < 973, -413 < y < 801 mm
        Muon const m{-300 + 600 * u(rng), -100 + 500 * u(rng), -0.04 + 0.08 * u(rng),
                     -0.02 + 0.04 * u(rng)};
        std::vector<std::int32_t> ids;
        fire(m, ids, rng);
        for (int k = 0; k < 3; ++k) {  // noise
            int const st = 1 + static_cast<int>(5 * u(rng));
            ids.push_back(st * 10000 + 1000 + 1 + static_cast<int>(184 * u(rng)));
        }
        auto const fired = rpc.fired(ids);
        auto const tracks = rpc.tracks(rpc.clusters(fired));
        // the track nearest to the muon in each direction
        std::array<double, 2> nearest{1e9, 1e9};
        for (auto const& t : tracks) {
            double const z1 = rpc.z_station1();
            double const truth = t.direction == 1 ? m.x + m.tx * z1 : m.y + m.ty * z1;
            nearest[t.direction] = std::min(nearest[t.direction], std::abs(t.at(z1) - truth));
        }
        if (nearest[0] < 1e9 && nearest[1] < 1e9) {
            ++n_ok;
            worst = std::max({worst, nearest[0], nearest[1]});
            n_close += nearest[0] < kPitch;                 // y: no noise strips
            n_cut += nearest[0] < 100 && nearest[1] < 100;  // within the 2018 match cut
        }
        // a drift-tube track: the muon itself, from z = 7500 mm
        double const z = 7500;
        auto const match = rpc.match(m.x + m.tx * z, m.y + m.ty * z, z, m.tx, m.ty, fired, tracks);
        if (std::abs(match.dx1) < 100 && std::abs(match.dy1) < 100 && match.n_stations >= 3) {
            ++n_matched;
        }
    }
    std::printf(
        "single muons: %d of %d with RPC tracks in both directions; y tracks within a strip pitch: "
        "%d; "
        "both within 10 cm: %d (worst %.1f mm: in x, a noise strip within the 15 cm window joins "
        "the "
        "track, as in the legacy reconstruction); %d matched (|dx|, |dy| < 10 cm, >= 3 stations)\n",
        n_ok, n_events, n_close, n_cut, worst, n_matched);
    check(n_ok > 0.97 * n_events, "RPC tracks found");
    check(n_close > 0.97 * n_ok, "RPC tracks without noise within a strip pitch");
    check(n_cut > 0.97 * n_ok, "RPC tracks within the match cut");
    check(n_matched > 0.97 * n_events, "drift-tube tracks matched");

    // a track far from the muon is not matched
    {
        Muon const m{0, 0, 0, 0};
        std::vector<std::int32_t> ids;
        fire(m, ids, rng);
        auto const fired = rpc.fired(ids);
        auto const tracks = rpc.tracks(rpc.clusters(fired));
        auto const far = rpc.match(300, 300, 7500, 0, 0, fired, tracks);
        check(std::abs(far.dx1 - 300) < kPitch && std::abs(far.dy1 - 300) < kPitch &&
                  far.n_stations == 0,
              "a track 30 cm away in x and y is not matched");
        // a station counts with a close strip in either direction
        auto const half = rpc.match(300, 0, 7500, 0, 0, fired, tracks);
        check(half.n_stations == 5, "close in y only: every station counts");
    }

    // two muons 40 cm apart: two tracks per direction
    {
        std::vector<std::int32_t> ids;
        fire({-200, -100, 0.01, 0.0}, ids, rng);
        fire({200, 300, -0.01, 0.0}, ids, rng);
        auto const tracks = rpc.tracks(rpc.clusters(rpc.fired(ids)));
        int nx = 0, ny = 0;
        for (auto const& t : tracks) {
            (t.direction == 1 ? nx : ny) += 1;
        }
        check(nx == 2 && ny == 2, "two muons, two tracks per direction");
    }

    // wide clusters are dropped; fewer than 3 stations give no track
    {
        std::vector<std::int32_t> ids;
        for (int st = 1; st <= 5; ++st) {
            for (int c = 50; c < 58; ++c) {
                ids.push_back(st * 10000 + 1000 + c);  // 8 strips: wider than 6
            }
        }
        auto const fired = rpc.fired(ids);
        check(rpc.clusters(fired).empty(), "clusters wider than cluster_max dropped");
        std::vector<std::int32_t> two{11050, 21050};
        check(rpc.tracks(rpc.clusters(rpc.fired(two))).empty(), "two stations give no track");
    }

    // dead strips are ignored
    {
        mf::RpcConfig with_dead;  // the legacy list
        mf::Rpc const rpc_dead{strips(), with_dead};
        std::vector<std::int32_t> ids{10001, 10110, 11005, 11100};
        auto const fired = rpc_dead.fired(ids);
        check(fired.size() == 1 && fired[0].id == 11100, "legacy dead strips ignored");
        check(with_dead.dead.size() == 5 * (10 + 1 + 11) + 1 + 10, "legacy dead list size");
    }

    if (failures == 0) {
        std::puts("muflux_rpc: all checks passed");
    }
    return failures == 0 ? 0 : 1;
}
