// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// MufluxRpc.hpp — RPC muon tagger of the 2018 muon-flux spectrometer
//
// Five RPC stations behind T4, each with horizontal strips (direction 0,
// measuring y) and vertical strips (direction 1, measuring x); strip ids
// station*1e4 + direction*1e3 + strip, read from a table exported from the
// geometry (rpc_strips.txt: id station direction x y z half_x half_y half_z, mm).
//
// RPC tracks, as the legacy reconstruction (muonTaggerClustering in
// drifttubeMonitoring.py) built them for the 2018 analysis, per direction:
// strips grouped into clusters where neighbouring channel numbers differ by at
// most cluster_gap, clusters wider than cluster_max strips dropped; the
// clusters' mean positions of all stations grouped where neighbours differ by
// at most track_window; a group with clusters in at least min_planes stations
// gives a straight line, coord = slope * z + intercept (least squares). The
// strips in `dead` are ignored: the legacy list of strips that were inactive in
// 2018 but fire in the simulation.
//
// A drift-tube track, extrapolated straight from its last state, is matched
// with the nearest RPC track per direction at the z of station 1 (midway
// between its two strip planes), and its distance to the nearest fired strip
// is measured in every station (either direction): the 2018 selection asks for
// |dx|, |dy| < 10 cm at station 1 and strips within 10 cm in >= 3 stations.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SHiP::trout::muflux {

struct RpcStrip {
    std::int32_t id = 0;
    int station = 0;             ///< 1-5
    int direction = 0;           ///< 0: horizontal (measures y), 1: vertical (measures x)
    double x = 0, y = 0, z = 0;  ///< centre [mm]
    /// the coordinate the strip measures
    [[nodiscard]] double coord() const { return direction == 0 ? y : x; }
};

/// Strip table: one line per strip, `id station direction x y z ...` (mm), '#'
/// comments.
inline std::map<std::int32_t, RpcStrip> read_rpc_table(std::string const& path) {
    std::ifstream in{path};
    if (!in) {
        throw std::runtime_error("muflux: cannot open RPC strip table '" + path + "'");
    }
    std::map<std::int32_t, RpcStrip> strips;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        auto const first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        std::istringstream fields{line};
        RpcStrip s;
        fields >> s.id >> s.station >> s.direction >> s.x >> s.y >> s.z;
        if (!fields || s.station < 1 || s.station > 5 || (s.direction != 0 && s.direction != 1)) {
            throw std::runtime_error("muflux: RPC strip table '" + path + "', line " +
                                     std::to_string(line_number) + ": bad strip");
        }
        if (!strips.emplace(s.id, s).second) {
            throw std::runtime_error("muflux: RPC strip table '" + path + "', line " +
                                     std::to_string(line_number) + ": duplicate id");
        }
    }
    if (strips.empty()) {
        throw std::runtime_error("muflux: RPC strip table '" + path + "' has no strips");
    }
    return strips;
}

/// The strips the legacy reconstruction ignored in the simulation
/// (deadChannelsRPC4MC in drifttubeMonitoring.py): inactive in 2018.
inline std::set<std::int32_t> legacy_dead_rpc_strips() {
    std::set<std::int32_t> dead;
    for (int s = 1; s <= 5; ++s) {
        for (int y = 107; y <= 116; ++y) {
            dead.insert(s * 10000 + y);
        }
        dead.insert(s * 10000 + 1);
        for (int x = 1; x <= 11; ++x) {
            dead.insert(s * 10000 + 1000 + x);
        }
    }
    dead.insert(11184);
    for (int c : {31093, 31094, 31095, 31096, 11029, 11030, 51141, 51142, 20057, 20058}) {
        dead.insert(c);
    }
    return dead;
}

struct RpcConfig {
    int cluster_gap = 3;        ///< channels: neighbours in a cluster differ by at most this
    int cluster_max = 6;        ///< strips: wider clusters are dropped
    double track_window = 150;  ///< mm: neighbouring cluster positions in a track
    int min_planes = 3;         ///< stations with a cluster in a track
    double station_dist = 100;  ///< mm: a strip this close counts its station (n_stations)
    std::set<std::int32_t> dead = legacy_dead_rpc_strips();
};

struct RpcCluster {
    int station = 0, direction = 0, size = 0;
    double coord = 0, z = 0;  ///< mm
};

/// coord = slope * z + intercept (mm)
struct RpcTrack {
    int direction = 0;
    double slope = 0, intercept = 0;
    int n_clusters = 0;
    [[nodiscard]] double at(double z) const { return slope * z + intercept; }
};

/// RPC information of one drift-tube track.
struct RpcMatch {
    double x1 = std::nan(""), y1 = std::nan("");    ///< track at station 1 [mm]
    double dx1 = std::nan(""), dy1 = std::nan("");  ///< minus the nearest RPC track [mm]
    std::array<double, 5> station_dist{};  ///< nearest fired strip per station [mm], NaN if none
    int n_stations = 0;                    ///< stations with a strip within station_dist
};

class Rpc {
   public:
    Rpc(std::map<std::int32_t, RpcStrip> strips, RpcConfig config)
        : strips_{std::move(strips)}, config_{std::move(config)} {
        std::array<double, 2> zs{0, 0};
        std::array<int, 2> n{0, 0};
        for (auto const& [id, s] : strips_) {
            if (s.station == 1) {
                zs[s.direction] += s.z;
                ++n[s.direction];
            }
        }
        if (n[0] == 0 || n[1] == 0) {
            throw std::runtime_error("muflux: RPC strip table without both planes of station 1");
        }
        z1_ = 0.5 * (zs[0] / n[0] + zs[1] / n[1]);
    }

    [[nodiscard]] double z_station1() const { return z1_; }
    [[nodiscard]] RpcConfig const& config() const { return config_; }

    /// The fired strips that are in the table and not dead.
    [[nodiscard]] std::vector<RpcStrip> fired(std::vector<std::int32_t> const& ids) const {
        std::vector<RpcStrip> out;
        for (auto id : ids) {
            if (config_.dead.contains(id)) {
                continue;
            }
            if (auto const it = strips_.find(id); it != strips_.end()) {
                out.push_back(it->second);
            }
        }
        return out;
    }

    [[nodiscard]] std::vector<RpcCluster> clusters(std::vector<RpcStrip> const& fired) const {
        std::map<int, std::vector<RpcStrip>> by_plane;  // 10 * station + direction
        for (auto const& s : fired) {
            by_plane[10 * s.station + s.direction].push_back(s);
        }
        std::vector<RpcCluster> out;
        for (auto& [plane, list] : by_plane) {
            std::ranges::sort(list, {}, [](RpcStrip const& s) { return s.id; });
            auto flush = [&](std::size_t b, std::size_t e) {
                if (e - b > static_cast<std::size_t>(config_.cluster_max)) {
                    return;
                }
                RpcCluster c{.station = plane / 10,
                             .direction = plane % 10,
                             .size = static_cast<int>(e - b)};
                for (std::size_t i = b; i < e; ++i) {
                    c.coord += list[i].coord() / static_cast<double>(e - b);
                    c.z += list[i].z / static_cast<double>(e - b);
                }
                out.push_back(c);
            };
            std::size_t begin = 0;
            for (std::size_t i = 1; i <= list.size(); ++i) {
                if (i == list.size() || list[i].id - list[i - 1].id > config_.cluster_gap) {
                    flush(begin, i);
                    begin = i;
                }
            }
        }
        return out;
    }

    [[nodiscard]] std::vector<RpcTrack> tracks(std::vector<RpcCluster> const& clusters) const {
        std::vector<RpcTrack> out;
        for (int dir = 0; dir < 2; ++dir) {
            std::vector<RpcCluster> list;
            for (auto const& c : clusters) {
                if (c.direction == dir) {
                    list.push_back(c);
                }
            }
            std::ranges::sort(list, {}, [](RpcCluster const& c) { return c.coord; });
            auto fit = [&](std::size_t b, std::size_t e) {
                std::set<int> stations;
                for (std::size_t i = b; i < e; ++i) {
                    stations.insert(list[i].station);
                }
                if (static_cast<int>(stations.size()) < config_.min_planes) {
                    return;
                }
                double sz = 0, sc = 0, szz = 0, szc = 0;
                double const n = static_cast<double>(e - b);
                for (std::size_t i = b; i < e; ++i) {
                    sz += list[i].z, sc += list[i].coord;
                    szz += list[i].z * list[i].z, szc += list[i].z * list[i].coord;
                }
                double const det = n * szz - sz * sz;
                if (!(std::abs(det) > 0)) {
                    return;
                }
                RpcTrack t{.direction = dir, .n_clusters = static_cast<int>(e - b)};
                t.slope = (n * szc - sz * sc) / det;
                t.intercept = (sc - t.slope * sz) / n;
                out.push_back(t);
            };
            std::size_t begin = 0;
            for (std::size_t i = 1; i <= list.size(); ++i) {
                if (i == list.size() || list[i].coord - list[i - 1].coord > config_.track_window) {
                    fit(begin, i);
                    begin = i;
                }
            }
        }
        return out;
    }

    /// Match a drift-tube track, straight from (x, y, z) with slopes (tx, ty).
    [[nodiscard]] RpcMatch match(double x, double y, double z, double tx, double ty,
                                 std::vector<RpcStrip> const& fired,
                                 std::vector<RpcTrack> const& tracks) const {
        RpcMatch m;
        m.station_dist.fill(std::nan(""));
        if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(tx) && std::isfinite(ty))) {
            return m;
        }
        m.x1 = x + tx * (z1_ - z);
        m.y1 = y + ty * (z1_ - z);
        for (auto const& t : tracks) {
            double const r = (t.direction == 1 ? m.x1 : m.y1) - t.at(z1_);
            double& d = t.direction == 1 ? m.dx1 : m.dy1;
            if (std::isnan(d) || std::abs(r) < std::abs(d)) {
                d = r;
            }
        }
        for (auto const& s : fired) {
            double const at = s.direction == 1 ? x + tx * (s.z - z) : y + ty * (s.z - z);
            double const r = std::abs(at - s.coord());
            double& d = m.station_dist[s.station - 1];
            if (std::isnan(d) || r < d) {
                d = r;
            }
        }
        for (double d : m.station_dist) {
            m.n_stations += d < config_.station_dist ? 1 : 0;
        }
        return m;
    }

   private:
    std::map<std::int32_t, RpcStrip> strips_;
    RpcConfig config_;
    double z1_ = 0;
};

}  // namespace SHiP::trout::muflux
