// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// MufluxGeometry.hpp — drift-tube layers of the 2018 muon-flux spectrometer
//
// The 2018 muon-flux test beam (SPS H4) had four drift-tube stations: T1 (x
// and u views) and T2 (v and x views) in front of the Goliath magnet, T3 and
// T4 (x views) behind it; the stereo wires are tilted by 60 degrees. Every
// tube is identified by its legacy channel id
//   station*1e7 + view*1e6 + plane*1e5 + layer*1e4 + 2000 + tube,
// and all tubes with the same id / 1e4 form one layer: parallel wires at one
// z. The tubes are read from the wire table exported from the legacy geometry
// (dt_wires.txt: id, the two wire ends in cm, gas radius in cm).
//
// The x layers of T3 and T4 are built from modules of 12 tubes at slightly
// different z (a 2.5 mm spread in the 2018 geometry), so each tube keeps its
// own z; a layer's z is the mean. The two ends of a wire must be at one z.
//
// Each layer measures one coordinate, along meas_dir = (w_y, -w_x), the
// direction across its wires w in the transverse plane: x for the x views. For
// a point (x, y), coord = meas_dir . (x, y). (meas_dir, wire_dir, z) is
// right-handed, so a plane surface with these local axes faces downstream.
//
// Kept free of ACTS and phlex so that the pattern recognition built on it can
// be tested on its own.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SHiP::trout::muflux {

using Vec2 = std::array<double, 2>;
using Vec3 = std::array<double, 3>;

/// The two ends of a wire and the gas radius of its tube, in mm.
struct WireEnds {
    Vec3 end1{};
    Vec3 end2{};
    double radius = 0;
};

/// Read the wire table (lengths in cm, '#' starts a comment line); returns mm.
inline std::map<std::int32_t, WireEnds> read_wire_table(std::string const& path) {
    std::ifstream in{path};
    if (!in) {
        throw std::runtime_error("muflux: cannot open wire table '" + path + "'");
    }
    std::map<std::int32_t, WireEnds> wires;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        auto const first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        std::istringstream fields{line};
        std::int64_t id = 0;
        std::array<double, 7> v{};
        fields >> id;
        for (auto& x : v) {
            fields >> x;
        }
        if (!fields) {
            throw std::runtime_error("muflux: wire table '" + path + "', line " +
                                     std::to_string(line_number) + ": expected 8 numbers");
        }
        constexpr double cm = 10.0;
        WireEnds const w{.end1 = {v[0] * cm, v[1] * cm, v[2] * cm},
                         .end2 = {v[3] * cm, v[4] * cm, v[5] * cm},
                         .radius = v[6] * cm};
        if (!wires.emplace(static_cast<std::int32_t>(id), w).second) {
            throw std::runtime_error("muflux: wire table '" + path + "': duplicate id " +
                                     std::to_string(id));
        }
    }
    if (wires.empty()) {
        throw std::runtime_error("muflux: wire table '" + path + "' has no wires");
    }
    return wires;
}

/// One tube of a layer.
struct Tube {
    std::int32_t id = 0;
    double coord = 0;  ///< wire position along the layer's meas_dir [mm]
    double along = 0;  ///< wire midpoint along wire_dir [mm]
    double half_length = 0;
    double radius = 0;
    double z = 0;  ///< z of the wire [mm]
};

/// One layer: parallel wires, at one z or (T3, T4) a few close ones.
struct Layer {
    std::int32_t key = 0;  ///< channel id / 1e4: station*1000 + view*100 + plane*10 + layer
    int station = 0, view = 0, plane = 0, layer = 0;
    double z = 0;             ///< mm
    Vec2 wire_dir{};          ///< unit vector along the wires, wire_dir[1] >= 0
    Vec2 meas_dir{};          ///< unit vector across the wires: (wire_dir[1], -wire_dir[0])
    bool stereo = false;      ///< wires not along y
    Vec3 centre{};            ///< centre of the layer's wire area
    double half_width = 0;    ///< half extent along meas_dir, tube walls included [mm]
    double half_length = 0;   ///< half extent along wire_dir [mm]
    double radius = 0;        ///< gas radius of its tubes [mm]
    std::vector<Tube> tubes;  ///< ordered by coord

    /// Coordinate measured by this layer for a point (x, y).
    double coord(double x, double y) const { return meas_dir[0] * x + meas_dir[1] * y; }
};

/// The drift-tube layers, ordered by z.
class Geometry {
   public:
    /// z_spread_mm: largest z difference between the tubes of a layer;
    /// z_tolerance_mm: between the two ends of a wire.
    explicit Geometry(std::map<std::int32_t, WireEnds> const& wires, double z_spread_mm = 10.0,
                      double z_tolerance_mm = 0.5, double parallel_tolerance = 1e-3) {
        std::map<std::int32_t, std::vector<std::pair<std::int32_t, WireEnds>>> by_layer;
        for (auto const& [id, w] : wires) {
            by_layer[id / 10000].emplace_back(id, w);
        }
        for (auto const& [key, tubes] : by_layer) {
            Layer l;
            l.key = key;
            l.station = key / 1000;
            l.view = key / 100 % 10;
            l.plane = key / 10 % 10;
            l.layer = key % 10;
            // wire direction from the first tube, pointing to positive y
            auto const& w0 = tubes.front().second;
            Vec2 d{w0.end2[0] - w0.end1[0], w0.end2[1] - w0.end1[1]};
            double const dn = std::hypot(d[0], d[1]);
            if (!(dn > 0)) {
                throw std::runtime_error("muflux: tube " + std::to_string(tubes.front().first) +
                                         " has a zero-length wire");
            }
            d = {d[0] / dn, d[1] / dn};
            if (d[1] < 0 || (d[1] == 0 && d[0] < 0)) {
                d = {-d[0], -d[1]};
            }
            l.wire_dir = d;
            l.meas_dir = {d[1], -d[0]};
            l.stereo = std::abs(d[0]) > parallel_tolerance;
            l.radius = w0.radius;
            double zsum = 0;
            double min_coord = std::numeric_limits<double>::max();
            double max_coord = std::numeric_limits<double>::lowest();
            double along_sum = 0;
            for (auto const& [id, w] : tubes) {
                Vec2 e{w.end2[0] - w.end1[0], w.end2[1] - w.end1[1]};
                double const en = std::hypot(e[0], e[1]);
                if (!(en > 0) ||
                    std::abs(e[0] / en * d[1] - e[1] / en * d[0]) > parallel_tolerance) {
                    throw std::runtime_error("muflux: tube " + std::to_string(id) +
                                             " is not parallel to the other wires of layer " +
                                             std::to_string(key));
                }
                if (std::abs(w.end1[2] - w.end2[2]) > z_tolerance_mm) {
                    throw std::runtime_error("muflux: the wire of tube " + std::to_string(id) +
                                             " is tilted in z");
                }
                if (std::abs(w.end1[2] - w0.end1[2]) > z_spread_mm) {
                    throw std::runtime_error("muflux: tube " + std::to_string(id) +
                                             " is more than " + std::to_string(z_spread_mm) +
                                             " mm from the z of layer " + std::to_string(key));
                }
                double const mx = 0.5 * (w.end1[0] + w.end2[0]);
                double const my = 0.5 * (w.end1[1] + w.end2[1]);
                Tube t{.id = id,
                       .coord = l.coord(mx, my),
                       .along = d[0] * mx + d[1] * my,
                       .half_length = 0.5 * en,
                       .radius = w.radius,
                       .z = 0.5 * (w.end1[2] + w.end2[2])};
                zsum += 0.5 * (w.end1[2] + w.end2[2]);
                min_coord = std::min(min_coord, t.coord - t.radius);
                max_coord = std::max(max_coord, t.coord + t.radius);
                along_sum += t.along;
                l.half_length = std::max(l.half_length, t.half_length);
                l.tubes.push_back(t);
            }
            l.z = zsum / static_cast<double>(tubes.size());
            double const cm = 0.5 * (min_coord + max_coord);
            double const ca = along_sum / static_cast<double>(tubes.size());
            l.centre = {cm * l.meas_dir[0] + ca * l.wire_dir[0],
                        cm * l.meas_dir[1] + ca * l.wire_dir[1], l.z};
            l.half_width = 0.5 * (max_coord - min_coord);
            for (auto const& t : l.tubes) {
                l.half_length = std::max(l.half_length, std::abs(t.along - ca) + t.half_length);
            }
            std::ranges::sort(l.tubes, {}, &Tube::coord);
            layers_.push_back(std::move(l));
        }
        std::ranges::stable_sort(layers_, {}, &Layer::z);
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            for (std::size_t k = 0; k < layers_[i].tubes.size(); ++k) {
                index_.emplace(layers_[i].tubes[k].id, std::pair{static_cast<int>(i), k});
            }
        }
    }

    std::vector<Layer> const& layers() const { return layers_; }

    /// Index of the layer of a tube in layers(), or -1 for an unknown id.
    int layer_index(std::int32_t tube_id) const {
        auto const it = index_.find(tube_id);
        return it == index_.end() ? -1 : it->second.first;
    }

    /// The tube of a known id.
    Tube const& tube(std::int32_t tube_id) const {
        auto const it = index_.find(tube_id);
        if (it == index_.end()) {
            throw std::out_of_range("muflux: unknown tube id " + std::to_string(tube_id));
        }
        return layers_[it->second.first].tubes[it->second.second];
    }

   private:
    std::vector<Layer> layers_;
    std::map<std::int32_t, std::pair<int, std::size_t>> index_;
};

}  // namespace SHiP::trout::muflux
