// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// muflux_toy.hpp — toy events for the tests of the 2018 muon-flux tracking
//
// Muons are propagated with a Runge-Kutta integrator through a Goliath-like
// dipole (By Gaussian in z with the measured peak and integral), or through the
// measured Goliath map read from its text file (x y z Bx By Bz in mm and T, as
// FairShip's field/GoliathFieldMap.txt), and digitised as in Shannon's
// muflux_digitise (|Gauss(DOCA, 0.27 mm)|), with 3% inefficiency, noise hits
// and one digit per tube. The layout follows the 2018 one (T1 x-u, T2 v-x, T3
// and T4 x; 60 degree stereo), with invented T3/T4 tube positions.

#pragma once

#include "detectors/muflux/MufluxGeometry.hpp"
#include "detectors/muflux/MufluxPatrec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace muflux_toy {

namespace mf = SHiP::trout::muflux;

constexpr double kRadius = 18.15;     // mm
constexpr double kPitch = 42.0;       // mm
constexpr double kZc = 3507.5;        // mm, Goliath centre
constexpr double kB0 = -1.4733;       // T
constexpr double kIntegral = -3540.;  // T mm (core of the measured map)

// One view: 4 layers (2 planes x 2 layers) of `n` tubes with wire direction w.
void add_view(std::map<std::int32_t, mf::WireEnds>& wires, int station, int view,
              std::array<double, 4> const& zs, std::array<double, 4> const& offsets, int n,
              double first_coord, mf::Vec2 w, double half_length, double along_centre,
              std::array<double, 4> const& module_dz = {0, 0, 0, 0}, double module_gap = 0) {
    mf::Vec2 const m{w[1], -w[0]};
    for (int k = 0; k < 4; ++k) {
        int const plane = k / 2, layer = k % 2;
        for (int t = 0; t < n; ++t) {
            int const module = std::min(t / 12, 3);
            double const c = first_coord + offsets[k] + kPitch * t + module_gap * module;
            double const z = zs[k] + module_dz[module];
            mf::Vec2 const mid{c * m[0] + along_centre * w[0], c * m[1] + along_centre * w[1]};
            std::int32_t const id =
                station * 10000000 + view * 1000000 + plane * 100000 + layer * 10000 + 2001 + t;
            wires[id] = {.end1 = {mid[0] - half_length * w[0], mid[1] - half_length * w[1], z},
                         .end2 = {mid[0] + half_length * w[0], mid[1] + half_length * w[1], z},
                         .radius = kRadius};
        }
    }
}

std::map<std::int32_t, mf::WireEnds> toy_wires() {
    std::map<std::int32_t, mf::WireEnds> w;
    std::array<double, 4> const stag{0, 20.9, 9.9, -11.1};
    double const s60 = std::sin(60.2 * M_PI / 180), c60 = std::cos(60.2 * M_PI / 180);
    add_view(w, 1, 0, {193.2, 229.6, 270.2, 306.6}, stag, 12, -217.7, {0, 1}, 549, -20);
    add_view(w, 1, 1, {505.4, 541.8, 582.4, 618.8}, stag, 12, -231, {-s60, c60}, 549, -20);
    add_view(w, 2, 0, {880.1, 916.5, 957.1, 993.5}, stag, 12, -231, {s60, c60}, 549, -20);
    add_view(w, 2, 1, {1120., 1156.4, 1197., 1233.4}, stag, 12, -217.7, {0, 1}, 549, -20);
    add_view(w, 3, 0, {5800., 5836.4, 5877., 5913.4}, stag, 48, -987, {0, 1}, 800, 0,
             {0, -1.975, -2.5, -0.6}, 1.185);
    add_view(w, 4, 0, {7430., 7466.4, 7507., 7543.4}, stag, 48, -987, {0, 1}, 800, 0,
             {0, -2.35, -1.2, -0.4}, 1.185);
    return w;
}

using Field = std::function<std::array<double, 3>(double, double, double)>;  // T

Field gaussian_dipole() {
    double const s = std::abs(kIntegral / kB0) / std::sqrt(2 * M_PI);
    return [s](double x, double y, double z) -> std::array<double, 3> {
        if (std::abs(x) > 1200 || std::abs(y) > 500) {
            return {0, 0, 0};
        }
        double const u = (z - kZc) / s;
        return {0, kB0 * std::exp(-0.5 * u * u), 0};
    };
}

// The measured map (local frame, origin at the Goliath centre), trilinear, 0 outside.
Field measured_map(std::string const& path) {
    std::ifstream in{path};
    std::vector<std::array<double, 6>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream s{line};
        std::array<double, 6> r{};
        for (auto& v : r) {
            s >> v;
        }
        if (s) {
            rows.push_back(r);
        }
    }
    struct Grid {
        std::array<double, 3> lo{}, step{};
        std::array<int, 3> n{};
        std::vector<std::array<double, 3>> b;
    };
    auto g = std::make_shared<Grid>();
    for (int a = 0; a < 3; ++a) {
        std::vector<double> v;
        for (auto const& r : rows) {
            v.push_back(r[a]);
        }
        std::ranges::sort(v);
        v.erase(std::unique(v.begin(), v.end()), v.end());
        g->lo[a] = v.front();
        g->n[a] = static_cast<int>(v.size());
        g->step[a] = (v.back() - v.front()) / (v.size() - 1);
    }
    g->b.resize(rows.size());
    for (auto const& r : rows) {
        int i[3];
        for (int a = 0; a < 3; ++a) {
            i[a] = static_cast<int>(std::lround((r[a] - g->lo[a]) / g->step[a]));
        }
        g->b[(i[0] * g->n[1] + i[1]) * g->n[2] + i[2]] = {r[3], r[4], r[5]};
    }
    std::printf("measured map: %zu nodes, %d x %d x %d\n", rows.size(), g->n[0], g->n[1], g->n[2]);
    return [g](double x, double y, double z) -> std::array<double, 3> {
        double const p[3] = {x, y, z - kZc};
        int i0[3];
        double f[3];
        for (int a = 0; a < 3; ++a) {
            double const u = (p[a] - g->lo[a]) / g->step[a];
            if (u < 0 || u > g->n[a] - 1) {
                return {0, 0, 0};
            }
            i0[a] = std::min(static_cast<int>(u), g->n[a] - 2);
            f[a] = u - i0[a];
        }
        std::array<double, 3> out{0, 0, 0};
        for (int c = 0; c < 8; ++c) {
            int const dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
            double const w =
                (dx ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dz ? f[2] : 1 - f[2]);
            auto const& b = g->b[((i0[0] + dx) * g->n[1] + i0[1] + dy) * g->n[2] + i0[2] + dz];
            for (int k = 0; k < 3; ++k) {
                out[k] += w * b[k];
            }
        }
        return out;
    };
}

struct State {
    double x, y, tx, ty;
};

// Runge-Kutta 4 in z; qop in 1/GeV, lengths in mm.
State propagate(State s, double z0, double z1, double qop, Field const& field) {
    constexpr double k = 0.299792458e-3;  // GeV / (T mm)
    auto deriv = [&](double z, State const& u) {
        auto const b = field(u.x, u.y, z);
        double const n = std::sqrt(1 + u.tx * u.tx + u.ty * u.ty);
        double const dtx =
            qop * k * n * (u.tx * u.ty * b[0] - (1 + u.tx * u.tx) * b[1] + u.ty * b[2]);
        double const dty =
            qop * k * n * ((1 + u.ty * u.ty) * b[0] - u.tx * u.ty * b[1] - u.tx * b[2]);
        return State{u.tx, u.ty, dtx, dty};
    };
    int const steps = std::max(1, static_cast<int>(std::ceil(std::abs(z1 - z0) / 10.)));
    double const h = (z1 - z0) / steps;
    double z = z0;
    for (int i = 0; i < steps; ++i) {
        auto add = [](State const& a, State const& d, double f) {
            return State{a.x + f * d.x, a.y + f * d.y, a.tx + f * d.tx, a.ty + f * d.ty};
        };
        auto const k1 = deriv(z, s);
        auto const k2 = deriv(z + h / 2, add(s, k1, h / 2));
        auto const k3 = deriv(z + h / 2, add(s, k2, h / 2));
        auto const k4 = deriv(z + h, add(s, k3, h));
        s = {s.x + h / 6 * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
             s.y + h / 6 * (k1.y + 2 * k2.y + 2 * k3.y + k4.y),
             s.tx + h / 6 * (k1.tx + 2 * k2.tx + 2 * k3.tx + k4.tx),
             s.ty + h / 6 * (k1.ty + 2 * k2.ty + 2 * k3.ty + k4.ty)};
        z += h;
    }
    return s;
}

struct TrueHit {
    mf::Hit hit;
    int muon;  ///< -1 for noise
    int sign;
};

struct Muon {
    double p;  ///< GeV
    int q;
    State first;                         ///< at the first layer
    std::array<int, 3> counts{0, 0, 0};  ///< digits: x in T1/T2, stereo, T3/T4
};

struct Event {
    std::vector<mf::Hit> hits;
    std::vector<TrueHit> truth;  ///< parallel to hits
    std::vector<Muon> muons;
};

/// The kick (q * delta px, GeV) and its z, from By on the beam axis.
inline std::pair<double, double> kick_of(Field const& field) {
    double integral = 0, moment = 0;
    for (double z = kZc - 3000; z < kZc + 3000; z += 1) {
        double const by = field(0, 0, z)[1];
        integral += by;
        moment += by * z;
    }
    return {-0.299792458e-3 * integral, moment / integral};
}

class Generator {
   public:
    Generator(mf::Geometry const& geo, Field field, std::uint64_t seed, double two_muon = 0.2)
        : geo_{geo}, field_{std::move(field)}, rng_{seed}, two_muon_{two_muon} {}

    Event next() {
        Event ev;
        int const n_mu = 1 + (u01_(rng_) < two_muon_);
        std::vector<TrueHit> truth;
        for (int m = 0; m < n_mu; ++m) {
            Muon mu;
            mu.p = 5 * std::pow(60., u01_(rng_));  // 5 - 300 GeV, log-uniform
            mu.q = u01_(rng_) < 0.5 ? -1 : 1;
            State s{-80 + 160 * u01_(rng_), -80 + 160 * u01_(rng_), -0.03 + 0.06 * u01_(rng_),
                    -0.03 + 0.06 * u01_(rng_)};
            double z = 0;
            for (int li = 0; li < static_cast<int>(geo_.layers().size()); ++li) {
                auto const& l = geo_.layers()[li];
                s = propagate(s, z, l.z, mu.q / mu.p, field_);
                z = l.z;
                if (li == 0) {
                    mu.first = s;
                }
                double const c0 = l.coord(s.x, s.y);
                auto const it = std::ranges::min_element(
                    l.tubes, {}, [&](mf::Tube const& t) { return std::abs(t.coord - c0); });
                // the crossing at the tube's own z (within a few mm of the layer's)
                double const xt = s.x + s.tx * (it->z - l.z), yt = s.y + s.ty * (it->z - l.z);
                double const c = l.coord(xt, yt);
                double const along = l.wire_dir[0] * xt + l.wire_dir[1] * yt;
                if (std::abs(along - it->along) > it->half_length) {
                    continue;
                }
                double const tm = l.meas_dir[0] * s.tx + l.meas_dir[1] * s.ty;
                double const doca = std::abs(c - it->coord) / std::sqrt(1 + tm * tm);
                if (doca > kRadius || u01_(rng_) < 0.03) {
                    continue;
                }
                double const drift = std::abs(doca + 0.27 * gauss_(rng_));
                truth.push_back({.hit = {.layer = li,
                                         .tube = it->id,
                                         .wire = it->coord,
                                         .drift = drift,
                                         .z = it->z},
                                 .muon = m,
                                 .sign = c >= it->coord ? 1 : -1});
            }
            ev.muons.push_back(mu);
        }
        for (int k = noise_(rng_); k > 0; --k) {
            int const li = static_cast<int>(u01_(rng_) * geo_.layers().size());
            auto const& l = geo_.layers()[li];
            auto const& t = l.tubes[static_cast<std::size_t>(u01_(rng_) * l.tubes.size())];
            truth.push_back({.hit = {.layer = li,
                                     .tube = t.id,
                                     .wire = t.coord,
                                     .drift = kRadius * u01_(rng_),
                                     .z = t.z},
                             .muon = -1,
                             .sign = 0});
        }
        // one digit per tube, as in the digitiser: keep the first
        std::set<std::int32_t> tubes;
        for (auto& t : truth) {
            if (tubes.insert(t.hit.tube).second) {
                t.hit.digit = ev.hits.size();
                ev.hits.push_back(t.hit);
                ev.truth.push_back(t);
                if (t.muon >= 0) {
                    auto const& l = geo_.layers()[t.hit.layer];
                    (l.station >= 3 ? ev.muons[t.muon].counts[2]
                     : l.stereo     ? ev.muons[t.muon].counts[1]
                                    : ev.muons[t.muon].counts[0]) += 1;
                }
            }
        }
        return ev;
    }

   private:
    mf::Geometry const& geo_;
    Field field_;
    std::mt19937_64 rng_;
    double two_muon_;
    std::uniform_real_distribution<double> u01_{0, 1};
    std::normal_distribution<double> gauss_{0, 1};
    std::poisson_distribution<int> noise_{2.0};
};

/// Index of the candidate with most hits of muon `m` (at least `min_frac` of its hits), or -1.
inline int best_candidate(Event const& ev, std::vector<mf::Candidate> const& cands, int m,
                          double min_frac = 0.7) {
    int best = -1;
    double best_f = min_frac;
    for (std::size_t c = 0; c < cands.size(); ++c) {
        if (!cands[c].matched) {
            continue;
        }
        int mine = 0;
        for (auto const& a : cands[c].hits) {
            mine += ev.truth[a.hit].muon == m;
        }
        double const f = static_cast<double>(mine) / cands[c].hits.size();
        if (f >= best_f) {
            best_f = f;
            best = static_cast<int>(c);
        }
    }
    return best;
}

}  // namespace muflux_toy
