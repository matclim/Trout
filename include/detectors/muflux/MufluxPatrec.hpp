// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// MufluxPatrec.hpp — pattern recognition for the 2018 muon-flux spectrometer
//
// Follows the strategy of the legacy reconstruction of this setup: straight
// tracklets in front of the Goliath magnet (T1, T2: x views in x-z, then the
// u and v views for y) and behind it (T3, T4: x views only), matched at the
// magnet's kick plane. The bend between the two tracklets gives the charge and
// a first momentum, p = |kick / delta(ux)|, with kick = q * delta px across
// the magnet (0.2998 * the field integral) and ux = px / p.
//
// A drift-tube hit measures |distance of closest approach| to its wire. For a
// track with slope t_m = meas_dir . (tx, ty) across the wires, the track
// crosses the layer plane at
//   coord = wire + sign * drift * sqrt(1 + t_m^2),
// and the sign (left or right of the wire) is chosen by the fits: starting
// from the wire positions, each pass takes the sign closest to the current
// line and refits, until no sign changes.
//
// The resulting candidates (hits, signs, tracklet parameters, charge and
// momentum) seed the Kalman fit (MufluxKalman.hpp). Kept free of ACTS and
// phlex so that it can be tested on its own.

#pragma once

#include "detectors/muflux/MufluxGeometry.hpp"

#include <tuple>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SHiP::trout::muflux {

/// One drift-tube digit, placed on its layer.
struct Hit {
    int layer = -1;  ///< index in Geometry::layers()
    std::int32_t tube = 0;
    double wire = 0;        ///< wire coordinate along the layer's meas_dir [mm]
    double drift = 0;       ///< drift distance [mm]
    double z = 0;           ///< z of its wire [mm]
    std::size_t digit = 0;  ///< index of the digit in the event
};

/// A hit used by a track, with its side of the wire (+1 or -1).
struct Assigned {
    std::size_t hit = 0;  ///< index in the hit list
    int sign = 1;
};

struct PatrecConfig {
    double sigma = 0.27;       ///< hit resolution used in the fits [mm]
    double window = 5.0;       ///< beyond the tube radius when collecting hits [mm]
    int min_x_up = 5;          ///< hits in the x layers of T1 and T2
    int min_stereo = 5;        ///< hits in the u and v layers
    int min_x_down = 5;        ///< hits in the layers of T3 and T4
    double max_chi2_ndf = 25;  ///< per tracklet
    double max_slope = 0.3;    ///< |tx|, |ty|
    double match_window = 30;  ///< |x difference| of the two tracklets at kick_z [mm]
    double kick_gev = 1.06;    ///< q * delta px across the magnet [GeV]
    double kick_z = 3507.5;    ///< z of the kick plane [mm]
};

/// A straight line in front of the magnet, x = p0 + p1 dz, y = p2 + p3 dz with
/// dz = z - z_ref, and behind it, x = d0 + d1 (z - z_ref_down), y as in front.
struct Candidate {
    double z_ref = 0;
    std::array<double, 4> up{};  ///< x0, tx, y0, ty
    double z_ref_down = 0;
    std::array<double, 2> down{};  ///< x0, tx
    bool matched = false;          ///< has a tracklet behind the magnet
    int charge = 0;                ///< 0 if not matched
    double p = 0;                  ///< GeV, 0 if not matched
    double match_dx = 0;           ///< mm, at kick_z
    std::vector<Assigned> hits;    ///< in front of the magnet first, then behind
    int n_x_up = 0, n_stereo = 0, n_down = 0;
    double chi2_up = 0, chi2_down = 0;
    int ndf_up = 0, ndf_down = 0;

    /// Track position (x, y) at z, from the tracklet in front of or behind the magnet.
    Vec2 position(double z, bool behind) const {
        double const y = up[2] + up[3] * (z - z_ref);
        if (behind && matched) {
            return {down[0] + down[1] * (z - z_ref_down), y};
        }
        return {up[0] + up[1] * (z - z_ref), y};
    }
    /// Slopes (tx, ty) in front of or behind the magnet.
    Vec2 slopes(bool behind) const { return {behind && matched ? down[1] : up[1], up[3]}; }
};

namespace detail {

/// Solve the n x n system a x = b (n <= 4) in place; false if singular.
inline bool solve(int n, std::array<std::array<double, 4>, 4>& a, std::array<double, 4>& b) {
    for (int c = 0; c < n; ++c) {
        int piv = c;
        for (int r = c + 1; r < n; ++r) {
            if (std::abs(a[r][c]) > std::abs(a[piv][c])) {
                piv = r;
            }
        }
        if (std::abs(a[piv][c]) < 1e-300) {
            return false;
        }
        std::swap(a[c], a[piv]);
        std::swap(b[c], b[piv]);
        for (int r = c + 1; r < n; ++r) {
            double const f = a[r][c] / a[c][c];
            for (int k = c; k < n; ++k) {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    for (int c = n - 1; c >= 0; --c) {
        for (int k = c + 1; k < n; ++k) {
            b[c] -= a[c][k] * b[k];
        }
        b[c] /= a[c][c];
    }
    return true;
}

/// The model of one hit: predicted coordinate = row . params + offset.
struct Row {
    std::array<double, 4> a{};
    double offset = 0;
};

/// Result of a fit with left/right resolution.
struct Fit {
    bool ok = false;
    std::array<double, 4> params{};
    std::vector<int> signs;
    double chi2 = 0;
    int ndf = 0;
};

}  // namespace detail

class Patrec {
   public:
    Patrec(Geometry const& geometry, PatrecConfig config) : geometry_{geometry}, config_{config} {
        auto const& layers = geometry_.layers();
        for (int i = 0; i < static_cast<int>(layers.size()); ++i) {
            auto const& l = layers[i];
            if (l.station <= 2) {
                (l.stereo ? stereo_up_ : x_up_).push_back(i);
            } else {
                x_down_.push_back(i);
            }
        }
        if (x_up_.empty() || stereo_up_.empty() || x_down_.empty()) {
            throw std::runtime_error(
                "muflux patrec: need x and stereo layers in T1/T2 and layers in T3/T4");
        }
        z_ref_up_ = layers[std::min(x_up_.front(), stereo_up_.front())].z;
        z_ref_down_ = layers[x_down_.front()].z;
    }

    PatrecConfig const& config() const { return config_; }

    /// Find the track candidates of one event, best first.
    std::vector<Candidate> find(std::vector<Hit> const& hits) const {
        std::map<int, std::vector<std::size_t>> by_layer;
        for (std::size_t i = 0; i < hits.size(); ++i) {
            by_layer[hits[i].layer].push_back(i);
        }

        std::vector<bool> used(hits.size(), false);
        auto const up_x = find_x(hits, by_layer, used, x_up_, 1, 2, z_ref_up_, config_.min_x_up);
        auto const down_x =
            find_x(hits, by_layer, used, x_down_, 3, 4, z_ref_down_, config_.min_x_down);

        std::vector<Candidate> candidates;
        for (auto const& xt : up_x) {
            Candidate c;
            c.z_ref = z_ref_up_;
            c.z_ref_down = z_ref_down_;
            if (add_stereo(hits, by_layer, used, xt, c)) {
                candidates.push_back(std::move(c));
            }
        }
        match(down_x, candidates);
        return candidates;
    }

    /// Choose the signs of a candidate's hits from given positions and slopes
    /// per hit (e.g. smoothed Kalman states); returns the number of changes.
    template <typename PredictFn>
    int reassign_signs(std::vector<Hit> const& hits, Candidate& c, PredictFn predict) const {
        int changed = 0;
        for (auto& a : c.hits) {
            auto const& h = hits[a.hit];
            auto const& l = geometry_.layers()[h.layer];
            auto const [pos, slopes] = predict(a);
            double const pred = l.coord(pos[0], pos[1]);
            double const tm = l.meas_dir[0] * slopes[0] + l.meas_dir[1] * slopes[1];
            int const s = sign_for(h, pred, tm);
            if (s != a.sign) {
                a.sign = s;
                ++changed;
            }
        }
        return changed;
    }

    /// Coordinate of a hit along its layer's meas_dir for a side and a track
    /// slope t_m across the wires.
    static double measured(Hit const& h, int sign, double tm) {
        return h.wire + sign * h.drift * std::sqrt(1 + tm * tm);
    }

   private:
    struct XTracklet {
        std::array<double, 2> params{};  // x0, tx at z_ref
        std::vector<Assigned> hits;
        double chi2 = 0;
        int ndf = 0;
    };

    static int sign_for(Hit const& h, double pred, double tm) {
        double const plus = measured(h, +1, tm);
        double const minus = measured(h, -1, tm);
        return std::abs(plus - pred) <= std::abs(minus - pred) ? +1 : -1;
    }

    /// Fit `n` parameters to the hits, given the model rows and the track
    /// slope across the wires of each hit as a function of the parameters,
    /// resolving the left/right ambiguity iteratively. Drops the worst hit
    /// while chi2/ndf exceeds the cut and more than `min_hits` remain.
    template <typename RowFn, typename SlopeFn>
    detail::Fit fit_lr(std::vector<Hit> const& hits, std::vector<std::size_t>& sel, int n,
                       RowFn row_of, SlopeFn slope_of, int min_hits) const {
        detail::Fit fit;
        while (static_cast<int>(sel.size()) >= std::max(min_hits, n + 1)) {
            fit = fit_lr_once(hits, sel, n, row_of, slope_of);
            if (!fit.ok) {
                return fit;
            }
            if (fit.chi2 <= config_.max_chi2_ndf * fit.ndf ||
                static_cast<int>(sel.size()) <= std::max(min_hits, n + 1)) {
                break;
            }
            // drop the hit with the largest residual and try again
            std::size_t worst = 0;
            double worst_r = -1;
            for (std::size_t k = 0; k < sel.size(); ++k) {
                auto const& h = hits[sel[k]];
                auto const r = row_of(h);
                double pred = r.offset;
                for (int j = 0; j < n; ++j) {
                    pred += r.a[j] * fit.params[j];
                }
                double const res =
                    std::abs(measured(h, fit.signs[k], slope_of(h, fit.params)) - pred);
                if (res > worst_r) {
                    worst_r = res;
                    worst = k;
                }
            }
            sel.erase(sel.begin() + static_cast<std::ptrdiff_t>(worst));
        }
        fit.ok = fit.ok && static_cast<int>(sel.size()) >= min_hits &&
                 fit.chi2 <= config_.max_chi2_ndf * std::max(fit.ndf, 1);
        return fit;
    }

    template <typename RowFn, typename SlopeFn>
    detail::Fit fit_lr_once(std::vector<Hit> const& hits, std::vector<std::size_t> const& sel,
                            int n, RowFn row_of, SlopeFn slope_of) const {
        detail::Fit fit;
        fit.signs.assign(sel.size(), 0);  // 0: the wire position, for the first pass
        for (int pass = 0; pass < 8; ++pass) {
            std::array<std::array<double, 4>, 4> a{};
            std::array<double, 4> b{};
            for (std::size_t k = 0; k < sel.size(); ++k) {
                auto const& h = hits[sel[k]];
                auto const r = row_of(h);
                double const tm = pass == 0 ? 0. : slope_of(h, fit.params);
                double const y =
                    (fit.signs[k] == 0 ? h.wire : measured(h, fit.signs[k], tm)) - r.offset;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        a[i][j] += r.a[i] * r.a[j];
                    }
                    b[i] += r.a[i] * y;
                }
            }
            if (!detail::solve(n, a, b)) {
                fit.ok = false;
                return fit;
            }
            fit.params = b;
            int changed = 0;
            for (std::size_t k = 0; k < sel.size(); ++k) {
                auto const& h = hits[sel[k]];
                auto const r = row_of(h);
                double pred = r.offset;
                for (int j = 0; j < n; ++j) {
                    pred += r.a[j] * fit.params[j];
                }
                int const s = sign_for(h, pred, slope_of(h, fit.params));
                if (s != fit.signs[k]) {
                    fit.signs[k] = s;
                    ++changed;
                }
            }
            if (changed == 0) {
                break;
            }
        }
        fit.chi2 = 0;
        for (std::size_t k = 0; k < sel.size(); ++k) {
            auto const& h = hits[sel[k]];
            auto const r = row_of(h);
            double pred = r.offset;
            for (int j = 0; j < n; ++j) {
                pred += r.a[j] * fit.params[j];
            }
            double const res = measured(h, fit.signs[k], slope_of(h, fit.params)) - pred;
            fit.chi2 += res * res / (config_.sigma * config_.sigma);
        }
        fit.ndf = static_cast<int>(sel.size()) - n;
        fit.ok = true;
        return fit;
    }

    /// The hit of `layer` closest to `pred` within the tube radius plus the
    /// window, or none.
    std::optional<std::size_t> closest(std::vector<Hit> const& hits,
                                       std::map<int, std::vector<std::size_t>> const& by_layer,
                                       std::vector<bool> const& used, int layer,
                                       double pred) const {
        auto const it = by_layer.find(layer);
        if (it == by_layer.end()) {
            return std::nullopt;
        }
        double const cut = geometry_.layers()[layer].radius + config_.window;
        std::optional<std::size_t> best;
        double best_d = cut;
        for (auto const i : it->second) {
            if (used[i]) {
                continue;
            }
            double const d = std::abs(hits[i].wire - pred);
            if (d < best_d) {
                best_d = d;
                best = i;
            }
        }
        return best;
    }

    /// Straight x-z tracklets in the x layers `group`, seeded by pairs of hits
    /// in stations `sa` and `sb`; best first, each hit in one tracklet only.
    std::vector<XTracklet> find_x(std::vector<Hit> const& hits,
                                  std::map<int, std::vector<std::size_t>> const& by_layer,
                                  std::vector<bool>& used, std::vector<int> const& group, int sa,
                                  int sb, double z_ref, int min_hits) const {
        auto const& layers = geometry_.layers();
        std::vector<std::size_t> seeds_a, seeds_b;
        for (auto const l : group) {
            auto const it = by_layer.find(l);
            if (it == by_layer.end()) {
                continue;
            }
            auto& target = layers[l].station == sa ? seeds_a : seeds_b;
            if (layers[l].station == sa || layers[l].station == sb) {
                target.insert(target.end(), it->second.begin(), it->second.end());
            }
        }
        auto const row_of = [&](Hit const& h) {
            double const dz = h.z - z_ref;
            return detail::Row{.a = {1, dz, 0, 0}, .offset = 0};
        };
        auto const slope_of = [](Hit const&, std::array<double, 4> const& p) { return p[1]; };

        std::vector<XTracklet> found;
        for (auto const ia : seeds_a) {
            for (auto const ib : seeds_b) {
                auto const& ha = hits[ia];
                auto const& hb = hits[ib];
                double const za = ha.z, zb = hb.z;
                if (za == zb) {
                    continue;
                }
                double const tx = (hb.wire - ha.wire) / (zb - za);
                if (std::abs(tx) > config_.max_slope) {
                    continue;
                }
                std::vector<std::size_t> sel;
                for (auto const l : group) {
                    double const pred = ha.wire + tx * (layers[l].z - za);
                    if (auto const i = closest(hits, by_layer, used, l, pred)) {
                        sel.push_back(*i);
                    }
                }
                if (std::ranges::find(sel, ia) == sel.end() ||
                    std::ranges::find(sel, ib) == sel.end() ||
                    static_cast<int>(sel.size()) < min_hits) {
                    continue;
                }
                auto const fit = fit_lr(hits, sel, 2, row_of, slope_of, min_hits);
                if (!fit.ok || std::abs(fit.params[1]) > config_.max_slope) {
                    continue;
                }
                XTracklet t{.params = {fit.params[0], fit.params[1]},
                            .hits = {},
                            .chi2 = fit.chi2,
                            .ndf = fit.ndf};
                for (std::size_t k = 0; k < sel.size(); ++k) {
                    t.hits.push_back({sel[k], fit.signs[k]});
                }
                found.push_back(std::move(t));
            }
        }
        // best first: most hits, then lowest chi2; each hit used once
        std::ranges::sort(found, [](XTracklet const& a, XTracklet const& b) {
            if (a.hits.size() != b.hits.size()) {
                return a.hits.size() > b.hits.size();
            }
            return a.chi2 < b.chi2;
        });
        std::vector<XTracklet> accepted;
        for (auto& t : found) {
            if (std::ranges::any_of(t.hits, [&](Assigned const& a) { return used[a.hit]; })) {
                continue;
            }
            for (auto const& a : t.hits) {
                used[a.hit] = true;
            }
            accepted.push_back(std::move(t));
        }
        return accepted;
    }

    /// Add the stereo hits to a tracklet of T1/T2 and refit all of them in
    /// 3D; false if too few stereo hits fit.
    bool add_stereo(std::vector<Hit> const& hits,
                    std::map<int, std::vector<std::size_t>> const& by_layer,
                    std::vector<bool>& used, XTracklet const& xt, Candidate& c) const {
        auto const& layers = geometry_.layers();
        auto const x_at = [&](double z) { return xt.params[0] + xt.params[1] * (z - z_ref_up_); };
        // y of a stereo hit's wire at the tracklet's x
        auto const y_of = [&](Hit const& h) {
            auto const& l = layers[h.layer];
            return (h.wire - l.meas_dir[0] * x_at(h.z)) / l.meas_dir[1];
        };
        std::vector<std::size_t> st1, st2;
        for (auto const l : stereo_up_) {
            auto const it = by_layer.find(l);
            if (it == by_layer.end()) {
                continue;
            }
            for (auto const i : it->second) {
                if (!used[i]) {
                    (layers[l].station == 1 ? st1 : st2).push_back(i);
                }
            }
        }
        // y from the stereo hits with x fixed: coord - m_x x = m_y (y0 + ty dz)
        auto const row_y = [&](Hit const& h) {
            auto const& l = layers[h.layer];
            double const dz = h.z - z_ref_up_;
            return detail::Row{.a = {l.meas_dir[1], l.meas_dir[1] * dz, 0, 0},
                               .offset = l.meas_dir[0] * x_at(h.z)};
        };
        auto const slope_y = [&](Hit const& h, std::array<double, 4> const& p) {
            auto const& l = layers[h.layer];
            return l.meas_dir[0] * xt.params[1] + l.meas_dir[1] * p[1];
        };
        detail::Fit best;
        std::vector<std::size_t> best_sel;
        for (auto const ia : st1) {
            for (auto const ib : st2) {
                double const za = hits[ia].z, zb = hits[ib].z;
                double const ya = y_of(hits[ia]), yb = y_of(hits[ib]);
                double const ty = (yb - ya) / (zb - za);
                if (std::abs(ty) > config_.max_slope) {
                    continue;
                }
                std::vector<std::size_t> sel;
                for (auto const l : stereo_up_) {
                    double const z = layers[l].z;
                    double const pred = layers[l].coord(x_at(z), ya + ty * (z - za));
                    if (auto const i = closest(hits, by_layer, used, l, pred)) {
                        sel.push_back(*i);
                    }
                }
                if (static_cast<int>(sel.size()) < config_.min_stereo) {
                    continue;
                }
                auto const fit = fit_lr(hits, sel, 2, row_y, slope_y, config_.min_stereo);
                if (!fit.ok || std::abs(fit.params[1]) > config_.max_slope) {
                    continue;
                }
                if (!best.ok || sel.size() > best_sel.size() ||
                    (sel.size() == best_sel.size() && fit.chi2 < best.chi2)) {
                    best = fit;
                    best_sel = sel;
                }
            }
        }
        if (!best.ok) {
            return false;
        }

        // all hits of T1/T2 in one 3D straight-line fit
        std::vector<std::size_t> sel;
        for (auto const& a : xt.hits) {
            sel.push_back(a.hit);
        }
        sel.insert(sel.end(), best_sel.begin(), best_sel.end());
        auto const row_3d = [&](Hit const& h) {
            auto const& l = layers[h.layer];
            double const dz = h.z - z_ref_up_;
            return detail::Row{
                .a = {l.meas_dir[0], l.meas_dir[0] * dz, l.meas_dir[1], l.meas_dir[1] * dz},
                .offset = 0};
        };
        auto const slope_3d = [&](Hit const& h, std::array<double, 4> const& p) {
            auto const& l = layers[h.layer];
            return l.meas_dir[0] * p[1] + l.meas_dir[1] * p[3];
        };
        int const min_3d = config_.min_x_up + config_.min_stereo;
        auto const fit = fit_lr(hits, sel, 4, row_3d, slope_3d, min_3d);
        if (!fit.ok) {
            return false;
        }
        c.up = fit.params;
        c.chi2_up = fit.chi2;
        c.ndf_up = fit.ndf;
        for (std::size_t k = 0; k < sel.size(); ++k) {
            c.hits.push_back({sel[k], fit.signs[k]});
            used[sel[k]] = true;
            (layers[hits[sel[k]].layer].stereo ? c.n_stereo : c.n_x_up) += 1;
        }
        return true;
    }

    /// Match the candidates to the tracklets behind the magnet: the pairs
    /// closest at the kick plane first, each tracklet used once.
    void match(std::vector<XTracklet> const& down, std::vector<Candidate>& cands) const {
        double const z = config_.kick_z;
        std::vector<std::tuple<double, std::size_t, std::size_t>> pairs;
        for (std::size_t i = 0; i < cands.size(); ++i) {
            double const x_up = cands[i].up[0] + cands[i].up[1] * (z - z_ref_up_);
            for (std::size_t k = 0; k < down.size(); ++k) {
                double const dx = down[k].params[0] + down[k].params[1] * (z - z_ref_down_) - x_up;
                if (std::abs(dx) < config_.match_window) {
                    pairs.emplace_back(std::abs(dx), i, k);
                }
            }
        }
        std::ranges::sort(pairs);
        std::vector<bool> up_done(cands.size(), false), down_done(down.size(), false);
        for (auto const& [adx, i, k] : pairs) {
            if (up_done[i] || down_done[k]) {
                continue;
            }
            up_done[i] = down_done[k] = true;
            auto& c = cands[i];
            auto const& d = down[k];
            c.matched = true;
            c.down = d.params;
            c.chi2_down = d.chi2;
            c.ndf_down = d.ndf;
            c.n_down = static_cast<int>(d.hits.size());
            c.match_dx = d.params[0] + d.params[1] * (z - z_ref_down_) -
                         (c.up[0] + c.up[1] * (z - z_ref_up_));
            c.hits.insert(c.hits.end(), d.hits.begin(), d.hits.end());
            double const ty = c.up[3];
            double const ux_in = c.up[1] / std::sqrt(1 + c.up[1] * c.up[1] + ty * ty);
            double const ux_out = d.params[1] / std::sqrt(1 + d.params[1] * d.params[1] + ty * ty);
            double const du = ux_out - ux_in;
            if (du != 0) {
                c.p = std::abs(config_.kick_gev / du);
                c.charge = du * config_.kick_gev > 0 ? +1 : -1;
            }
        }
    }

    Geometry const& geometry_;
    PatrecConfig config_;
    std::vector<int> x_up_, stereo_up_, x_down_;
    double z_ref_up_ = 0, z_ref_down_ = 0;
};

}  // namespace SHiP::trout::muflux
