// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// muflux_tracking.cpp — Phlex module plugin
//
// Track reconstruction for the 2018 muon-flux spectrometer: per event, the
// pattern recognition of MufluxPatrec.hpp (tracklets in T1/T2 and T3/T4,
// matched across Goliath) and the Kalman fit of MufluxKalman.hpp through the
// field of acts_field_provider (the Goliath map). The kick used by the pattern
// recognition, 0.2998 * the field integral along the beam axis, and the z of
// the kick plane are computed from that field when the first event arrives.
//
// Input: `muflux_digits` (read_muflux_digits) and `field` of
// spectrometer_field (acts_field_provider). Output: `muflux_tracks`.
//
// Configuration:
//   layer, wire_table                       required
//   sigma           hit resolution [mm] (0.27, as in Shannon's muflux_digitise)
//   window          beyond the tube radius when collecting hits [mm] (5)
//   lr_seeds        seed with the left/right combinations of the seed hits (true)
//   min_x_up, min_stereo, min_x_down        hits per tracklet (4, 4, 4)
//   max_chi2_ndf    per tracklet (25)
//   match_window    at the kick plane [mm] (30)
//   lr_passes       refits after choosing the sides from the fit (2)

#include "acts_field_provider.hpp"
#include "detectors/muflux/MufluxEvent.hpp"
#include "detectors/muflux/MufluxGeometry.hpp"
#include "detectors/muflux/MufluxKalman.hpp"
#include "detectors/muflux/MufluxPatrec.hpp"
#include "phlex/core/product_selector.hpp"
#include "phlex/module.hpp"

#include <Acts/Definitions/Units.hpp>
#include <Acts/MagneticField/MagneticFieldContext.hpp>

#include <atomic>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace mf = SHiP::trout::muflux;

class Tracker {
   public:
    Tracker(std::string const& wire_table, mf::PatrecConfig patrec, mf::KalmanFit::Config kalman)
        : geometry_{mf::read_wire_table(wire_table)},
          patrec_config_{patrec},
          kalman_config_{kalman} {
        spdlog::info("muflux_tracking: {} drift-tube layers from '{}'", geometry_.layers().size(),
                     wire_table);
    }

    ~Tracker() {
        try {
            spdlog::info(
                "muflux_tracking: {} events, {} digits ({} not in the wire table), {} matched "
                "candidates, {} fitted, {} T1/T2 candidates without a match",
                events_.load(), digits_.load(), unknown_.load(), matched_.load(), fitted_.load(),
                upstream_only_.load());
        } catch (...) {
        }
    }

    mf::MufluxTracks process(mf::MufluxDigits const& d,
                             std::shared_ptr<SHiP::trout::DetectorField> const& field) {
        std::call_once(init_, [&] { init(field); });

        mf::MufluxTracks out;
        out.entry = d.entry;
        out.n_digits = static_cast<std::int32_t>(d.dt_id.size());
        out.n_rpc = static_cast<std::int32_t>(d.rpc_id.size());
        std::vector<mf::Hit> hits;
        hits.reserve(d.dt_id.size());
        for (std::size_t i = 0; i < d.dt_id.size(); ++i) {
            int const layer = geometry_.layer_index(d.dt_id[i]);
            if (layer < 0) {
                ++out.n_unknown;
                continue;
            }
            hits.push_back({.layer = layer,
                            .tube = d.dt_id[i],
                            .wire = geometry_.tube(d.dt_id[i]).coord,
                            .drift = d.dt_drift[i],
                            .z = geometry_.tube(d.dt_id[i]).z,
                            .digit = i});
        }

        auto cands = patrec_->find(hits);
        double const z_first = geometry_.layers().front().z;
        for (auto& c : cands) {
            if (!c.matched) {
                ++out.n_upstream_only;
                continue;
            }
            auto const fit = kalman_->fit(hits, c, *patrec_);
            out.q.push_back(fit.ok ? fit.charge : c.charge);
            out.fit_ok.push_back(fit.ok);
            out.p.push_back(static_cast<float>(fit.ok ? fit.p : c.p));
            out.sigma_p.push_back(static_cast<float>(fit.sigma_p));
            out.chi2.push_back(static_cast<float>(fit.chi2));
            out.ndf.push_back(fit.ndf);
            out.n_hits.push_back(fit.ok ? fit.n_meas : static_cast<int>(c.hits.size()));
            out.n_x_up.push_back(c.n_x_up);
            out.n_stereo.push_back(c.n_stereo);
            out.n_down.push_back(c.n_down);
            out.p_patrec.push_back(static_cast<float>(c.p));
            out.match_dx.push_back(static_cast<float>(c.match_dx));
            out.lr_changes.push_back(fit.lr_changes);
            out.n_outliers.push_back(fit.n_outliers);
            if (fit.ok) {
                auto const& f = fit.first;
                double const dz = z_first - f.position.z();
                out.x0.push_back(
                    static_cast<float>(f.position.x() + f.momentum.x() / f.momentum.z() * dz));
                out.y0.push_back(
                    static_cast<float>(f.position.y() + f.momentum.y() / f.momentum.z() * dz));
                out.z0.push_back(static_cast<float>(z_first));
                out.px0.push_back(static_cast<float>(f.momentum.x()));
                out.py0.push_back(static_cast<float>(f.momentum.y()));
                out.pz0.push_back(static_cast<float>(f.momentum.z()));
                auto const& l = fit.last;
                out.x1.push_back(static_cast<float>(l.position.x()));
                out.y1.push_back(static_cast<float>(l.position.y()));
                out.z1.push_back(static_cast<float>(l.position.z()));
                out.px1.push_back(static_cast<float>(l.momentum.x()));
                out.py1.push_back(static_cast<float>(l.momentum.y()));
                out.pz1.push_back(static_cast<float>(l.momentum.z()));
                ++fitted_;
            } else {
                // the pattern-recognition state at the first layer
                auto const xy = c.position(z_first, false);
                auto const t = c.slopes(false);
                double const pz = c.p / std::sqrt(1 + t[0] * t[0] + t[1] * t[1]);
                for (auto* v : {&out.x0, &out.y0, &out.z0, &out.px0, &out.py0, &out.pz0}) {
                    v->push_back(0);
                }
                out.x0.back() = static_cast<float>(xy[0]);
                out.y0.back() = static_cast<float>(xy[1]);
                out.z0.back() = static_cast<float>(z_first);
                out.px0.back() = static_cast<float>(pz * t[0]);
                out.py0.back() = static_cast<float>(pz * t[1]);
                out.pz0.back() = static_cast<float>(pz);
                for (auto* v : {&out.x1, &out.y1, &out.z1, &out.px1, &out.py1, &out.pz1}) {
                    v->push_back(std::nanf(""));
                }
            }
            // simulation: the most frequent track among the hits
            int true_track = -1;
            float true_frac = 0;
            if (!d.dt_track.empty()) {
                std::map<std::int32_t, int> count;
                for (auto const& a : c.hits) {
                    ++count[d.dt_track[hits[a.hit].digit]];
                }
                for (auto const& [t, n] : count) {
                    if (n > true_frac * c.hits.size()) {
                        true_track = t;
                        true_frac = static_cast<float>(n) / c.hits.size();
                    }
                }
            }
            out.true_track.push_back(true_track);
            out.true_frac.push_back(true_frac);
            ++matched_;
        }
        ++events_;
        digits_ += out.n_digits;
        unknown_ += out.n_unknown;
        upstream_only_ += out.n_upstream_only;
        return out;
    }

   private:
    void init(std::shared_ptr<SHiP::trout::DetectorField> const& field) {
        if (!field || !field->field) {
            throw std::runtime_error("muflux_tracking: no magnetic field");
        }
        // kick and kick plane from By on the beam axis
        Acts::MagneticFieldContext const mctx;
        auto cache = field->field->makeCache(mctx);
        double integral = 0, moment = 0;
        constexpr double step = 5.;  // mm
        for (double z = -2000.; z < 12000.; z += step) {
            auto const b = field->field->getField(Acts::Vector3(0, 0, z), cache);
            double const by = b.ok() ? b.value().y() / Acts::UnitConstants::T : 0.;
            integral += by * step;
            moment += by * z * step;
        }
        if (integral == 0) {
            throw std::runtime_error("muflux_tracking: the field has no By along the beam axis");
        }
        auto config = patrec_config_;
        config.kick_gev = -0.299792458e-3 * integral;  // GeV per T mm
        config.kick_z = moment / integral;
        spdlog::info(
            "muflux_tracking: field integral {:.4f} T m along the beam axis, kick {:.4f} GeV at z "
            "= "
            "{:.1f} mm",
            integral / 1000., config.kick_gev, config.kick_z);
        patrec_ = std::make_unique<mf::Patrec>(geometry_, config);
        kalman_ = std::make_unique<mf::KalmanFit>(geometry_, field->field, kalman_config_);
    }

    mf::Geometry geometry_;
    mf::PatrecConfig patrec_config_;
    mf::KalmanFit::Config kalman_config_;
    std::once_flag init_;
    std::unique_ptr<mf::Patrec> patrec_;
    std::unique_ptr<mf::KalmanFit> kalman_;
    std::atomic<long> events_{0}, digits_{0}, unknown_{0}, matched_{0}, fitted_{0},
        upstream_only_{0};
};

}  // namespace

PHLEX_REGISTER_ALGORITHMS(m, config) {
    auto const layer = phlex::experimental::identifier{config.get<std::string>("layer")};
    mf::PatrecConfig patrec;
    patrec.sigma = config.get<double>("sigma", double{patrec.sigma});
    patrec.window = config.get<double>("window", double{patrec.window});
    patrec.lr_seeds = config.get<bool>("lr_seeds", bool{patrec.lr_seeds});
    patrec.min_x_up = config.get<int>("min_x_up", int{patrec.min_x_up});
    patrec.min_stereo = config.get<int>("min_stereo", int{patrec.min_stereo});
    patrec.min_x_down = config.get<int>("min_x_down", int{patrec.min_x_down});
    patrec.max_chi2_ndf = config.get<double>("max_chi2_ndf", double{patrec.max_chi2_ndf});
    patrec.match_window = config.get<double>("match_window", double{patrec.match_window});
    mf::KalmanFit::Config kalman;
    kalman.sigma = patrec.sigma;
    kalman.lr_passes = config.get<int>("lr_passes", int{kalman.lr_passes});

    auto tracker = std::make_shared<Tracker>(config.get<std::string>("wire_table"), patrec, kalman);

    m.transform(
         "muflux_tracking",
         [tracker](mf::MufluxDigits const& digits,
                   std::shared_ptr<SHiP::trout::DetectorField> const& field) {
             return tracker->process(digits, field);
         },
         phlex::concurrency::unlimited)
        .input_family(
            phlex::product_selector{
                .creator = "muflux_source", .layer = layer, .suffix = "muflux_digits"},
            phlex::product_selector{
                .creator = "spectrometer_field", .layer = "job", .suffix = "field"})
        .output_product_suffixes("muflux_tracks");
}
