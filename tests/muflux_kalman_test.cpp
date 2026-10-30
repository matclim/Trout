// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// muflux_kalman_test.cpp — Kalman fit of the 2018 muon-flux tracks on toy
// events (muflux_toy.hpp): the pattern-recognition candidates of single muons
// are fitted through the same field the toys were propagated in, and the fit
// is compared with the truth (momentum, charge, position at the first layer,
// pulls, chi2). With the measured Goliath map as the first argument
// (FairShip's field/GoliathFieldMap.txt), it is used instead of the Gaussian
// dipole.

#include "detectors/muflux/MufluxKalman.hpp"
#include "muflux_toy.hpp"

#include <Acts/Definitions/Units.hpp>
#include <Acts/MagneticField/MagneticFieldProvider.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, std::string const& what) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++failures;
    }
}

// The toy field (T, mm) as an ACTS field provider.
class ToyField final : public Acts::MagneticFieldProvider {
   public:
    explicit ToyField(muflux_toy::Field f) : f_{std::move(f)} {}
    struct Cache {
        explicit Cache(Acts::MagneticFieldContext const&) {}
    };
    Acts::MagneticFieldProvider::Cache makeCache(
        Acts::MagneticFieldContext const& mctx) const override {
        return Acts::MagneticFieldProvider::Cache(std::in_place_type<Cache>, mctx);
    }
    Acts::Result<Acts::Vector3> getField(Acts::Vector3 const& p,
                                         Acts::MagneticFieldProvider::Cache&) const override {
        auto const b = f_(p.x(), p.y(), p.z());
        return Acts::Result<Acts::Vector3>::success(Acts::Vector3(b[0], b[1], b[2]) *
                                                    Acts::UnitConstants::T);
    }

   private:
    muflux_toy::Field f_;
};

struct Quantiles {
    double median = 0, width = 0;  // width: half the central 68% interval
    explicit Quantiles(std::vector<double> v) {
        if (v.empty()) {
            return;
        }
        std::ranges::sort(v);
        auto const at = [&](double f) { return v[static_cast<std::size_t>(f * (v.size() - 1))]; };
        median = at(0.5);
        width = 0.5 * (at(0.84) - at(0.16));
    }
    static long tail(std::vector<double> const& v, double cut) {
        return std::ranges::count_if(v, [cut](double x) { return std::abs(x) > cut; });
    }
};

}  // namespace

int main(int argc, char** argv) {
    namespace mf = SHiP::trout::muflux;
    using namespace muflux_toy;
    auto const wires = toy_wires();
    mf::Geometry const geo{wires};
    Field const field = argc > 1 ? measured_map(argv[1]) : gaussian_dipole();
    auto const [kick, kick_z] = kick_of(field);

    mf::PatrecConfig pconfig;
    pconfig.kick_gev = kick;
    pconfig.kick_z = kick_z;
    mf::Patrec const patrec{geo, pconfig};
    mf::KalmanFit const kf{geo, std::make_shared<ToyField>(field), mf::KalmanFit::Config{}};
    Generator gen{geo, field, 20180702, 0.0};  // single muons

    long n_cand = 0, n_ok = 0, n_charge = 0, n_lr = 0, n_out = 0;
    std::vector<double> dp_lo, dp_hi, pull, chi2ndf, res, dx0, dy0;
    for (int ev_n = 0; ev_n < 1000; ++ev_n) {
        auto const ev = gen.next();
        auto cands = patrec.find(ev.hits);
        int const best = best_candidate(ev, cands, 0);
        if (best < 0) {
            continue;
        }
        ++n_cand;
        auto& c = cands[best];
        auto const fit = kf.fit(ev.hits, c, patrec);
        if (!fit.ok) {
            continue;
        }
        ++n_ok;
        auto const& mu = ev.muons[0];
        n_charge += fit.charge == mu.q;
        n_lr += fit.lr_changes > 0;
        n_out += fit.n_outliers > 0;
        (mu.p < 50 ? dp_lo : dp_hi).push_back(fit.p / mu.p - 1);
        pull.push_back((fit.p - mu.p) / fit.sigma_p);
        chi2ndf.push_back(fit.chi2 / std::max(fit.ndf, 1));
        for (double r : fit.residuals) {
            if (!std::isnan(r)) {
                res.push_back(r);
            }
        }
        // position at the first layer, straight back from the first fitted
        // point: T1 is field-free
        double const z0 = geo.layers().front().z;
        auto const& f = fit.first;
        dx0.push_back(f.position.x() + f.momentum.x() / f.momentum.z() * (z0 - f.position.z()) -
                      mu.first.x);
        dy0.push_back(f.position.y() + f.momentum.y() / f.momentum.z() * (z0 - f.position.z()) -
                      mu.first.y);
    }
    std::printf(
        "%ld candidates, %ld fitted; charge correct %.4f; with outliers %.3f; refitted after a "
        "side change %.3f\n",
        n_cand, n_ok, static_cast<double>(n_charge) / n_ok, static_cast<double>(n_out) / n_ok,
        static_cast<double>(n_lr) / n_ok);
    // medians and half-widths of the central 68%, robust against the few tails
    auto const lo = Quantiles{dp_lo}, hi = Quantiles{dp_hi}, pl = Quantiles{pull},
               ch = Quantiles{chi2ndf}, rs = Quantiles{res}, qx = Quantiles{dx0},
               qy = Quantiles{dy0};
    std::printf(
        "  p_fit/p_true - 1: p < 50 GeV median %.4f width %.4f, p > 50 GeV median %.4f width "
        "%.4f\n  p pull median %.3f width %.3f; chi2/ndf median %.3f; residual median %.4f "
        "width %.4f mm\n  first layer: dx median %.3f width %.3f mm, dy median %.3f width %.3f "
        "mm; |dp/p| > 0.3: %.4f\n",
        lo.median, lo.width, hi.median, hi.width, pl.median, pl.width, ch.median, rs.median,
        rs.width, qx.median, qx.width, qy.median, qy.width,
        static_cast<double>(Quantiles::tail(dp_lo, 0.3) + Quantiles::tail(dp_hi, 0.3)) / n_ok);
    check(n_ok > 0.98 * n_cand, "fits converge");
    check(static_cast<double>(n_charge) / n_ok > 0.999, "charge");
    check(std::abs(lo.median) < 0.002 && std::abs(hi.median) < 0.005, "momentum bias");
    check(lo.width < 0.01, "momentum resolution below 50 GeV < 1%");
    check(std::abs(pl.median) < 0.2 && pl.width > 0.8 && pl.width < 1.25, "p pulls");
    check(ch.median > 0.5 && ch.median < 1.3, "chi2/ndf");
    check(rs.width > 0.2 && rs.width < 0.4, "unbiased residuals about sigma");
    check(qx.width < 0.3 && qy.width < 1.0, "position at the first layer");
    check((Quantiles::tail(dp_lo, 0.3) + Quantiles::tail(dp_hi, 0.3)) < 0.01 * n_ok,
          "momentum tails < 1%");

    // the same muons fitted without the material: the pulls show the multiple
    // scattering the fit then ignores
    {
        mf::KalmanFit::Config no_material;
        no_material.material = false;
        mf::KalmanFit const kf0{geo, std::make_shared<ToyField>(field), no_material};
        Generator gen0{geo, field, 20180702, 0.0};
        std::vector<double> pull0, dp0;
        for (int ev_n = 0; ev_n < 1000; ++ev_n) {
            auto const ev = gen0.next();
            auto cands = patrec.find(ev.hits);
            int const best = best_candidate(ev, cands, 0);
            if (best < 0) {
                continue;
            }
            auto const fit = kf0.fit(ev.hits, cands[best], patrec);
            if (fit.ok) {
                pull0.push_back((fit.p - ev.muons[0].p) / fit.sigma_p);
                if (ev.muons[0].p < 50) {
                    dp0.push_back(fit.p / ev.muons[0].p - 1);
                }
            }
        }
        auto const q0 = Quantiles{pull0}, d0 = Quantiles{dp0};
        std::printf(
            "  without material in the fit: p pull median %.3f width %.3f; p_fit/p_true - 1 "
            "(< 50 GeV) median %.4f width %.4f\n",
            q0.median, q0.width, d0.median, d0.width);
        check(q0.width > pl.width, "the material model narrows the pulls");
    }

    if (failures == 0) {
        std::puts("muflux_kalman: all checks passed");
    }
    return failures == 0 ? 0 : 1;
}
