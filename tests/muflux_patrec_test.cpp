// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// muflux_patrec_test.cpp — layers and pattern recognition of the 2018
// muon-flux spectrometer on toy events (muflux_toy.hpp). With the measured
// Goliath map as the first argument (FairShip's field/GoliathFieldMap.txt),
// the toy muons are propagated through it instead of the Gaussian dipole.

#include "muflux_toy.hpp"

#include <cmath>
#include <cstdio>
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

}  // namespace

int main(int argc, char** argv) {
    namespace mf = SHiP::trout::muflux;
    using namespace muflux_toy;
    auto const wires = toy_wires();
    mf::Geometry const geo{wires};

    // geometry
    check(geo.layers().size() == 24, "24 layers");
    int nstereo = 0;
    for (auto const& l : geo.layers()) {
        nstereo += l.stereo;
        double const cross = l.meas_dir[0] * l.wire_dir[1] - l.meas_dir[1] * l.wire_dir[0];
        check(std::abs(cross - 1) < 1e-12, "meas_dir x wire_dir = +z");
    }
    check(nstereo == 8, "8 stereo layers");
    check(geo.layers().front().z < geo.layers().back().z, "layers ordered by z");
    {
        auto const& l = geo.layers()[geo.layer_index(10002001)];
        check(l.meas_dir[0] == 1 && l.meas_dir[1] == 0 &&
                  std::abs(geo.tube(10002001).coord + 217.7) < 1e-9,
              "x view measures x");
        check(geo.layer_index(99999999) == -1, "unknown tube");
    }

    Field const field = argc > 1 ? measured_map(argv[1]) : gaussian_dipole();
    auto const [kick, kick_z] = kick_of(field);
    std::printf("kick %.4f GeV at z = %.1f mm\n", kick, kick_z);

    mf::PatrecConfig config;
    config.kick_gev = kick;
    config.kick_z = kick_z;
    mf::Patrec const patrec{geo, config};
    Generator gen{geo, field, 20180701};

    long n_true1 = 0, n_found1 = 0, n_true = 0, n_reco = 0, n_found = 0, n_charge = 0, n_ghost = 0,
         n_sign = 0, n_signed = 0;
    std::vector<double> dp;  // p_reco / p_true - 1
    for (int ev_n = 0; ev_n < 2000; ++ev_n) {
        auto const ev = gen.next();
        auto const cands = patrec.find(ev.hits);
        std::vector<bool> taken(cands.size(), false);
        int const n_mu = static_cast<int>(ev.muons.size());
        for (int m = 0; m < n_mu; ++m) {
            auto const& mu = ev.muons[m];
            bool const reconstructible =
                mu.counts[0] >= 6 && mu.counts[1] >= 6 && mu.counts[2] >= 6;
            n_true += reconstructible;
            n_true1 += reconstructible && n_mu == 1;
            int const best = best_candidate(ev, cands, m);
            if (best < 0) {
                continue;
            }
            taken[best] = true;
            if (!reconstructible) {
                continue;
            }
            ++n_found;
            n_found1 += n_mu == 1;
            auto const& c = cands[best];
            n_charge += c.charge == mu.q;
            dp.push_back(c.p / mu.p - 1);
            for (auto const& a : c.hits) {
                auto const& t = ev.truth[a.hit];
                if (t.muon == m && t.hit.drift > 1.0) {
                    ++n_signed;
                    n_sign += a.sign == t.sign;
                }
            }
        }
        for (std::size_t c = 0; c < cands.size(); ++c) {
            n_reco += cands[c].matched;
            n_ghost += cands[c].matched && !taken[c];
        }
    }
    // core of the momentum estimate, and its tails (wrong matches)
    double mean = 0, rms = 0;
    long n_core = 0, n_tail = 0;
    for (double d : dp) {
        if (std::abs(d) < 0.3) {
            mean += d;
            ++n_core;
        } else {
            ++n_tail;
        }
    }
    mean /= n_core;
    for (double d : dp) {
        if (std::abs(d) < 0.3) {
            rms += (d - mean) * (d - mean);
        }
    }
    rms = std::sqrt(rms / n_core);
    double const tail = static_cast<double>(n_tail) / dp.size();
    double const eff = static_cast<double>(n_found) / n_true;
    double const eff1 = static_cast<double>(n_found1) / n_true1;
    double const charge = static_cast<double>(n_charge) / n_found;
    double const sign = static_cast<double>(n_sign) / n_signed;
    double const ghost = static_cast<double>(n_ghost) / n_reco;
    std::printf(
        "%ld reconstructible muons: efficiency %.4f (single-muon events %.4f), charge correct "
        "%.4f, left/right correct %.4f (drift > 1 mm), ghosts %.4f of %ld matched candidates; "
        "p_reco/p_true - 1: mean %.4f rms %.4f within 30%%, %.4f outside\n",
        n_true, eff, eff1, charge, sign, ghost, n_reco, mean, rms, tail);
    check(eff1 > 0.97, "single-muon efficiency > 97%");
    check(eff > 0.93, "efficiency > 93%");
    check(charge > 0.99, "charge > 99% correct");
    check(sign > 0.98, "left/right > 98% correct");
    check(ghost < 0.03, "ghosts < 3%");
    check(std::abs(mean) < 0.03, "momentum bias < 3%");
    check(rms < 0.08, "momentum spread < 8%");
    check(tail < 0.01, "momentum outliers < 1%");

    if (failures == 0) {
        std::puts("muflux_patrec: all checks passed");
    }
    return failures == 0 ? 0 : 1;
}
