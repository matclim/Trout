// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// muflux_tracks_output.cpp — Phlex module plugin
//
// Writes the 2018 muon-flux tracks of muflux_tracking to an RNTuple named
// `muflux_tracks`, one entry per event, with the fields of
// SHiP::trout::muflux::MufluxTracks (MufluxEvent.hpp): fundamental types and
// std::vectors of them only, so plain ROOT, PyROOT or uproot read the file
// without dictionaries. Lengths in mm, momenta in GeV. Events finish in any
// order, so entries are not sorted by `entry`.
//
// Configuration: rntuple_file (default muflux_tracks.root), layer (required),
// creator (default muflux_tracking).

#include "detectors/muflux/MufluxEvent.hpp"
#include "phlex/core/product_selector.hpp"
#include "phlex/module.hpp"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <cstdint>
#include <memory>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace {

using SHiP::trout::muflux::MufluxTracks;

class TracksWriter {
   public:
    explicit TracksWriter(std::string const& filename) {
        auto model = ROOT::RNTupleModel::Create();
        entry_ = model->MakeField<std::uint64_t>("entry");
        auto i32 = [&](char const* name) {
            ints_.emplace_back(model->MakeField<std::int32_t>(name));
        };
        for (auto const* n : {"n_digits", "n_unknown", "n_upstream_only", "n_rpc"}) {
            i32(n);
        }
        for (auto const* n : {"q", "fit_ok", "ndf", "n_hits", "n_x_up", "n_stereo", "n_down",
                              "lr_changes", "n_outliers", "true_track"}) {
            ivecs_.emplace_back(model->MakeField<std::vector<std::int32_t>>(n));
        }
        for (auto const* n :
             {"p", "sigma_p", "chi2", "p_patrec", "match_dx", "x0", "y0", "z0", "px0", "py0", "pz0",
              "x1", "y1", "z1", "px1", "py1", "pz1", "true_frac"}) {
            fvecs_.emplace_back(model->MakeField<std::vector<float>>(n));
        }
        writer_ = ROOT::RNTupleWriter::Recreate(std::move(model), "muflux_tracks", filename);
        spdlog::info("muflux_tracks_output: writing '{}'", filename);
    }

    // Called serially (concurrency::serial): the writer is not thread-safe.
    void write(MufluxTracks const& t) {
        *entry_ = t.entry;
        std::int32_t const ints[] = {t.n_digits, t.n_unknown, t.n_upstream_only, t.n_rpc};
        for (std::size_t i = 0; i < ints_.size(); ++i) {
            *ints_[i] = ints[i];
        }
        std::vector<std::int32_t> const* ivecs[] = {
            &t.q,        &t.fit_ok, &t.ndf,        &t.n_hits,     &t.n_x_up,
            &t.n_stereo, &t.n_down, &t.lr_changes, &t.n_outliers, &t.true_track};
        for (std::size_t i = 0; i < ivecs_.size(); ++i) {
            *ivecs_[i] = *ivecs[i];
        }
        std::vector<float> const* fvecs[] = {
            &t.p,  &t.sigma_p, &t.chi2, &t.p_patrec, &t.match_dx, &t.x0,
            &t.y0, &t.z0,      &t.px0,  &t.py0,      &t.pz0,      &t.x1,
            &t.y1, &t.z1,      &t.px1,  &t.py1,      &t.pz1,      &t.true_frac};
        for (std::size_t i = 0; i < fvecs_.size(); ++i) {
            *fvecs_[i] = *fvecs[i];
        }
        writer_->Fill();
    }

   private:
    std::shared_ptr<std::uint64_t> entry_;
    std::vector<std::shared_ptr<std::int32_t>> ints_;
    std::vector<std::shared_ptr<std::vector<std::int32_t>>> ivecs_;
    std::vector<std::shared_ptr<std::vector<float>>> fvecs_;
    std::unique_ptr<ROOT::RNTupleWriter> writer_;  // last: written out before the fields go
};

}  // namespace

PHLEX_REGISTER_ALGORITHMS(m, config) {
    auto const rntuple_file =
        config.get<std::string>("rntuple_file", std::string{"muflux_tracks.root"});
    auto const layer = phlex::experimental::identifier{config.get<std::string>("layer")};
    auto const creator = phlex::experimental::identifier{
        config.get<std::string>("creator", std::string{"muflux_tracking"})};

    auto writer = m.make<TracksWriter>(rntuple_file);
    writer.observe("write_muflux_tracks", &TracksWriter::write, phlex::concurrency::serial)
        .input_family(
            phlex::product_selector{.creator = creator, .layer = layer, .suffix = "muflux_tracks"});
}
