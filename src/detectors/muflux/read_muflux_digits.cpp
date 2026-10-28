// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// read_muflux_digits.cpp — Phlex provider plugin
//
// Reads the 2018 muon-flux digits written by Shannon's muflux_digits_output
// (RNTuple `muflux_digits`, one entry per event; the 2018 data are converted
// to the same layout), one event per cell of `layer`: product
// `muflux_digits` of creator `muflux_source`. dt_track is read when present
// (simulation). Shannon writes the events in the order they finish, so a
// row's `entry` field, not its row number, is the event (the entry of the
// simulation input); it is passed on as MufluxDigits::entry.
//
// Configuration: input_file, layer (required), ntuple_name (default muflux_digits).

#include "detectors/muflux/MufluxEvent.hpp"
#include "phlex/configuration.hpp"
#include "phlex/model/data_cell_index.hpp"
#include "phlex/module.hpp"
#include "phlex/source.hpp"

#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>

#include <cstdint>
#include <memory>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using SHiP::trout::muflux::MufluxDigits;

class DigitReader {
   public:
    DigitReader(std::string const& file, std::string const& ntuple)
        : reader_{ROOT::RNTupleReader::Open(ntuple, file)},
          entry_{reader_->GetView<std::uint64_t>("entry")},
          dt_id_{reader_->GetView<std::vector<std::int32_t>>("dt_id")},
          dt_drift_{reader_->GetView<std::vector<float>>("dt_drift")},
          rpc_id_{reader_->GetView<std::vector<std::int32_t>>("rpc_id")} {
        if (reader_->GetDescriptor().FindFieldId("dt_track") != ROOT::kInvalidDescriptorId) {
            dt_track_.emplace(reader_->GetView<std::vector<std::int32_t>>("dt_track"));
        }
        spdlog::info("read_muflux_digits: '{}', {} events{}", file, reader_->GetNEntries(),
                     dt_track_ ? ", with simulated tracks" : "");
    }

    MufluxDigits read(std::uint64_t entry) {
        if (entry >= reader_->GetNEntries()) {
            throw std::out_of_range("read_muflux_digits: entry " + std::to_string(entry) +
                                    " beyond the " + std::to_string(reader_->GetNEntries()) +
                                    " of the input");
        }
        MufluxDigits d;
        d.entry = entry_(entry);
        d.dt_id = dt_id_(entry);
        d.dt_drift = dt_drift_(entry);
        d.rpc_id = rpc_id_(entry);
        if (dt_track_) {
            d.dt_track = (*dt_track_)(entry);
        }
        return d;
    }

   private:
    std::unique_ptr<ROOT::RNTupleReader> reader_;
    ROOT::RNTupleView<std::uint64_t> entry_;
    ROOT::RNTupleView<std::vector<std::int32_t>> dt_id_;
    ROOT::RNTupleView<std::vector<float>> dt_drift_;
    ROOT::RNTupleView<std::vector<std::int32_t>> rpc_id_;
    std::optional<ROOT::RNTupleView<std::vector<std::int32_t>>> dt_track_;
};

}  // namespace

PHLEX_REGISTER_PROVIDERS(m, config) {
    auto const input_file = config.get<std::string>("input_file");
    auto const ntuple = config.get<std::string>("ntuple_name", std::string{"muflux_digits"});
    auto const layer = phlex::experimental::identifier{config.get<std::string>("layer")};
    auto reader = std::make_shared<DigitReader>(input_file, ntuple);

    // RNTupleReader is not thread-safe: read serially
    m.provide(
         "read_muflux_digits",
         [reader](phlex::data_cell_index const& id) -> MufluxDigits {
             return reader->read(static_cast<std::uint64_t>(id.number()));
         },
         phlex::concurrency::serial)
        .output_product("muflux_source", phlex::experimental::identifier{"muflux_digits"}, layer);
}
