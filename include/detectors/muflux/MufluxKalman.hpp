// SPDX-FileCopyrightText: 2026 CERN for the benefit of the SHiP Collaboration
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// MufluxKalman.hpp — Kalman fit of the 2018 muon-flux tracks
//
// Fits a pattern-recognition candidate (MufluxPatrec.hpp) with ACTS's
// KalmanFitter and smoother, through the Goliath field, with a direct
// navigator over the measured tubes: no tracking geometry is needed. Each
// tube is a plane surface at its z with the local axes (meas_dir, wire_dir) of
// its layer, and each hit a 1D measurement of loc0, the coordinate across the
// wires: wire +- drift * sqrt(1 + t_m^2) relative to the layer centre (see
// MufluxPatrec.hpp). In the refits, hits whose predicted chi2 exceeds
// outlier_chi2 are left out of the filter. After a fit, the side of every hit is chosen again from
// the track without that hit (the unbiased smoothed state; the smoothed state
// for an outlier) and the track refitted while sides change (lr_passes).
//
// Material (option `material`, on by default): multiple scattering and energy
// loss in the aluminium tube walls. Each layer carries, on its tube surfaces
// and on one plane of its own, the wall thickness averaged over the layer
// (mean_wall_thickness in MufluxGeometry.hpp: 2.4 mm of aluminium, 2.7% of a
// radiation length, for 0.85 mm walls at 42 mm pitch). The wall density is an
// option (tube_wall_density, default that of aluminium): the geometry of the
// legacy simulation, and so a simulation built from it, gives the walls
// 0.9 g/cm3 ("CoilAluminium"), a third of a real tube's. A track meets the
// material of every layer it crosses between its first and last hit: the tube
// surface of its hit in that layer, or the layer's plane if it has none. The
// gas (Ar/CO2) and the air are left out (< 0.1% of a radiation length per
// layer, about 2% for the air between T1 and T4).

#pragma once

#include "detectors/muflux/MufluxGeometry.hpp"
#include "detectors/muflux/MufluxPatrec.hpp"

#include <Acts/Definitions/Algebra.hpp>
#include <Acts/Definitions/Units.hpp>
#include <Acts/EventData/AnyTrackStateProxy.hpp>
#include <Acts/EventData/BoundTrackParameters.hpp>
#include <Acts/EventData/ParticleHypothesis.hpp>
#include <Acts/EventData/SourceLink.hpp>
#include <Acts/EventData/TrackContainer.hpp>
#include <Acts/EventData/VectorMultiTrajectory.hpp>
#include <Acts/EventData/VectorTrackContainer.hpp>
#include <Acts/Geometry/GeometryContext.hpp>
#include <Acts/MagneticField/MagneticFieldContext.hpp>
#include <Acts/MagneticField/MagneticFieldProvider.hpp>
#include <Acts/Material/HomogeneousSurfaceMaterial.hpp>
#include <Acts/Material/Material.hpp>
#include <Acts/Material/MaterialSlab.hpp>
#include <Acts/Propagator/DirectNavigator.hpp>
#include <Acts/Propagator/EigenStepper.hpp>
#include <Acts/Propagator/Propagator.hpp>
#include <Acts/Surfaces/PlaneSurface.hpp>
#include <Acts/Surfaces/RectangleBounds.hpp>
#include <Acts/Surfaces/Surface.hpp>
#include <Acts/TrackFitting/GainMatrixSmoother.hpp>
#include <Acts/TrackFitting/GainMatrixUpdater.hpp>
#include <Acts/TrackFitting/KalmanFitter.hpp>
#include <Acts/Utilities/CalibrationContext.hpp>
#include <Acts/Utilities/Logger.hpp>
#include <Acts/Utilities/TrackHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace SHiP::trout::muflux {

/// Source link of a muflux measurement: its index in the fit's measurements.
struct MufluxSourceLink {
    std::size_t index = 0;
    friend bool operator==(MufluxSourceLink const&, MufluxSourceLink const&) = default;
};

/// Position [mm] and momentum [GeV] of a fitted track at one point.
struct FittedPoint {
    Acts::Vector3 position = Acts::Vector3::Zero();
    Acts::Vector3 momentum = Acts::Vector3::Zero();
};

struct FitOutput {
    bool ok = false;
    double chi2 = 0;
    int ndf = 0;
    int n_meas = 0;
    int n_outliers = 0;  ///< hits left out of the fit by the outlier cut
    int charge = 0;
    double p = 0;        ///< GeV, at the first measurement
    double sigma_p = 0;  ///< GeV
    FittedPoint first;   ///< at the first measured layer
    FittedPoint last;    ///< at the last measured layer
    int lr_changes = 0;  ///< sides changed after the first fit
    /// unbiased residual (measured loc0 - track without the hit) per candidate
    /// hit [mm]; NaN if not in the fit
    std::vector<double> residuals;
};

class KalmanFit {
   public:
    struct Config {
        double sigma = 0.27;         ///< hit resolution [mm]
        int lr_passes = 2;           ///< refits after choosing the sides from the fit (>= 1)
        double sigma_loc0 = 2.0;     ///< start covariance [mm]
        double sigma_loc1 = 20.0;    ///< along the wires [mm]
        double sigma_theta = 0.005;  ///< rad
        double sigma_qop_rel = 0.3;  ///< of q/p
        double outlier_chi2 = 16;    ///< predicted chi2 above which a hit is an outlier
        bool material = true;        ///< multiple scattering and energy loss in the tube walls
        double tube_wall = 0.85;     ///< aluminium wall of the drift tubes [mm]
        double tube_wall_density = 2.699;  ///< of the walls [g/cm3]; 0.9 in the legacy geometry
    };

    KalmanFit(Geometry const& geometry, std::shared_ptr<const Acts::MagneticFieldProvider> field,
              Config config)
        : geometry_{geometry},
          config_{config},
          gctx_{Acts::GeometryContext::dangerouslyDefaultConstruct()} {
        // aluminium (PDG): X0 24.01 g/cm2, nuclear interaction length 107.2 g/cm2,
        // at the configured density
        using Acts::UnitConstants::cm;
        using Acts::UnitConstants::cm3;
        using Acts::UnitConstants::g;
        double const rho = config_.tube_wall_density;
        auto const aluminium = Acts::Material::fromMassDensity(
            static_cast<float>(24.01 / rho * cm), static_cast<float>(107.2 / rho * cm), 26.98f,
            13.f, static_cast<float>(rho * g / cm3));
        // one plane per tube, at its z, with the local axes of its layer; the
        // tubes of a layer share the origin of loc0 (the layer centre); and one
        // plane per layer, at its z, for the material of the layers without a hit
        for (auto const& l : geometry_.layers()) {
            Acts::RotationMatrix3 rot;
            rot.col(0) = Acts::Vector3(l.meas_dir[0], l.meas_dir[1], 0);
            rot.col(1) = Acts::Vector3(l.wire_dir[0], l.wire_dir[1], 0);
            rot.col(2) = Acts::Vector3(0, 0, 1);
            auto bounds =
                std::make_shared<Acts::RectangleBounds>(l.half_width + 10., l.half_length + 10.);
            std::shared_ptr<const Acts::ISurfaceMaterial> material;
            if (config_.material) {
                double const t = mean_wall_thickness(l, config_.tube_wall);
                material = std::make_shared<const Acts::HomogeneousSurfaceMaterial>(
                    Acts::MaterialSlab(aluminium, static_cast<float>(t * Acts::UnitConstants::mm)));
            }
            for (auto const& t : l.tubes) {
                Acts::Transform3 trf = Acts::Transform3::Identity();
                trf.translate(Acts::Vector3(l.centre[0], l.centre[1], t.z));
                trf.rotate(rot);
                auto surface = Acts::Surface::makeShared<Acts::PlaneSurface>(trf, bounds);
                if (material) {
                    surface->assignSurfaceMaterial(material);
                }
                surfaces_.emplace(t.id, std::move(surface));
            }
            Acts::Transform3 trf = Acts::Transform3::Identity();
            trf.translate(Acts::Vector3(l.centre[0], l.centre[1], l.z));
            trf.rotate(rot);
            // wider than the tubes: a track may cross the layer plane beside them
            auto plane = Acts::Surface::makeShared<Acts::PlaneSurface>(
                trf,
                std::make_shared<Acts::RectangleBounds>(l.half_width + 500., l.half_length + 500.));
            if (material) {
                plane->assignSurfaceMaterial(material);
            }
            layer_surfaces_.push_back(std::move(plane));
        }
        Stepper stepper(std::move(field));
        Acts::DirectNavigator navigator{
            Acts::getDefaultLogger("MufluxDirectNavigator", Acts::Logging::WARNING)};
        Propagator propagator(std::move(stepper), std::move(navigator),
                              Acts::getDefaultLogger("MufluxPropagator", Acts::Logging::WARNING));
        fitter_ = std::make_unique<Fitter>(
            std::move(propagator), Acts::getDefaultLogger("MufluxKalman", Acts::Logging::WARNING));
    }

    /// The plane surface of a tube.
    Acts::Surface const& surface(std::int32_t tube) const { return *surfaces_.at(tube); }

    /// Fit a matched candidate; its hit sides are updated from the fit.
    FitOutput fit(std::vector<Hit> const& hits, Candidate& cand, Patrec const& patrec) const {
        FitOutput out;
        out.residuals.assign(cand.hits.size(), std::numeric_limits<double>::quiet_NaN());
        if (!cand.matched || cand.p <= 0 || cand.charge == 0 || cand.hits.size() < 5) {
            return out;
        }
        // per candidate hit: track position and slopes at its layer, from the
        // pattern recognition first, then from the smoothed fit
        std::vector<std::pair<Vec2, Vec2>> pred(cand.hits.size());
        for (std::size_t k = 0; k < cand.hits.size(); ++k) {
            auto const& h = hits[cand.hits[k].hit];
            bool const behind = geometry_.layers()[h.layer].station >= 3;
            pred[k] = {cand.position(h.z, behind), cand.slopes(behind)};
        }
        for (int pass = 0; pass <= config_.lr_passes; ++pass) {
            // the first fit uses every hit: a poor starting momentum can make good
            // hits look like outliers; the refits apply the outlier cut
            auto result = fit_once(hits, cand, pred, pass > 0);
            if (!result.output.ok) {
                return pass == 0 ? result.output : out;
            }
            out = std::move(result.output);
            if (pass == config_.lr_passes) {
                break;
            }
            // choose the sides again from the smoothed states
            for (std::size_t k = 0; k < cand.hits.size(); ++k) {
                if (result.smoothed[k]) {
                    pred[k] = *result.smoothed[k];
                }
            }
            std::size_t idx = 0;
            int const changed =
                patrec.reassign_signs(hits, cand, [&](Assigned const&) { return pred[idx++]; });
            out.lr_changes += changed;
            if (changed == 0 && pass > 0) {
                break;
            }
        }
        return out;
    }

   private:
    using Stepper = Acts::EigenStepper<>;
    using Propagator = Acts::Propagator<Stepper, Acts::DirectNavigator>;
    using Trajectory = Acts::VectorMultiTrajectory;
    using Fitter = Acts::KalmanFitter<Propagator, Trajectory>;

    struct Measurement {
        Acts::Surface const* surface = nullptr;
        double loc0 = 0;
        double variance = 0;
        std::size_t cand_hit = 0;  // index in Candidate::hits
    };

    struct Measurements {
        std::vector<Measurement> list;

        Acts::Surface const* surface(Acts::SourceLink const& sl) const {
            return list[sl.get<MufluxSourceLink>().index].surface;
        }

        template <typename trajectory_t>
        void calibrate(Acts::GeometryContext const&, Acts::CalibrationContext const&,
                       Acts::SourceLink const& sl,
                       typename trajectory_t::TrackStateProxy ts) const {
            auto const& m = list[sl.get<MufluxSourceLink>().index];
            ts.setUncalibratedSourceLink(Acts::SourceLink{sl});
            ts.allocateCalibrated(1);
            ts.template calibrated<1>()(0) = m.loc0;
            ts.template calibratedCovariance<1>()(0, 0) = m.variance;
            ts.setProjectorSubspaceIndices(std::array{Acts::eBoundLoc0});
        }
    };

    // A hit whose predicted chi2 exceeds the cut is not used in the filter
    // (typically the wrong side of its wire); its side is chosen again from
    // the smoothed track before the refit.
    struct OutlierFinder {
        double chi2_cut = 16;
        bool is_outlier(Trajectory::ConstTrackStateProxy ts) const {
            return Acts::calculatePredictedChi2(ts) > chi2_cut;
        }
    };

    struct Result {
        FitOutput output;
        std::vector<std::optional<std::pair<Vec2, Vec2>>> smoothed;  // per candidate hit
    };

    static Vec2 slopes_of(Acts::Vector3 const& dir) {
        return {dir.x() / dir.z(), dir.y() / dir.z()};
    }

    Result fit_once(std::vector<Hit> const& hits, Candidate const& cand,
                    std::vector<std::pair<Vec2, Vec2>> const& pred, bool reject_outliers) const {
        Result result;
        result.smoothed.resize(cand.hits.size());
        result.output.residuals.assign(cand.hits.size(), std::numeric_limits<double>::quiet_NaN());

        // measurements, ordered by z (one per layer)
        Measurements meas;
        std::map<int, Acts::Surface const*> measured_layer;  // layer -> its hit's surface
        for (std::size_t k = 0; k < cand.hits.size(); ++k) {
            auto const& a = cand.hits[k];
            auto const& h = hits[a.hit];
            auto const& l = geometry_.layers()[h.layer];
            auto const& [pos, sl] = pred[k];
            double const tm = l.meas_dir[0] * sl[0] + l.meas_dir[1] * sl[1];
            double const coord = Patrec::measured(h, a.sign, tm);
            double const centre = l.coord(l.centre[0], l.centre[1]);
            meas.list.push_back({.surface = surfaces_.at(h.tube).get(),
                                 .loc0 = coord - centre,
                                 .variance = config_.sigma * config_.sigma,
                                 .cand_hit = k});
            measured_layer[h.layer] = surfaces_.at(h.tube).get();
        }
        std::ranges::sort(meas.list, {}, [](Measurement const& m) {
            return m.surface->center(Acts::GeometryContext::dangerouslyDefaultConstruct()).z();
        });
        std::vector<Acts::SourceLink> source_links;
        for (std::size_t i = 0; i < meas.list.size(); ++i) {
            source_links.emplace_back(MufluxSourceLink{i});
        }
        // the surfaces in z order, from the first measured layer to the last:
        // the hit's tube surface where the layer has one, else (with material)
        // the layer's plane
        std::vector<Acts::Surface const*> sequence;
        int const first_layer = measured_layer.begin()->first;
        int const last_layer = measured_layer.rbegin()->first;
        for (int li = first_layer; li <= last_layer; ++li) {
            if (auto const it = measured_layer.find(li); it != measured_layer.end()) {
                sequence.push_back(it->second);
            } else if (config_.material) {
                sequence.push_back(layer_surfaces_[li].get());
            }
        }

        // start parameters on the first measured layer, from the candidate
        auto const& first = *meas.list.front().surface;
        auto const& first_hit = hits[cand.hits[meas.list.front().cand_hit].hit];
        bool const behind = geometry_.layers()[first_hit.layer].station >= 3;
        Vec2 const xy = cand.position(first_hit.z, behind);
        Vec2 const t = cand.slopes(behind);
        Acts::Vector3 const dir = Acts::Vector3(t[0], t[1], 1).normalized();
        Acts::Vector3 const global(xy[0], xy[1], first_hit.z);
        auto const local = first.globalToLocal(gctx_, global, dir, 1.0);
        if (!local.ok()) {
            return result;
        }
        Acts::BoundVector pars = Acts::BoundVector::Zero();
        pars[Acts::eBoundLoc0] = local.value()[0];
        pars[Acts::eBoundLoc1] = local.value()[1];
        pars[Acts::eBoundPhi] = std::atan2(dir.y(), dir.x());
        pars[Acts::eBoundTheta] = std::acos(dir.z());
        pars[Acts::eBoundQOverP] = cand.charge / (cand.p * Acts::UnitConstants::GeV);
        pars[Acts::eBoundTime] = 0;
        Acts::BoundMatrix cov = Acts::BoundMatrix::Zero();
        double const s_theta = std::max(std::sin(pars[Acts::eBoundTheta]), 1e-6);
        double const s_phi = std::min(config_.sigma_theta / s_theta, std::numbers::pi);
        cov(Acts::eBoundLoc0, Acts::eBoundLoc0) = std::pow(config_.sigma_loc0, 2);
        cov(Acts::eBoundLoc1, Acts::eBoundLoc1) = std::pow(config_.sigma_loc1, 2);
        cov(Acts::eBoundPhi, Acts::eBoundPhi) = s_phi * s_phi;
        cov(Acts::eBoundTheta, Acts::eBoundTheta) = std::pow(config_.sigma_theta, 2);
        cov(Acts::eBoundQOverP, Acts::eBoundQOverP) =
            std::pow(config_.sigma_qop_rel * pars[Acts::eBoundQOverP], 2);
        cov(Acts::eBoundTime, Acts::eBoundTime) = std::pow(1.0 * Acts::UnitConstants::ns, 2);
        Acts::BoundTrackParameters const start(first.getSharedPtr(), pars, cov,
                                               Acts::ParticleHypothesis::muon());

        Acts::GainMatrixUpdater updater;
        Acts::GainMatrixSmoother smoother;
        Acts::KalmanFitterExtensions<Trajectory> ext;
        ext.surfaceAccessor.template connect<&Measurements::surface>(&meas);
        ext.calibrator.template connect<&Measurements::calibrate<Trajectory>>(&meas);
        ext.updater.template connect<&Acts::GainMatrixUpdater::operator()<Trajectory>>(&updater);
        ext.smoother.template connect<&Acts::GainMatrixSmoother::operator()<Trajectory>>(&smoother);
        OutlierFinder const outliers{config_.outlier_chi2};
        if (reject_outliers) {
            ext.outlierFinder.template connect<&OutlierFinder::is_outlier>(&outliers);
        }

        Acts::MagneticFieldContext const mctx;
        Acts::CalibrationContext const cctx;
        Acts::PropagatorPlainOptions prop(gctx_, mctx);
        prop.maxSteps = 2000;
        prop.pathLimit = 20 * Acts::UnitConstants::m;
        Acts::KalmanFitterOptions<Trajectory> options(gctx_, mctx, std::cref(cctx), ext, prop,
                                                      &first);
        options.multipleScattering = config_.material;
        options.energyLoss = config_.material;
        options.referenceSurfaceStrategy = Acts::TrackExtrapolationStrategy::first;

        Acts::TrackContainer tracks(Acts::VectorTrackContainer{}, Trajectory{});
        auto fitted = fitter_->fit(source_links.begin(), source_links.end(), start, options,
                                   sequence, tracks);
        if (!fitted.ok()) {
            return result;
        }
        auto track = fitted.value();
        auto& out = result.output;
        out.ok = true;
        out.chi2 = track.chi2();
        // ACTS counts measurement dimensions; subtract the 5 track parameters
        out.ndf = static_cast<int>(track.nDoF()) - 5;
        double const qop = track.parameters()[Acts::eBoundQOverP];
        out.charge = qop > 0 ? 1 : -1;
        out.p = std::abs(1 / qop) / Acts::UnitConstants::GeV;
        out.sigma_p = out.p * out.p *
                      std::sqrt(track.covariance()(Acts::eBoundQOverP, Acts::eBoundQOverP)) *
                      Acts::UnitConstants::GeV;

        double zmin = std::numeric_limits<double>::max();
        double zmax = std::numeric_limits<double>::lowest();
        tracks.trackStateContainer().visitBackwards(track.tipIndex(), [&](auto ts) {
            bool const outlier = ts.typeFlags().isOutlier();
            if (!(ts.typeFlags().isMeasurement() || outlier) || !ts.hasSmoothed()) {
                return true;
            }
            auto const sl = ts.getUncalibratedSourceLink().template get<MufluxSourceLink>();
            auto const& m = meas.list[sl.index];
            auto const to_global = [&](Acts::BoundVector const& v) {
                double const phi = v[Acts::eBoundPhi], theta = v[Acts::eBoundTheta];
                Acts::Vector3 const d(std::sin(theta) * std::cos(phi),
                                      std::sin(theta) * std::sin(phi), std::cos(theta));
                Acts::Vector3 const pos = ts.referenceSurface().localToGlobal(
                    gctx_, Acts::Vector2{v[Acts::eBoundLoc0], v[Acts::eBoundLoc1]}, d);
                return std::pair{pos, d};
            };
            Acts::BoundVector const s = ts.smoothed();
            auto const [pos, d] = to_global(s);
            // the track without this hit: for its residual and its side
            Acts::BoundVector const u =
                outlier
                    ? s
                    : Acts::calculateUnbiasedParametersCovariance(Acts::AnyConstTrackStateProxy{ts})
                          .first;
            auto const [upos, ud] = to_global(u);
            double const pabs = std::abs(1 / s[Acts::eBoundQOverP]) / Acts::UnitConstants::GeV;
            if (!outlier) {
                ++out.n_meas;
            } else {
                ++out.n_outliers;
            }
            out.residuals[m.cand_hit] = m.loc0 - u[Acts::eBoundLoc0];
            result.smoothed[m.cand_hit] = std::pair{Vec2{upos.x(), upos.y()}, slopes_of(ud)};
            if (pos.z() < zmin) {
                zmin = pos.z();
                out.first = {pos, pabs * d};
            }
            if (pos.z() > zmax) {
                zmax = pos.z();
                out.last = {pos, pabs * d};
            }
            return true;
        });
        return result;
    }

    Geometry const& geometry_;
    Config config_;
    Acts::GeometryContext gctx_;
    std::map<std::int32_t, std::shared_ptr<Acts::PlaneSurface>> surfaces_;  // per tube
    std::vector<std::shared_ptr<Acts::PlaneSurface>> layer_surfaces_;       // per layer
    std::unique_ptr<Fitter> fitter_;
};

}  // namespace SHiP::trout::muflux
