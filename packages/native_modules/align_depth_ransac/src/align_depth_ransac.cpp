#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <common/py_output_array.h>
#include <random>

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace mdi::pointcloud {

template<typename T>
using c_style_pyarray_t = py::array_t<T, py::array::c_style | py::array::forcecast>;

struct model_t
{
    float scale;
    float offset;
};

model_t align_depth_least_squares(const c_style_pyarray_t<float>& depth, const c_style_pyarray_t<float>& gt_depth,
                                  const std::vector<size_t>& sample_indices) {
    float* depth_ptr = static_cast<float*>(depth.request().ptr);
    float* gt_ptr = static_cast<float*>(gt_depth.request().ptr);

    size_t num_samples = sample_indices.size();
    constexpr size_t depth_rows = 2; // depth_with_ones has shape (2, N)

    Eigen::MatrixXf depth_selected(depth_rows, num_samples);
    Eigen::VectorXf gt_selected(num_samples);

    // Fill matrices with selected samples
    for (size_t i = 0; i < num_samples; ++i) {
        size_t idx = sample_indices[i];
        depth_selected(0, i) = depth_ptr[idx];
        depth_selected(1, i) = 1.0f; // for offset
        gt_selected(i) = gt_ptr[idx];
    }

    Eigen::Matrix2f A = depth_selected * depth_selected.transpose();
    Eigen::Vector2f b = depth_selected * gt_selected;
    Eigen::Vector2f h = A.completeOrthogonalDecomposition().solve(b);

    return {h(0), h(1)};
}

float ransac_loss(const std::vector<float>& dists, float inlier_threshold) {
    unsigned int loss = 0;
    for (const auto& dist : dists) {
        if (dist >= inlier_threshold) {
            loss++;
        }
    }
    return static_cast<float>(loss);
}

float msac_loss(const std::vector<float>& dists, float inlier_threshold) {
    float loss = 0.0f;
    for (const auto& dist : dists) {
        loss += std::min(dist, inlier_threshold);
    }
    return loss;
}

size_t required_samples(size_t num_inliers, size_t total_samples, size_t sample_size, float confidence) {
    if (num_inliers == 0) {
        return std::numeric_limits<size_t>::max();
    }
    float inlier_ratio = static_cast<float>(num_inliers) / static_cast<float>(total_samples);
    float p_no_outliers = 1.0f - std::pow(inlier_ratio, static_cast<float>(sample_size));
    p_no_outliers = std::clamp(p_no_outliers, 1e-8f, 1.0f - 1e-8f); // avoid log(0)

    return static_cast<size_t>(std::ceil(std::log(1.0f - confidence) / std::log(p_no_outliers)));
}

py::array_t<size_t> vec_to_array(const std::vector<size_t>& vec) {
    py::array_t<size_t> array(vec.size());
    auto array_ptr = static_cast<size_t*>(array.request().ptr);
    for (size_t i = 0; i < vec.size(); ++i) {
        array_ptr[i] = vec[i];
    }
    return array;
}

std::tuple<float, float, size_t, size_t, size_t>
  align_depth_ransac(c_style_pyarray_t<float> depth, c_style_pyarray_t<float> gt_depth, bool use_msac_loss,
                     size_t sample_size, size_t max_iters, float inlier_threshold, float confidence, size_t min_iters) {
    auto num_samples = depth.shape(0);

    std::optional<model_t> h_best_lo = std::nullopt;
    std::vector<size_t> inlier_indices_best_lo;
    size_t num_inliers_best_lo = 0;

    size_t num_inliers_best_pre_lo = 0;
    float loss_best_lo = std::numeric_limits<float>::infinity();
    float loss_best_sample = std::numeric_limits<float>::infinity();

    std::mt19937 engine(std::random_device{}());
    std::vector<size_t> all_sample_indices(num_samples);
    std::iota(all_sample_indices.begin(), all_sample_indices.end(), 0);

    std::vector<size_t> curr_sample_indices;
    curr_sample_indices.resize(sample_size);
    std::vector<float> curr_sample_dists;
    curr_sample_dists.resize(num_samples);
    std::vector<size_t> inlier_indices_sample;
    std::vector<size_t> inlier_indices_lo;

    const auto update_sample_indices = [&]() {
        std::sample(all_sample_indices.begin(), all_sample_indices.end(), curr_sample_indices.begin(), sample_size,
                    engine);
    };

    const auto update_curr_sample_dists = [&, depth_ptr = static_cast<float*>(depth.request().ptr),
                                           gt_ptr = static_cast<float*>(gt_depth.request().ptr)](const model_t& h) {
        for (size_t i = 0; i < num_samples; ++i) {
            float diff = h.scale * depth_ptr[i] + h.offset - gt_ptr[i];
            curr_sample_dists[i] = diff * diff;
        }
    };

    const auto calc_inlier_indices
      = [&](const std::vector<float>& dists, float threshold, std::vector<size_t>& inlier_indices_out) {
            inlier_indices_out.clear();
            for (size_t i = 0; i < num_samples; ++i) {
                if (dists[i] < threshold) {
                    inlier_indices_out.push_back(i);
                }
            }
        };

    const auto loss_func = use_msac_loss ? &msac_loss : *ransac_loss;

    size_t iteration = 0;
    for (; iteration < max_iters; ++iteration) {
        update_sample_indices();

        auto h_sample = align_depth_least_squares(depth, gt_depth, curr_sample_indices);
        update_curr_sample_dists(h_sample);

        float loss_sample = loss_func(curr_sample_dists, inlier_threshold);

        if (loss_sample < loss_best_sample) {
            calc_inlier_indices(curr_sample_dists, inlier_threshold, inlier_indices_sample);
            auto h_lo = align_depth_least_squares(depth, gt_depth, inlier_indices_sample);

            update_curr_sample_dists(h_lo);
            float loss_lo = loss_func(curr_sample_dists, inlier_threshold);
            calc_inlier_indices(curr_sample_dists, inlier_threshold, inlier_indices_lo);

            if (loss_lo < loss_best_lo) {
                h_best_lo = h_lo;
                loss_best_lo = loss_lo;
                loss_best_sample = loss_sample;
                inlier_indices_best_lo = inlier_indices_lo;
                num_inliers_best_lo = inlier_indices_lo.size();
                num_inliers_best_pre_lo = inlier_indices_sample.size();
            }
        }

        if (required_samples(num_inliers_best_lo, num_samples, sample_size, confidence) <= iteration
            && h_best_lo.has_value() && iteration >= min_iters) {
            break;
        }
    }
    if (!h_best_lo.has_value()) {
        throw std::runtime_error("RANSAC failed to find a valid model.");
    }

    return std::tuple{h_best_lo->scale, h_best_lo->offset, num_inliers_best_lo, num_inliers_best_pre_lo, iteration};
}

PYBIND11_MODULE(_align_depth_ransac, m) {
    m.doc() = R"pbdoc(
        RANSAC depth alignment native module
        -----------------------

        .. currentmodule:: align_depth_ransac

        .. autosummary::
           :toctree: _generate

           TSDFIntegrator   
    )pbdoc";

    m.def("align_depth_ransac", &align_depth_ransac, py::arg("depth"), py::arg("gt_depth"),
          py::arg("use_msac_loss") = false, py::arg("sample_size") = 3, py::arg("max_iters"),
          py::arg("inlier_threshold"), py::arg("confidence"), py::arg("min_iters"));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

} // namespace mdi::pointcloud
