#ifndef NATIVE_MODULES_COMMON_SRC_PY_OUTPUT_ARRAY
#define NATIVE_MODULES_COMMON_SRC_PY_OUTPUT_ARRAY

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mdi::common {

template<typename ElementT, size_t DataDimension>
class py_output_array
{
    using py_array_t = py::array_t<ElementT, py::array::c_style>;
    py_array_t _data;
    decltype(_data.template mutable_unchecked<2>()) _mut_ref = _data.template mutable_unchecked<2>();
    py::ssize_t _write_index = 0;

public:
    py_output_array(size_t size) : _data(py_array_t({size, DataDimension})) {}

    void push_back(const ElementT* const item_ptr) {
        std::memcpy(_mut_ref.mutable_data(_write_index, 0), item_ptr, DataDimension * sizeof(ElementT));
        _write_index++;
    }

    auto push_back(ElementT item) requires (DataDimension == 1) { push_back(&item); }

    void push_back(const std::array<ElementT, DataDimension>& item) { push_back(item.data()); }

    py::array_t<ElementT> finalize() && {
        _data.resize({_write_index, static_cast<py::ssize_t>(DataDimension)});
        return std::move(_data);
    }
};

} // namespace mdi::common

#endif /* NATIVE_MODULES_COMMON_SRC_PY_OUTPUT_ARRAY */
