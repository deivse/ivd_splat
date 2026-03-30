#ifndef NATIVE_MODULES_SUBSAMPLING_SRC_UTIL
#define NATIVE_MODULES_SUBSAMPLING_SRC_UTIL

#include <fstream>

namespace mdi::common {

inline void dump_str_to_file(const std::string& filename, const std::string& content) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << content;
        file.close();
    } else {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
}
} // namespace mdi::common

#endif /* NATIVE_MODULES_SUBSAMPLING_SRC_UTIL */
