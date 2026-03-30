from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.build import check_min_cppstd

required_conan_version = ">=2.0"


class NativeModules(ConanFile):
    name = "monodepth_native_modules"
    version = "0.1.0"
    package_type = "shared-library"

    settings = "os", "arch", "compiler", "build_type"
    options = {"fPIC": [True, False]}
    default_options = {"fPIC": True}

    exports_sources = "CMakeLists.txt", "src/*", "include/*"

    build_requires = ["cmake/3.26.4", "ninja/1.11.1"]
    requires = [
        "eigen/3.4.0",
        "pybind11/3.0.1",
        "fmt/10.1.1",
    ]

    def configure(self):
        check_min_cppstd(self, "20")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self, generator="Ninja")
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
