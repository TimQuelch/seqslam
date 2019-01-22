from conans import ConanFile

class SeqslamBenchmarkConan(ConanFile):
    requires = ("fmt/5.3.0@bincrafters/stable",
                "opencv/3.4.5@conan/stable",
                "eigen/3.3.5@conan/stable",
                "catch2/2.5.0@bincrafters/stable",
                "jsonformoderncpp/3.5.0@vthiery/stable")
    default_options = {"opencv:openexr": False}
    generators = ("cmake", "cmake_paths")
