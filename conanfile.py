from conans import ConanFile

class SeqslamBenchmarkConan(ConanFile):
    requires = ("fmt/5.3.0@bincrafters/stable",
                "opencv/3.4.5@conan/stable",
                "eigen/3.3.7@conan/stable",
                "Catch2/2.7.1@catchorg/stable",
                "jsonformoderncpp/3.6.1@vthiery/stable")
    default_options = {"opencv:openexr": False,
                       "opencv:jasper": False}
    generators = ("cmake", "cmake_paths")
