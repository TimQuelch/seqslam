from conans import ConanFile

class SeqslamBenchmarkConan(ConanFile):
    requires = ("fmt/5.2.1@bincrafters/stable",
                "opencv/3.4.3@conan/stable",
                "eigen/3.3.5@conan/stable")
    default_options = {"opencv:openexr": False}
    generators = ("cmake", "cmake_paths")
