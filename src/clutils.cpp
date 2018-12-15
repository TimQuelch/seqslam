#include "clutils.h"

#include <numeric>

#include <fmt/format.h>
#include <fstream>

namespace clutils {
    namespace {
        [[nodiscard]] auto toNdRange(std::vector<std::size_t> const& range) -> cl::NDRange {
            if (range.size() == 1) {
                return cl::NDRange(range[0]);
            }
            if (range.size() == 2) {
                return cl::NDRange(range[0], range[1]);
            }
            if (range.size() == 3) {
                return cl::NDRange(range[0], range[1], range[2]);
            }
            throw std::runtime_error{fmt::format(
                "Invalid number of dimensions ({}) for NDRange. Must be between 1 and 3",
                range.size())};
        }

        [[nodiscard]] auto toClAccess(Buffer::Access val) noexcept -> cl_mem_flags {
            switch (val) {
            case Buffer::Access::read:
                return CL_MEM_READ_ONLY;
            case Buffer::Access::write:
                return CL_MEM_WRITE_ONLY;
            case Buffer::Access::readwrite:
                return CL_MEM_READ_WRITE;
            }
            return {};
        }

        [[nodiscard]] auto clErrorToException(cl::Error const& e) noexcept {
            return std::runtime_error{
                fmt::format("OpenCL error in {}. Error code {}", e.what(), e.err())};
        }
    } // namespace

    namespace detail {
        [[nodiscard]] auto compileClSource(std::filesystem::path const& sourceFile,
                                           cl::Context& context,
                                           std::vector<cl::Device> const& devices) -> cl::Program {
            auto const program = [&context, &sourceFile]() {
                auto filestream = std::ifstream{sourceFile};
                auto const sourceString = std::string{std::istreambuf_iterator<char>{filestream},
                                                      std::istreambuf_iterator<char>{}};
                auto const source = cl::Program::Sources{
                    std::pair{sourceString.c_str(), std::strlen(sourceString.c_str())}};
                return cl::Program{context, source};
            }();

            program.build(devices, "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros");

            return program;
        }
    } // namespace detail

    Buffer::Buffer(Context const& context, std::size_t size, Access access)
        : size_{size}, buffer_{context.context(), toClAccess(access) | CL_MEM_ALLOC_HOST_PTR, size},
          queue_{context.queue()} {}

    void Buffer::readBuffer(void* destination) const { readBuffer(destination, 0, size_); }

    void Buffer::readBuffer(void* destination, std::size_t offset, std::size_t size) const {
        try {
            queue_.enqueueReadBuffer(
                buffer_, static_cast<cl_bool>(true), offset, size, destination);
        } catch (cl::Error& e) { throw clErrorToException(e); }
    }

    void Buffer::writeBuffer(void const* source) const { writeBuffer(source, 0, size_); }

    void Buffer::writeBuffer(void const* source, std::size_t offset, std::size_t size) const {
        try {
            queue_.enqueueWriteBuffer(buffer_, static_cast<cl_bool>(true), offset, size, source);
        } catch (cl::Error& e) { throw clErrorToException(e); }
    }

    Context::Context() : Context{0, 0} {}

    Context::Context(unsigned platformId, unsigned deviceId) {
        // Query the available platforms and get specified platform
        auto const platforms = []() {
            auto p = std::vector<cl::Platform>{};
            cl::Platform::get(&p);
            return p;
        }();

        if (platformId > platforms.size() - 1) {
            throw std::runtime_error{"Requested OpenCL platform does not exist"};
        }

        auto const platform = platforms[platformId];

        auto const allDevices = [&platform]() {
            auto d = std::vector<cl::Device>{};
            platform.getDevices(CL_DEVICE_TYPE_ALL, &d);
            return d;
        }();

        if (deviceId > allDevices.size() - 1) {
            throw std::runtime_error{"Requested OpenCL device does not exist"};
        }

        auto const device = allDevices[deviceId];
        context_ = cl::Context{device};
        queue_ = cl::CommandQueue{context_};
        devices_.push_back(device);
    }

    void Context::runKernel(std::string_view kernelName,
                            std::vector<std::size_t> const& globalDims,
                            std::vector<std::size_t> const& localDims) const {
        constexpr auto maxWorkgroupSize = 1024u;
        constexpr auto maxZDimSize = 64u;
        if (auto s = std::accumulate(
                localDims.begin(), localDims.end(), 1u, std::multiplies<std::size_t>{});
            s > maxWorkgroupSize) {
            throw std::runtime_error{fmt::format(
                "Workgroup size of {} requested. Max allowable is {}", s, maxWorkgroupSize)};
        }
        if (localDims.size() >= 3 && localDims[2] > maxZDimSize) {
            throw std::runtime_error{
                fmt::format("Workgroup z dimension size of {} requested. Max allowable is {}",
                            localDims[2],
                            maxZDimSize)};
        }

        try {
            queue_.enqueueNDRangeKernel(kernels_.at(std::string{kernelName}),
                                        {0, 0, 0},
                                        toNdRange(globalDims),
                                        toNdRange(localDims));
            queue_.finish();
        } catch (cl::Error& e) { throw clErrorToException(e); }
    }

    void Context::setKernelArg(std::string_view kernelName, unsigned index, Buffer const& arg) {
        try {
            kernels_.at(std::string{kernelName}).setArg(index, arg.buffer());
        } catch (cl::Error& e) { throw clErrorToException(e); }
    }

    void Context::setKernelLocalArg(std::string_view kernelName, unsigned index, std::size_t size) {
        try {
            kernels_.at(std::string{kernelName}).setArg(index, cl::Local(size));
        } catch (cl::Error& e) { throw clErrorToException(e); }
    };
} // namespace clutils
