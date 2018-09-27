#include "clutils.h"

#include <fmt/format.h>
#include <fstream>

namespace clutils {
    namespace {
        auto toNdRange(std::vector<std::size_t> const& range) -> cl::NDRange {
            if (range.size() == 1) {
                return cl::NDRange(range[0]);
            } else if (range.size() == 2) {
                return cl::NDRange(range[0], range[1]);
            } else if (range.size() == 3) {
                return cl::NDRange(range[0], range[1], range[2]);
            } else {
                throw std::runtime_error{fmt::format(
                    "Invalid number of dimensions ({}) for NDRange. Must be between 1 and 3",
                    range.size())};
            }
        }

        auto toClAccess(Buffer::Access val) -> cl_mem_flags {
            switch (val) {
            case Buffer::Access::read:
                return CL_MEM_READ_ONLY;
            case Buffer::Access::write:
                return CL_MEM_WRITE_ONLY;
            case Buffer::Access::readwrite:
                return CL_MEM_READ_WRITE;
            }
        }
    } // namespace

    Buffer::Buffer(Context const& context, std::size_t size, Access access)
        : size_{size}, buffer_{context.context(), toClAccess(access) | CL_MEM_ALLOC_HOST_PTR, size},
          queue_{context.queue()} {}

    void Buffer::readBuffer(void* destination) const { readBuffer(destination, 0, size_); }

    void Buffer::readBuffer(void* destination, std::size_t offset, std::size_t size) const {
        queue_.enqueueReadBuffer(buffer_, true, offset, size, destination);
    }

    void Buffer::writeBuffer(void const* source) const { writeBuffer(source, 0, size_); }

    void Buffer::writeBuffer(void const* source, std::size_t offset, std::size_t size) const {
        queue_.enqueueWriteBuffer(buffer_, true, offset, size, source);
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

    void Context::addKernels(std::filesystem::path const& sourceFile,
                             std::vector<std::string> const& kernelNames) {
        auto const program = [this, &sourceFile]() {
            auto filestream = std::ifstream{sourceFile};
            auto const sourceString = std::string{std::istreambuf_iterator<char>{filestream},
                                                  std::istreambuf_iterator<char>{}};
            auto const source = cl::Program::Sources{
                std::pair{sourceString.c_str(), std::strlen(sourceString.c_str())}};
            return cl::Program{context_, source};
        }();

        program.build(devices_);

        for (auto name : kernelNames) {
            kernels_.insert({name, cl::Kernel{program, name.c_str()}});
        }
    }

    void Context::runKernel(std::string const& kernelName,
                            std::vector<std::size_t> const& globalDims,
                            std::vector<std::size_t> const& localDims) const {
        queue_.enqueueNDRangeKernel(
            kernels_.at(kernelName), {0, 0, 0}, toNdRange(globalDims), toNdRange(localDims));
    }

    void Context::setKernelArg(std::string const& kernelName, unsigned index, Buffer const& arg) {
        kernels_.at(kernelName).setArg(index, arg.buffer());
    }

    void
    Context::setKernelLocalArg(std::string const& kernelName, unsigned index, std::size_t size) {
        kernels_.at(kernelName).setArg(index, size, NULL);
    };
} // namespace clutils
