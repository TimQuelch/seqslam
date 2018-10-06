#ifndef CLUTILS_CLUTILS_H
#define CLUTILS_CLUTILS_H

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace clutils {
    class Context;

    class Buffer {
    public:
        enum class Access { read, write, readwrite };

        Buffer(Context const& context, std::size_t size, Access access);

        void readBuffer(void* destination) const;
        void readBuffer(void* destination, std::size_t offset, std::size_t size) const;
        void writeBuffer(void const* source) const;
        void writeBuffer(void const* source, std::size_t offset, std::size_t size) const;

        auto& buffer() noexcept { return buffer_; }
        auto const& buffer() const noexcept { return buffer_; }

        auto size() const noexcept { return size_; }

    private:
        std::size_t size_;
        cl::Buffer buffer_;
        cl::CommandQueue const& queue_;
    };

    class Context {
    public:
        Context();
        Context(unsigned platformId, unsigned deviceId);

        void addKernels(std::filesystem::path const& sourceFile,
                        std::vector<std::string> const& kernelNames);

        void runKernel(std::string const& kernelName,
                       std::vector<std::size_t> const& globalDims,
                       std::vector<std::size_t> const& localDims) const;

        template <typename T>
        void setKernelArg(std::string const& kernelName, unsigned index, T const& arg) {
            kernels_.at(kernelName).setArg(index, arg);
        }

        void setKernelArg(std::string const& kernelName, unsigned index, Buffer const& arg);

        void setKernelLocalArg(std::string const& kernelName, unsigned index, std::size_t size);

        auto& context() noexcept { return context_; }
        auto const& context() const noexcept { return context_; }
        auto& queue() noexcept { return queue_; }
        auto const& queue() const noexcept { return queue_; }
        auto& kernel(std::string const& name) noexcept { return kernels_.at(name); };
        auto const& kernel(std::string const& name) const noexcept { return kernels_.at(name); };

    private:
        cl::Context context_;
        cl::CommandQueue queue_;
        std::vector<cl::Device> devices_;
        std::unordered_map<std::string, cl::Kernel> kernels_;
    };
} // namespace clutils

#endif
