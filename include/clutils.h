#ifndef CLUTILS_CLUTILS_H
#define CLUTILS_CLUTILS_H

#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace clutils {
    class Context;
    namespace detail {
        [[nodiscard]] auto compileClSource(std::filesystem::path const& sourceFile,
                                           cl::Context& context,
                                           std::vector<cl::Device> const& devices) -> cl::Program;
    } // namespace detail

    class Error : std::exception {
    public:
        Error(std::string_view where, int code);

        [[nodiscard]] virtual auto what() const noexcept -> char const* override {
            return what_.c_str();
        }
        [[nodiscard]] auto where() const noexcept { return where_; }
        [[nodiscard]] auto error() const noexcept { return error_; }
        [[nodiscard]] auto code() const noexcept { return code_; }

    private:
        static std::map<int, std::string> errorCodes_;

        std::string what_;
        std::string where_;
        std::string error_;
        int code_;
    };

    class Buffer {
    public:
        enum class Access { read, write, readwrite };

        Buffer(Context const& context, std::size_t size, Access access);

        void readBuffer(void* destination) const;
        void readBuffer(void* destination, std::size_t offset, std::size_t size) const;
        void writeBuffer(void const* source) const;
        void writeBuffer(void const* source, std::size_t offset, std::size_t size) const;

        [[nodiscard]] auto& buffer() noexcept { return buffer_; }
        [[nodiscard]] auto const& buffer() const noexcept { return buffer_; }

        [[nodiscard]] auto size() const noexcept { return size_; }

    private:
        std::size_t size_;
        cl::Buffer buffer_;
        cl::CommandQueue const& queue_;
    };

    class Context {
    public:
        Context();
        Context(std::filesystem::path const& configFile);
        Context(unsigned platformId, unsigned deviceId);

        template <typename StringContainer>
        void addKernels(std::filesystem::path const& sourceFile,
                        StringContainer const& kernelNames) {
            auto const program = detail::compileClSource(sourceFile, context_, devices_);
            for (auto name : kernelNames) {
                try {
                    kernels_.insert(
                        {std::string{name}, cl::Kernel{program, std::string{name}.c_str()}});
                } catch (cl::Error& e) { throw Error(e.what(), e.err()); }
            }
        }

        void runKernel(std::string_view kernelName,
                       std::vector<std::size_t> const& globalDims,
                       std::vector<std::size_t> const& localDims) const;

        template <typename T>
        void setKernelArg(std::string_view kernelName, unsigned index, T const& arg) {
            kernels_.at(std::string{kernelName}).setArg(index, arg);
        }

        void setKernelArg(std::string_view kernelName, unsigned index, Buffer const& arg);

        void setKernelLocalArg(std::string_view kernelName, unsigned index, std::size_t size);

        [[nodiscard]] auto& context() noexcept { return context_; }
        [[nodiscard]] auto const& context() const noexcept { return context_; }
        [[nodiscard]] auto& queue() noexcept { return queue_; }
        [[nodiscard]] auto const& queue() const noexcept { return queue_; }
        [[nodiscard]] auto& kernel(std::string const& name) noexcept { return kernels_.at(name); };
        [[nodiscard]] auto const& kernel(std::string const& name) const noexcept {
            return kernels_.at(name);
        };

    private:
        cl::Context context_;
        cl::CommandQueue queue_;
        std::vector<cl::Device> devices_;
        std::unordered_map<std::string, cl::Kernel> kernels_;
    };
} // namespace clutils

#endif
