#include "clutils.h"

#include "utils.h"

#include <fstream>
#include <numeric>
#include <stdexcept>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

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

        [[nodiscard]] auto getSpecifiedDevice(unsigned platformId, unsigned deviceId) {
            auto const platforms = []() {
                auto p = std::vector<cl::Platform>{};
                cl::Platform::get(&p);
                return p;
            }();

            if (platformId > platforms.size() - 1) {
                throw std::runtime_error{fmt::format("Requested OpenCL platform does not exist. "
                                                     "Requested index {}, when only {} available",
                                                     platformId,
                                                     platforms.size())};
            }

            auto const platform = platforms[platformId];

            auto const allDevices = [&platform]() {
                auto d = std::vector<cl::Device>{};
                platform.getDevices(CL_DEVICE_TYPE_ALL, &d);
                return d;
            }();

            if (deviceId > allDevices.size() - 1) {
                throw std::runtime_error{fmt::format("Requested OpenCL device does not exist. "
                                                     "Requested index {}, when only {} available",
                                                     deviceId,
                                                     allDevices.size())};
            }
            return allDevices[deviceId];
        }

        [[nodiscard]] auto constructContextQueueDevices(unsigned platformId, unsigned deviceId) {
            auto device = getSpecifiedDevice(platformId, deviceId);
            try {
                auto context = cl::Context{device};
                auto queue = cl::CommandQueue{context};
                auto devices = std::vector<cl::Device>{device};
                return std::tuple{context, queue, std::move(devices)};
            } catch (cl::Error& e) { throw Error(e.what(), e.err()); }
        }

        [[nodiscard]] auto readJsonConfig(std::filesystem::path const& configFile) {
            auto stream = std::ifstream{configFile};
            auto json = nlohmann::json{};
            stream >> json;
            return json;
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
        } catch (cl::Error& e) { throw Error(e.what(), e.err()); }
    }

    void Buffer::writeBuffer(void const* source) const { writeBuffer(source, 0, size_); }

    void Buffer::writeBuffer(void const* source, std::size_t offset, std::size_t size) const {
        try {
            queue_.enqueueWriteBuffer(buffer_, static_cast<cl_bool>(true), offset, size, source);
        } catch (cl::Error& e) { throw Error(e.what(), e.err()); }
    }

    Context::Context() {
        auto const defaultConfigPath = utils::traverseUpUntilMatch(defaultConfig);
        if (defaultConfigPath && std::filesystem::exists(*defaultConfigPath)) {
            auto const config = readJsonConfig(*defaultConfigPath);
            std::tie(context_, queue_, devices_) = constructContextQueueDevices(
                config["platform"].get<unsigned>(), config["device"].get<unsigned>());
        } else {
            std::tie(context_, queue_, devices_) = constructContextQueueDevices(0, 0);
        }
    }

    Context::Context(std::filesystem::path const& configFile) {
        if (!std::filesystem::exists(configFile)) {
            throw std::runtime_error{
                fmt::format("Specified config file ({}) does not exist", configFile.string())};
        }
        auto const config = readJsonConfig(configFile);

        std::tie(context_, queue_, devices_) = constructContextQueueDevices(
            config["platform"].get<unsigned>(), config["device"].get<unsigned>());
    }

    Context::Context(unsigned platformId, unsigned deviceId) {
        std::tie(context_, queue_, devices_) = constructContextQueueDevices(platformId, deviceId);
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
        } catch (cl::Error& e) { throw Error(e.what(), e.err()); }
    }

    void Context::setKernelArg(std::string_view kernelName, unsigned index, Buffer const& arg) {
        try {
            kernels_.at(std::string{kernelName}).setArg(index, arg.buffer());
        } catch (cl::Error& e) { throw Error(e.what(), e.err()); }
    }

    void Context::setKernelLocalArg(std::string_view kernelName, unsigned index, std::size_t size) {
        try {
            kernels_.at(std::string{kernelName}).setArg(index, cl::Local(size));
        } catch (cl::Error& e) { throw Error(e.what(), e.err()); }
    }

    Error::Error(std::string_view where, int code)
        : what_{fmt::format(
              "OpenCL error in {}. Error code {} ({})", where, code, errorCodes_[code])},
          where_{where}, error_{errorCodes_[code]}, code_{code} {
        fmt::print("{}\n", what_);
    }

    std::map<int, std::string> Error::errorCodes_ = {
        {0, "CL_SUCCESS"},
        {-1, "CL_DEVICE_NOT_FOUND"},
        {-2, "CL_DEVICE_NOT_AVAILABLE"},
        {-3, "CL_COMPILER_NOT_AVAILABLE"},
        {-4, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
        {-5, "CL_OUT_OF_RESOURCES"},
        {-6, "CL_OUT_OF_HOST_MEMORY"},
        {-7, "CL_PROFILING_INFO_NOT_AVAILABLE"},
        {-8, "CL_MEM_COPY_OVERLAP"},
        {-9, "CL_IMAGE_FORMAT_MISMATCH"},
        {-10, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
        {-11, "CL_BUILD_PROGRAM_FAILURE"},
        {-12, "CL_MAP_FAILURE"},
        {-13, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
        {-14, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
        {-15, "CL_COMPILE_PROGRAM_FAILURE"},
        {-16, "CL_LINKER_NOT_AVAILABLE"},
        {-17, "CL_LINK_PROGRAM_FAILURE"},
        {-18, "CL_DEVICE_PARTITION_FAILED"},
        {-19, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
        {-30, "CL_INVALID_VALUE"},
        {-31, "CL_INVALID_DEVICE_TYPE"},
        {-32, "CL_INVALID_PLATFORM"},
        {-33, "CL_INVALID_DEVICE"},
        {-34, "CL_INVALID_CONTEXT"},
        {-35, "CL_INVALID_QUEUE_PROPERTIES"},
        {-36, "CL_INVALID_COMMAND_QUEUE"},
        {-37, "CL_INVALID_HOST_PTR"},
        {-38, "CL_INVALID_MEM_OBJECT"},
        {-39, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
        {-40, "CL_INVALID_IMAGE_SIZE"},
        {-41, "CL_INVALID_SAMPLER"},
        {-42, "CL_INVALID_BINARY"},
        {-43, "CL_INVALID_BUILD_OPTIONS"},
        {-44, "CL_INVALID_PROGRAM"},
        {-45, "CL_INVALID_PROGRAM_EXECUTABLE"},
        {-46, "CL_INVALID_KERNEL_NAME"},
        {-47, "CL_INVALID_KERNEL_DEFINITION"},
        {-48, "CL_INVALID_KERNEL"},
        {-49, "CL_INVALID_ARG_INDEX"},
        {-50, "CL_INVALID_ARG_VALUE"},
        {-51, "CL_INVALID_ARG_SIZE"},
        {-52, "CL_INVALID_KERNEL_ARGS"},
        {-53, "CL_INVALID_WORK_DIMENSION"},
        {-54, "CL_INVALID_WORK_GROUP_SIZE"},
        {-55, "CL_INVALID_WORK_ITEM_SIZE"},
        {-56, "CL_INVALID_GLOBAL_OFFSET"},
        {-57, "CL_INVALID_EVENT_WAIT_LIST"},
        {-58, "CL_INVALID_EVENT"},
        {-59, "CL_INVALID_OPERATION"},
        {-60, "CL_INVALID_GL_OBJECT"},
        {-61, "CL_INVALID_BUFFER_SIZE"},
        {-62, "CL_INVALID_MIP_LEVEL"},
        {-63, "CL_INVALID_GLOBAL_WORK_SIZE"},
        {-64, "CL_INVALID_PROPERTY"},
        {-65, "CL_INVALID_IMAGE_DESCRIPTOR"},
        {-66, "CL_INVALID_COMPILER_OPTIONS"},
        {-67, "CL_INVALID_LINKER_OPTIONS"},
        {-68, "CL_INVALID_DEVICE_PARTITION_COUNT"},
        {-69, "CL_INVALID_PIPE_SIZE"},
        {-70, "CL_INVALID_DEVICE_QUEUE"},
        {-71, "CL_INVALID_SPEC_ID"},
        {-72, "CL_MAX_SIZE_RESTRICTION_EXCEEDED"}};
} // namespace clutils
