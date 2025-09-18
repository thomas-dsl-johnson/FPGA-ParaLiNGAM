#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iostream>

int main() {
    // We will target the FPGA EMULATOR for this test.
    // This compiles the SYCL code to run on the host CPU.
    #if defined(FPGA_EMULATOR)
        auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    #else
        // Provide a default for other cases, though we only care about the emulator here.
        auto selector = sycl::default_selector_v;
    #endif

    try {
        sycl::queue q(selector);
        std::cout << "Running on device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        // A buffer to hold a single integer result.
        sycl::buffer<int, 1> result_buf(1);

        // Submit a single_task kernel that does almost nothing.
        q.submit([&](sycl::handler &h) {
            auto accessor = result_buf.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() {
                accessor[0] = 10 + 20; // A trivial operation
            });
        });

        // Read the result back to the host to verify.
        auto host_accessor = result_buf.get_host_access();
        std::cout << "Minimal test successful. Result: " << host_accessor[0] << std::endl;

    } catch (sycl::exception const &e) {
        std::cerr << "Caught SYCL exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
