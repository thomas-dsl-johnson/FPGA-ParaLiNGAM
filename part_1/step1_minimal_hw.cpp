#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

int main() {
    #if defined(FPGA_HARDWARE)
        auto selector = sycl::ext::intel::fpga_selector_v;
        sycl::queue q(selector);
        // Submit an empty kernel. This is the simplest possible hardware task.
        q.submit([&](sycl::handler &h) {
            h.single_task([=]() {});
        });
        q.wait();
    #endif
    return 0;
}
