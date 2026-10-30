#include "api.h"
#include <string>
#include <stdexcept>
using namespace sph_native;
struct Request {
    int task, device, precision, dimension, rank, operation, kernel, support, scheme, lap,
        direction, dot;
    int offset_bits, index_bits, validation, threads;
    int64_t queries, references;
    const void *inputs[OUTPUT];
    int64_t sizes[OUTPUT];
    void *output;
    int64_t output_size;
    const void *cotangent;
    int64_t cotangent_size;
    void *gradients[OUTPUT];
    int64_t gradient_sizes[OUTPUT];
    uint64_t mask;
    int accumulate;
    void *stream;
    int64_t result;
};
static thread_local std::string error;
extern "C" const char *standalone_error() { return error.c_str(); }
extern "C" int standalone_supports(int device) { return supports(Device(device)); }
extern "C" int standalone_run(Request *r) {
    try {
        error.clear();
        Problem p{};
        auto b = [&](int s) { return ConstBuffer{r->inputs[s], r->sizes[s]}; };
        p.precision = Precision(r->precision);
        p.dimension = r->dimension;
        p.field_rank = r->rank;
        p.query = {r->queries, b(QX), b(QH), {}, b(QDENSITY), b(QKIND)};
        p.reference = {r->references, b(RX), b(RH), b(MASS), b(DENSITY), b(RKIND)};
        p.domain = {b(LO), b(HI), b(PERIODIC)};
        p.adjacency = {b(OFFSETS), b(COUNTS), b(NEIGHBORS), r->offset_bits, r->index_bits};
        p.query_values = b(QVALUE);
        p.reference_values = b(RVALUE);
        p.properties = {
            Operation(r->operation), Kernel(r->kernel), Support(r->support), Scheme(r->scheme),
            Laplacian(r->lap),       r->direction,      bool(r->dot)};
        Context c{Device(r->device), 0, r->stream, r->threads, Validation(r->validation)};
        Buffer out{r->output, r->output_size};
        if (r->task == 0)
            forward(c, p, out);
        else if (r->task == 1) {
            Adjoints bars{};
            bars.requested = r->mask;
            bars.accumulate = r->accumulate;
            for (int s = 0; s < OUTPUT; ++s)
                bars.slots[s] = {r->gradients[s], r->gradient_sizes[s]};
            adjoint(c, p, {r->cotangent, r->cotangent_size}, bars);
        } else if (r->task == 2)
            radius_count(c, p, out);
        else if (r->task == 3)
            r->result = radius_scan(c, r->queries, b(COUNTS), out);
        else if (r->task == 4)
            radius_collect(c, p, out);
        else if (r->task == 5)
            r->result = output_elements(p);
        else
            throw std::invalid_argument("Unknown test task");
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return 1;
    }
}
