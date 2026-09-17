#ifndef PLASTIX_PY_MODULE_HPP
#define PLASTIX_PY_MODULE_HPP

// Generic nanobind binding of NetworkHandle (ADR-007). A user shim writes
// `PLASTIX_PY_MODULE(name, factory)` where factory() returns a
// unique_ptr<NetworkHandle> (typically plastix::py::Wrap<Adapter<TheirNet>>).
// The same binding code serves every user network.

#include "plastix/py/adapter.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/unique_ptr.h>

#include <algorithm>
#include <memory>

namespace plastix::py {
namespace nb = nanobind;

using FloatArray = nb::ndarray<const float, nb::ndim<1>, nb::c_contig>;

// Bind the non-template handle once; reused by every module.
inline void bind_network(nb::module_ &m) {
  nb::class_<NetworkHandle>(m, "Network")
      .def("step",
           [](NetworkHandle &h, FloatArray x, FloatArray y) {
             h.step({x.data(), x.size()}, {y.data(), y.size()});
           })
      .def("output",
           [](NetworkHandle &h) {
             auto o = h.output();
             float *buf = new float[o.size()];
             std::copy(o.begin(), o.end(), buf);
             nb::capsule owner(buf, [](void *p) noexcept {
               delete[] static_cast<float *>(p);
             });
             return nb::ndarray<nb::numpy, float>(buf, {o.size()}, owner);
           })
      .def("num_live_conns", &NetworkHandle::num_live_conns)
      .def_prop_ro("num_input", &NetworkHandle::num_input)
      .def_prop_ro("num_output", &NetworkHandle::num_output)
      .def_prop_ro("step_count", &NetworkHandle::step_count)
      .def_prop_ro("device_resident", &NetworkHandle::device_resident);
}

} // namespace plastix::py

// Factory returns std::unique_ptr<plastix::py::NetworkHandle>.
#define PLASTIX_PY_MODULE(ModName, Factory)                                     \
  NB_MODULE(ModName, m) {                                                       \
    plastix::py::bind_network(m);                                              \
    m.def("create", []() { return Factory(); });                              \
  }

#endif // PLASTIX_PY_MODULE_HPP
