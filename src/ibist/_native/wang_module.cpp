#define PY_SSIZE_T_CLEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Python.h>
#include <climits>
#include <exception>
#include <new>
#include "wang_core.h"

static PyObject* lower(PyObject*, PyObject* args) {
  int b, t, c, grid_one, grid_two;
  double level, precision;
  if (!PyArg_ParseTuple(args, "iiiddii", &b, &t, &c, &level, &precision,
                        &grid_one, &grid_two)) return nullptr;
  const long long n = static_cast<long long>(b) + t + c;
  if (b < 0 || t < 0 || c < 0 || n > INT_MAX ||
      (n + 1) * (n + 2) / 2 > 1000000 ||
      !std::isfinite(level) || !(level > 0 && level < 1) ||
      !std::isfinite(precision) || !(precision > 0) ||
      grid_one < 2 || grid_two < 2) {
    PyErr_SetString(PyExc_ValueError, "Invalid Wang counts or settings; "
                    "at most 1,000,000 states are supported.");
    return nullptr;
  }
  double result = 0;
  std::exception_ptr error;
  // Long-running native calculations do not hold Python's interpreter lock.
  Py_BEGIN_ALLOW_THREADS
  try {
    result = ibist_wang::wang_lower_one_sided(
      b, t, c, level, precision, grid_one, grid_two);
  } catch (...) {
    error = std::current_exception();
  }
  Py_END_ALLOW_THREADS
  if (error) {
    try {
      std::rethrow_exception(error);
    } catch (const std::bad_alloc&) {
      return PyErr_NoMemory();
    } catch (const std::exception& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
      return nullptr;
    } catch (...) {
      PyErr_SetString(PyExc_RuntimeError, "Unexpected Wang core error.");
      return nullptr;
    }
  }
  return PyFloat_FromDouble(result);
}

static PyMethodDef methods[] = {
  {"lower", lower, METH_VARARGS, "Native Wang lower confidence bound."},
  {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT, "_wang_native", "Shared ibist Wang core.", -1,
  methods, nullptr, nullptr, nullptr, nullptr
};

PyMODINIT_FUNC PyInit__wang_native(void) {
  return PyModule_Create(&module);
}
