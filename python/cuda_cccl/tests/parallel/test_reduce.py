# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import random

import cupy as cp
import numba.cuda
import numba.types
import numpy as np
import pytest

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.cccl.parallel.experimental.iterators as iterators


def random_int(shape, dtype):
    return np.random.randint(0, 5, size=shape).astype(dtype)


def type_to_problem_sizes(dtype):
    if dtype in [np.uint8, np.int8]:
        return [2, 4, 5, 6]
    elif dtype in [np.uint16, np.int16]:
        return [4, 8, 12, 14]
    elif dtype in [np.uint32, np.int32]:
        return [16, 20, 24, 26]
    elif dtype in [np.uint64, np.int64]:
        return [16, 20, 24, 25]
    else:
        raise ValueError("Unsupported dtype")


def get_mark(dt, log_size):
    if log_size + np.log2(np.dtype(dt).itemsize) < 21:
        return tuple()
    return pytest.mark.large


dtype_size_pairs = [
    pytest.param(dt, 2**log_size, marks=get_mark(dt, log_size))
    for dt in [np.uint8, np.uint16, np.uint32, np.uint64]
    for log_size in type_to_problem_sizes(dt)
]


@pytest.mark.parametrize("dtype,num_items", dtype_size_pairs)
def test_device_reduce(dtype, num_items):
    def op(a, b):
        return a + b

    init_value = 42
    h_init = np.array([init_value], dtype=dtype)
    d_output = numba.cuda.device_array(1, dtype=dtype)

    h_input = random_int(num_items, dtype)
    d_input = numba.cuda.to_device(h_input)
    algorithms.reduce_into(d_input, d_output, op, d_input.size, h_init)
    h_output = d_output.copy_to_host()
    assert h_output[0] == sum(h_input) + init_value


def test_complex_device_reduce():
    def op(a, b):
        return a + b

    h_init = np.array([40.0 + 2.0j], dtype=complex)
    d_output = numba.cuda.device_array(1, dtype=complex)

    for num_items in [42, 420000]:
        real_imag = np.random.random((2, num_items))
        h_input = real_imag[0] + 1j * real_imag[1]
        d_input = numba.cuda.to_device(h_input)
        assert d_input.size == num_items
        algorithms.reduce_into(d_input, d_output, op, num_items, h_init)

        result = d_output.copy_to_host()[0]
        expected = np.sum(h_input, initial=h_init[0])
        assert result == pytest.approx(expected)


def _test_device_sum_with_iterator(
    l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
):
    def add_op(a, b):
        return a + b

    expected_result = start_sum_with
    for v in l_varr:
        expected_result = add_op(expected_result, v)

    if use_numpy_array:
        h_input = np.array(l_varr, dtype_inp)
        d_input = numba.cuda.to_device(h_input)
    else:
        d_input = i_input

    d_output = numba.cuda.device_array(1, dtype_out)  # to store device sum

    h_init = np.array([start_sum_with], dtype_out)

    algorithms.reduce_into(d_input, d_output, add_op, len(l_varr), h_init)

    h_output = d_output.copy_to_host()
    assert h_output[0] == expected_result


def mul2(val):
    return 2 * val


def mul3(val):
    return 3 * val


SUPPORTED_VALUE_TYPE_NAMES = (
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
)


@pytest.fixture(params=SUPPORTED_VALUE_TYPE_NAMES)
def supported_value_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy_array(request):
    return request.param


def test_device_sum_cache_modified_input_it(
    use_numpy_array, supported_value_type, num_items=3, start_sum_with=10
):
    rng = random.Random(0)
    l_varr = [rng.randrange(100) for _ in range(num_items)]
    dtype_inp = np.dtype(supported_value_type)
    dtype_out = dtype_inp
    input_devarr = numba.cuda.to_device(np.array(l_varr, dtype=dtype_inp))
    i_input = iterators.CacheModifiedInputIterator(input_devarr, modifier="stream")
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


def test_device_sum_constant_it(
    use_numpy_array, supported_value_type, num_items=3, start_sum_with=10
):
    l_varr = [42 for distance in range(num_items)]
    dtype_inp = np.dtype(supported_value_type)
    dtype_out = dtype_inp
    i_input = iterators.ConstantIterator(dtype_inp.type(42))
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


def test_device_sum_counting_it(
    use_numpy_array, supported_value_type, num_items=3, start_sum_with=10
):
    l_varr = [start_sum_with + distance for distance in range(num_items)]
    dtype_inp = np.dtype(supported_value_type)
    dtype_out = dtype_inp
    i_input = iterators.CountingIterator(dtype_inp.type(start_sum_with))
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


@pytest.mark.parametrize(
    "value_type_name_pair",
    list(zip(SUPPORTED_VALUE_TYPE_NAMES, SUPPORTED_VALUE_TYPE_NAMES))
    + [
        ("float32", "int16"),
        ("float32", "int32"),
        ("float64", "int32"),
        ("float64", "int64"),
        ("int64", "float32"),
    ],
)
def test_device_sum_map_mul2_count_it(
    use_numpy_array, value_type_name_pair, num_items=3, start_sum_with=10
):
    l_varr = [2 * (start_sum_with + distance) for distance in range(num_items)]
    vtn_out, vtn_inp = value_type_name_pair
    dtype_inp = np.dtype(vtn_inp)
    dtype_out = np.dtype(vtn_out)
    i_input = iterators.TransformIterator(
        iterators.CountingIterator(dtype_inp.type(start_sum_with)), mul2
    )
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


@pytest.mark.parametrize(
    ("fac_out", "fac_mid", "vtn_out", "vtn_mid", "vtn_inp"),
    [
        (3, 2, "int32", "int32", "int32"),
        (2, 2, "float64", "float32", "int16"),
    ],
)
def test_device_sum_map_mul_map_mul_count_it(
    use_numpy_array,
    fac_out,
    fac_mid,
    vtn_out,
    vtn_mid,
    vtn_inp,
    num_items=3,
    start_sum_with=10,
):
    l_varr = [
        fac_out * (fac_mid * (start_sum_with + distance))
        for distance in range(num_items)
    ]
    dtype_inp = np.dtype(vtn_inp)
    dtype_out = np.dtype(vtn_out)
    mul_funcs = {2: mul2, 3: mul3}
    i_input = iterators.TransformIterator(
        iterators.TransformIterator(
            iterators.CountingIterator(dtype_inp.type(start_sum_with)),
            mul_funcs[fac_mid],
        ),
        mul_funcs[fac_out],
    )
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


@pytest.mark.parametrize(
    "value_type_name_pair",
    [
        ("int32", "int32"),
        ("int64", "int32"),
        ("int32", "int64"),
    ],
)
def test_device_sum_map_mul2_cp_array_it(
    use_numpy_array, value_type_name_pair, num_items=3, start_sum_with=10
):
    vtn_out, vtn_inp = value_type_name_pair
    dtype_inp = np.dtype(vtn_inp)
    dtype_out = np.dtype(vtn_out)
    rng = random.Random(0)
    l_d_in = [rng.randrange(100) for _ in range(num_items)]
    a_d_in = cp.array(l_d_in, dtype_inp)
    i_input = iterators.TransformIterator(a_d_in, mul2)
    l_varr = [mul2(v) for v in l_d_in]
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


def test_reducer_caching():
    def sum_op(x, y):
        return x + y

    # inputs are device arrays
    reducer_1 = algorithms.make_reduce(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are device arrays of different dtype:
    reducer_1 = algorithms.make_reduce(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        cp.zeros(3, dtype="int32"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is not reducer_2

    # outputs are of different dtype:
    reducer_1 = algorithms.make_reduce(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int32"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is not reducer_2

    # inputs are of same dtype but different size
    # (should still use cached reducer):
    reducer_1 = algorithms.make_reduce(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        cp.zeros(5, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are counting iterators of the
    # same value type:
    reducer_1 = algorithms.make_reduce(
        iterators.CountingIterator(np.int32(0)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        iterators.CountingIterator(np.int32(0)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are counting iterators of different value type:
    reducer_1 = algorithms.make_reduce(
        iterators.CountingIterator(np.int32(0)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        iterators.CountingIterator(np.int64(0)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is not reducer_2

    def op1(x):
        return x

    def op2(x):
        return 2 * x

    def op3(x):
        return x

    # inputs are TransformIterators
    reducer_1 = algorithms.make_reduce(
        iterators.TransformIterator(iterators.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        iterators.TransformIterator(iterators.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are TransformIterators with different
    # op:
    reducer_1 = algorithms.make_reduce(
        iterators.TransformIterator(iterators.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        iterators.TransformIterator(iterators.CountingIterator(np.int32(0)), op2),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is not reducer_2

    # inputs are TransformIterators with same op
    # but different name:
    reducer_1 = algorithms.make_reduce(
        iterators.TransformIterator(iterators.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        iterators.TransformIterator(iterators.CountingIterator(np.int32(0)), op3),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )

    # inputs are CountingIterators of same kind
    # but different state:
    reducer_1 = algorithms.make_reduce(
        iterators.CountingIterator(np.int32(0)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        iterators.CountingIterator(np.int32(1)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )

    assert reducer_1 is reducer_2

    # inputs are TransformIterators of same kind
    # but different state:
    ary1 = cp.asarray([0, 1, 2], dtype="int64")
    ary2 = cp.asarray([0, 1], dtype="int64")
    reducer_1 = algorithms.make_reduce(
        iterators.TransformIterator(ary1, op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        iterators.TransformIterator(ary2, op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are TransformIterators of same kind
    # but different state:
    reducer_1 = algorithms.make_reduce(
        iterators.TransformIterator(iterators.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        iterators.TransformIterator(iterators.CountingIterator(np.int32(1)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are TransformIterators with different kind:
    reducer_1 = algorithms.make_reduce(
        iterators.TransformIterator(iterators.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = algorithms.make_reduce(
        iterators.TransformIterator(iterators.CountingIterator(np.int64(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is not reducer_2


@pytest.fixture(params=[True, False])
def array_2d(request):
    f_contiguous = request.param
    arr = cp.random.rand(5, 10)
    if f_contiguous:
        try:
            return cp.asfortranarray(arr)
        except ImportError:  # cublas unavailable
            return arr
    else:
        return arr


def test_reduce_2d_array(array_2d):
    def binary_op(x, y):
        return x + y

    d_out = cp.empty(1, dtype=array_2d.dtype)
    h_init = np.asarray([0], dtype=array_2d.dtype)
    d_in = array_2d
    algorithms.reduce_into(d_in, d_out, binary_op, d_in.size, h_init)
    np.testing.assert_allclose(d_in.sum().get(), d_out.get())


def test_reduce_non_contiguous():
    def binary_op(x, y):
        return x + y

    size = 10
    d_out = cp.empty(1, dtype="int64")
    h_init = np.asarray([0], dtype="int64")

    d_in = cp.zeros((size, 2))[:, 0]
    with pytest.raises(ValueError, match="Non-contiguous arrays are not supported."):
        _ = algorithms.make_reduce(d_in, d_out, binary_op, h_init)

    d_in = cp.zeros(size)[::2]
    with pytest.raises(ValueError, match="Non-contiguous arrays are not supported."):
        _ = algorithms.make_reduce(d_in, d_out, binary_op, h_init)


def test_reduce_with_stream(cuda_stream):
    def add_op(x, y):
        return x + y

    h_init = np.asarray([0], dtype=np.int32)
    h_in = random_int(5, np.int32)

    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)
    with cp_stream:
        d_in = cp.asarray(h_in)
        d_out = cp.empty(1, dtype=np.int32)

    algorithms.reduce_into(d_in, d_out, add_op, d_in.size, h_init, stream=cuda_stream)
    with cp_stream:
        cp.testing.assert_allclose(d_in.sum().get(), d_out.get())


def test_reduce_invalid_stream():
    # Invalid stream that doesn't implement __cuda_stream__
    class Stream1:
        def __init__(self):
            pass

    # Invalid stream that implements __cuda_stream__ but returns the wrong type
    class Stream2:
        def __init__(self):
            pass

        def __cuda_stream__(self):
            return None

    # Invalid stream that returns an invalid handle
    class Stream3:
        def __init__(self):
            pass

        def __cuda_stream__(self):
            return (0, None)

    def add_op(x, y):
        return x + y

    d_out = cp.empty(1)
    h_init = np.empty(1)
    d_in = cp.empty(1)
    reduce_into = algorithms.make_reduce(d_in, d_out, add_op, h_init)

    with pytest.raises(
        TypeError, match="does not implement the '__cuda_stream__' protocol"
    ):
        _ = reduce_into(
            None,
            d_in=d_in,
            d_out=d_out,
            num_items=d_in.size,
            h_init=h_init,
            stream=Stream1(),
        )

    with pytest.raises(
        TypeError, match="could not obtain __cuda_stream__ protocol version and handle"
    ):
        _ = reduce_into(
            None,
            d_in=d_in,
            d_out=d_out,
            num_items=d_in.size,
            h_init=h_init,
            stream=Stream2(),
        )

    with pytest.raises(TypeError, match="invalid stream handle"):
        _ = reduce_into(
            None,
            d_in=d_in,
            d_out=d_out,
            num_items=d_in.size,
            h_init=h_init,
            stream=Stream3(),
        )


def test_make_reduce_object_api():
    def add_op(x, y):
        return x + y

    # Test with simple integer array
    dtype = np.int32
    init_value = 5
    h_init = np.array([init_value], dtype=dtype)
    h_input = np.array([1, 2, 3, 4], dtype=dtype)
    d_input = cp.asarray(h_input)
    d_output = cp.empty(1, dtype=dtype)

    # Create reducer
    reducer = algorithms.make_reduce(d_input, d_output, add_op, h_init)

    # Allocate temporary storage and run reduction
    temp_storage_size = reducer(None, d_input, d_output, len(h_input), h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    reducer(d_temp_storage, d_input, d_output, len(h_input), h_init)

    # Verify result
    expected_result = np.sum(h_input) + init_value
    actual_result = d_output.get()[0]
    assert actual_result == expected_result


def test_exclusive_scan_object_api():
    """Test that the make_exclusive_scan object API works correctly."""

    def add_op(x, y):
        return x + y

    # Test with simple integer array
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    h_input = np.array([1, 2, 3, 4], dtype=dtype)
    d_input = cp.asarray(h_input)
    d_output = cp.empty(len(h_input), dtype=dtype)

    # Create scanner
    scanner = algorithms.make_exclusive_scan(d_input, d_output, add_op, h_init)

    # Allocate temporary storage and run scan
    temp_storage_size = scanner(None, d_input, d_output, len(h_input), h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    scanner(d_temp_storage, d_input, d_output, len(h_input), h_init)

    # Verify result - exclusive scan with init=0 should be [0, 1, 3, 6]
    expected_result = np.array([0, 1, 3, 6], dtype=dtype)
    actual_result = d_output.get()
    np.testing.assert_array_equal(actual_result, expected_result)


def test_inclusive_scan_object_api():
    """Test that the make_inclusive_scan object API works correctly."""

    def add_op(x, y):
        return x + y

    # Test with simple integer array
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    h_input = np.array([1, 2, 3, 4], dtype=dtype)
    d_input = cp.asarray(h_input)
    d_output = cp.empty(len(h_input), dtype=dtype)

    # Create scanner
    scanner = algorithms.make_inclusive_scan(d_input, d_output, add_op, h_init)

    # Allocate temporary storage and run scan
    temp_storage_size = scanner(None, d_input, d_output, len(h_input), h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    scanner(d_temp_storage, d_input, d_output, len(h_input), h_init)

    # Verify result - inclusive scan with init=0 should be [1, 3, 6, 10]
    expected_result = np.array([1, 3, 6, 10], dtype=dtype)
    actual_result = d_output.get()
    np.testing.assert_array_equal(actual_result, expected_result)


def test_merge_sort_object_api():
    """Test that the make_merge_sort object API works correctly."""

    def compare_op(lhs, rhs):
        return np.uint8(lhs < rhs)

    # Test with simple integer array
    dtype = np.int32
    h_input_keys = np.array([4, 2, 3, 1], dtype=dtype)
    h_input_values = np.array([40, 20, 30, 10], dtype=dtype)
    d_input_keys = cp.asarray(h_input_keys)
    d_input_values = cp.asarray(h_input_values)
    d_output_keys = cp.empty_like(d_input_keys)
    d_output_values = cp.empty_like(d_input_values)

    # Create sorter
    sorter = algorithms.make_merge_sort(
        d_input_keys, d_input_values, d_output_keys, d_output_values, compare_op
    )

    # Allocate temporary storage and run sort
    temp_storage_size = sorter(
        None,
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        len(h_input_keys),
    )
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    sorter(
        d_temp_storage,
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        len(h_input_keys),
    )

    # Verify result - should be sorted
    expected_keys = np.array([1, 2, 3, 4], dtype=dtype)
    expected_values = np.array([10, 20, 30, 40], dtype=dtype)
    actual_keys = d_output_keys.get()
    actual_values = d_output_values.get()
    np.testing.assert_array_equal(actual_keys, expected_keys)
    np.testing.assert_array_equal(actual_values, expected_values)


def test_radix_sort_object_api():
    """Test that the make_radix_sort object API works correctly."""
    # Test with simple integer array
    dtype = np.int32
    h_input_keys = np.array([4, 2, 3, 1], dtype=dtype)
    h_input_values = np.array([40, 20, 30, 10], dtype=dtype)
    d_input_keys = cp.asarray(h_input_keys)
    d_input_values = cp.asarray(h_input_values)
    d_output_keys = cp.empty_like(d_input_keys)
    d_output_values = cp.empty_like(d_input_values)

    # Create sorter
    sorter = algorithms.make_radix_sort(
        d_input_keys,
        d_output_keys,
        d_input_values,
        d_output_values,
        algorithms.SortOrder.ASCENDING,
    )

    # Allocate temporary storage and run sort
    temp_storage_size = sorter(
        None,
        d_input_keys,
        d_output_keys,
        d_input_values,
        d_output_values,
        len(h_input_keys),
    )
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    sorter(
        d_temp_storage,
        d_input_keys,
        d_output_keys,
        d_input_values,
        d_output_values,
        len(h_input_keys),
    )

    # Verify result - should be sorted
    expected_keys = np.array([1, 2, 3, 4], dtype=dtype)
    expected_values = np.array([10, 20, 30, 40], dtype=dtype)
    actual_keys = d_output_keys.get()
    actual_values = d_output_values.get()
    np.testing.assert_array_equal(actual_keys, expected_keys)
    np.testing.assert_array_equal(actual_values, expected_values)


def test_unique_by_key_object_api():
    """Test that the make_unique_by_key object API works correctly."""

    def compare_op(lhs, rhs):
        return np.uint8(lhs == rhs)

    # Test with simple array with duplicates
    dtype = np.int32
    h_input_keys = np.array([1, 1, 2, 3, 3], dtype=dtype)
    h_input_values = np.array([10, 20, 30, 40, 50], dtype=dtype)
    d_input_keys = cp.asarray(h_input_keys)
    d_input_values = cp.asarray(h_input_values)
    d_output_keys = cp.empty_like(d_input_keys)
    d_output_values = cp.empty_like(d_input_values)
    d_num_selected = cp.empty(1, dtype=np.int32)

    # Create uniquer
    uniquer = algorithms.make_unique_by_key(
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        d_num_selected,
        compare_op,
    )

    # Allocate temporary storage and run unique
    temp_storage_size = uniquer(
        None,
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        d_num_selected,
        len(h_input_keys),
    )
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    uniquer(
        d_temp_storage,
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        d_num_selected,
        len(h_input_keys),
    )

    # Verify result - should remove consecutive duplicates
    num_selected = d_num_selected.get()[0]
    expected_keys = np.array([1, 2, 3], dtype=dtype)
    expected_values = np.array([10, 30, 40], dtype=dtype)
    actual_keys = d_output_keys.get()[:num_selected]
    actual_values = d_output_values.get()[:num_selected]
    np.testing.assert_array_equal(actual_keys, expected_keys)
    np.testing.assert_array_equal(actual_values, expected_values)


def test_segmented_reduce_object_api():
    """Test that the make_segmented_reduce object API works correctly."""

    def add_op(a, b):
        return a + b

    # Test with simple segmented data
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    h_input = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)  # [1,2,3] and [4,5,6]
    d_input = cp.asarray(h_input)
    d_output = cp.empty(2, dtype=dtype)  # Two segments

    # Define segment offsets
    start_offsets = cp.array([0, 3], dtype=np.int64)
    end_offsets = cp.array([3, 6], dtype=np.int64)

    # Create segmented reducer
    reducer = algorithms.make_segmented_reduce(
        d_input, d_output, start_offsets, end_offsets, add_op, h_init
    )

    # Allocate temporary storage and run segmented reduction
    temp_storage_size = reducer(
        None, d_input, d_output, 2, start_offsets, end_offsets, h_init
    )
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    reducer(d_temp_storage, d_input, d_output, 2, start_offsets, end_offsets, h_init)

    # Verify result - segment sums should be [6, 15]
    expected_result = np.array([6, 15], dtype=dtype)  # 1+2+3=6, 4+5+6=15
    actual_result = d_output.get()
    np.testing.assert_array_equal(actual_result, expected_result)


def test_unary_transform_object_api():
    """Test that the make_unary_transform object API works correctly."""

    def add_one_op(a):
        return a + 1

    # Test with simple integer array
    dtype = np.int32
    h_input = np.array([1, 2, 3, 4], dtype=dtype)
    d_input = cp.asarray(h_input)
    d_output = cp.empty_like(d_input)

    # Create transformer
    transformer = algorithms.make_unary_transform(d_input, d_output, add_one_op)

    # Run transformation
    transformer(d_input, d_output, len(h_input))

    # Verify result - should add 1 to each element
    expected_result = np.array([2, 3, 4, 5], dtype=dtype)
    actual_result = d_output.get()
    np.testing.assert_array_equal(actual_result, expected_result)


def test_binary_transform_object_api():
    """Test that the make_binary_transform object API works correctly."""

    def add_op(a, b):
        return a + b

    # Test with simple integer arrays
    dtype = np.int32
    h_input1 = np.array([1, 2, 3, 4], dtype=dtype)
    h_input2 = np.array([10, 20, 30, 40], dtype=dtype)
    d_input1 = cp.asarray(h_input1)
    d_input2 = cp.asarray(h_input2)
    d_output = cp.empty_like(d_input1)

    # Create transformer
    transformer = algorithms.make_binary_transform(d_input1, d_input2, d_output, add_op)

    # Run transformation
    transformer(d_input1, d_input2, d_output, len(h_input1))

    # Verify result - should add corresponding elements
    expected_result = np.array([11, 22, 33, 44], dtype=dtype)
    actual_result = d_output.get()
    np.testing.assert_array_equal(actual_result, expected_result)
