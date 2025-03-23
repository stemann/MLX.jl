module Wrapper

using MLX_C_jll
export MLX_C_jll

using CEnum

struct mlx_array_
    ctx::Ptr{Cvoid}
end

const mlx_array = mlx_array_

@cenum mlx_dtype_::UInt32 begin
    MLX_BOOL = 0
    MLX_UINT8 = 1
    MLX_UINT16 = 2
    MLX_UINT32 = 3
    MLX_UINT64 = 4
    MLX_INT8 = 5
    MLX_INT16 = 6
    MLX_INT32 = 7
    MLX_INT64 = 8
    MLX_FLOAT16 = 9
    MLX_FLOAT32 = 10
    MLX_BFLOAT16 = 11
    MLX_COMPLEX64 = 12
end

const mlx_dtype = mlx_dtype_

function mlx_dtype_size(dtype)
    ccall((:mlx_dtype_size, libmlxc), Csize_t, (mlx_dtype,), dtype)
end

struct mlx_string_
    ctx::Ptr{Cvoid}
end

const mlx_string = mlx_string_

function mlx_array_tostring(str, arr)
    ccall((:mlx_array_tostring, libmlxc), Cint, (Ptr{mlx_string}, mlx_array), str, arr)
end

# no prototype is found for this function at array.h:66:11, please use with caution
function mlx_array_new()
    ccall((:mlx_array_new, libmlxc), mlx_array, ())
end

function mlx_array_free(arr)
    ccall((:mlx_array_free, libmlxc), Cint, (mlx_array,), arr)
end

function mlx_array_new_bool(val)
    ccall((:mlx_array_new_bool, libmlxc), mlx_array, (Bool,), val)
end

function mlx_array_new_int(val)
    ccall((:mlx_array_new_int, libmlxc), mlx_array, (Cint,), val)
end

function mlx_array_new_float(val)
    ccall((:mlx_array_new_float, libmlxc), mlx_array, (Cfloat,), val)
end

function mlx_array_new_complex(real_val, imag_val)
    ccall((:mlx_array_new_complex, libmlxc), mlx_array, (Cfloat, Cfloat), real_val, imag_val)
end

function mlx_array_new_data(data, shape, dim, dtype)
    ccall((:mlx_array_new_data, libmlxc), mlx_array, (Ptr{Cvoid}, Ptr{Cint}, Cint, mlx_dtype), data, shape, dim, dtype)
end

function mlx_array_set(arr, src)
    ccall((:mlx_array_set, libmlxc), Cint, (Ptr{mlx_array}, mlx_array), arr, src)
end

function mlx_array_set_bool(arr, val)
    ccall((:mlx_array_set_bool, libmlxc), Cint, (Ptr{mlx_array}, Bool), arr, val)
end

function mlx_array_set_int(arr, val)
    ccall((:mlx_array_set_int, libmlxc), Cint, (Ptr{mlx_array}, Cint), arr, val)
end

function mlx_array_set_float(arr, val)
    ccall((:mlx_array_set_float, libmlxc), Cint, (Ptr{mlx_array}, Cfloat), arr, val)
end

function mlx_array_set_complex(arr, real_val, imag_val)
    ccall((:mlx_array_set_complex, libmlxc), Cint, (Ptr{mlx_array}, Cfloat, Cfloat), arr, real_val, imag_val)
end

function mlx_array_set_data(arr, data, shape, dim, dtype)
    ccall((:mlx_array_set_data, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cvoid}, Ptr{Cint}, Cint, mlx_dtype), arr, data, shape, dim, dtype)
end

function mlx_array_itemsize(arr)
    ccall((:mlx_array_itemsize, libmlxc), Csize_t, (mlx_array,), arr)
end

function mlx_array_size(arr)
    ccall((:mlx_array_size, libmlxc), Csize_t, (mlx_array,), arr)
end

function mlx_array_nbytes(arr)
    ccall((:mlx_array_nbytes, libmlxc), Csize_t, (mlx_array,), arr)
end

function mlx_array_ndim(arr)
    ccall((:mlx_array_ndim, libmlxc), Csize_t, (mlx_array,), arr)
end

function mlx_array_shape(arr)
    ccall((:mlx_array_shape, libmlxc), Ptr{Cint}, (mlx_array,), arr)
end

function mlx_array_strides(arr)
    ccall((:mlx_array_strides, libmlxc), Ptr{Csize_t}, (mlx_array,), arr)
end

function mlx_array_dim(arr, dim)
    ccall((:mlx_array_dim, libmlxc), Cint, (mlx_array, Cint), arr, dim)
end

function mlx_array_dtype(arr)
    ccall((:mlx_array_dtype, libmlxc), mlx_dtype, (mlx_array,), arr)
end

function mlx_array_eval(arr)
    ccall((:mlx_array_eval, libmlxc), Cint, (mlx_array,), arr)
end

function mlx_array_item_bool(res, arr)
    ccall((:mlx_array_item_bool, libmlxc), Cint, (Ptr{Bool}, mlx_array), res, arr)
end

function mlx_array_item_uint8(res, arr)
    ccall((:mlx_array_item_uint8, libmlxc), Cint, (Ptr{UInt8}, mlx_array), res, arr)
end

function mlx_array_item_uint16(res, arr)
    ccall((:mlx_array_item_uint16, libmlxc), Cint, (Ptr{UInt16}, mlx_array), res, arr)
end

function mlx_array_item_uint32(res, arr)
    ccall((:mlx_array_item_uint32, libmlxc), Cint, (Ptr{UInt32}, mlx_array), res, arr)
end

function mlx_array_item_uint64(res, arr)
    ccall((:mlx_array_item_uint64, libmlxc), Cint, (Ptr{UInt64}, mlx_array), res, arr)
end

function mlx_array_item_int8(res, arr)
    ccall((:mlx_array_item_int8, libmlxc), Cint, (Ptr{Int8}, mlx_array), res, arr)
end

function mlx_array_item_int16(res, arr)
    ccall((:mlx_array_item_int16, libmlxc), Cint, (Ptr{Int16}, mlx_array), res, arr)
end

function mlx_array_item_int32(res, arr)
    ccall((:mlx_array_item_int32, libmlxc), Cint, (Ptr{Int32}, mlx_array), res, arr)
end

function mlx_array_item_int64(res, arr)
    ccall((:mlx_array_item_int64, libmlxc), Cint, (Ptr{Int64}, mlx_array), res, arr)
end

function mlx_array_item_float32(res, arr)
    ccall((:mlx_array_item_float32, libmlxc), Cint, (Ptr{Cfloat}, mlx_array), res, arr)
end

function mlx_array_item_complex64(res, arr)
    ccall((:mlx_array_item_complex64, libmlxc), Cint, (Ptr{ComplexF32}, mlx_array), res, arr)
end

function mlx_array_data_bool(arr)
    ccall((:mlx_array_data_bool, libmlxc), Ptr{Bool}, (mlx_array,), arr)
end

function mlx_array_data_uint8(arr)
    ccall((:mlx_array_data_uint8, libmlxc), Ptr{UInt8}, (mlx_array,), arr)
end

function mlx_array_data_uint16(arr)
    ccall((:mlx_array_data_uint16, libmlxc), Ptr{UInt16}, (mlx_array,), arr)
end

function mlx_array_data_uint32(arr)
    ccall((:mlx_array_data_uint32, libmlxc), Ptr{UInt32}, (mlx_array,), arr)
end

function mlx_array_data_uint64(arr)
    ccall((:mlx_array_data_uint64, libmlxc), Ptr{UInt64}, (mlx_array,), arr)
end

function mlx_array_data_int8(arr)
    ccall((:mlx_array_data_int8, libmlxc), Ptr{Int8}, (mlx_array,), arr)
end

function mlx_array_data_int16(arr)
    ccall((:mlx_array_data_int16, libmlxc), Ptr{Int16}, (mlx_array,), arr)
end

function mlx_array_data_int32(arr)
    ccall((:mlx_array_data_int32, libmlxc), Ptr{Int32}, (mlx_array,), arr)
end

function mlx_array_data_int64(arr)
    ccall((:mlx_array_data_int64, libmlxc), Ptr{Int64}, (mlx_array,), arr)
end

function mlx_array_data_float32(arr)
    ccall((:mlx_array_data_float32, libmlxc), Ptr{Cfloat}, (mlx_array,), arr)
end

function mlx_array_data_complex64(arr)
    ccall((:mlx_array_data_complex64, libmlxc), Ptr{ComplexF32}, (mlx_array,), arr)
end

function _mlx_array_is_available(res, arr)
    ccall((:_mlx_array_is_available, libmlxc), Cint, (Ptr{Bool}, mlx_array), res, arr)
end

function _mlx_array_wait(arr)
    ccall((:_mlx_array_wait, libmlxc), Cint, (mlx_array,), arr)
end

function _mlx_array_is_contiguous(res, arr)
    ccall((:_mlx_array_is_contiguous, libmlxc), Cint, (Ptr{Bool}, mlx_array), res, arr)
end

function _mlx_array_is_row_contiguous(res, arr)
    ccall((:_mlx_array_is_row_contiguous, libmlxc), Cint, (Ptr{Bool}, mlx_array), res, arr)
end

function _mlx_array_is_col_contiguous(res, arr)
    ccall((:_mlx_array_is_col_contiguous, libmlxc), Cint, (Ptr{Bool}, mlx_array), res, arr)
end

struct mlx_closure_
    ctx::Ptr{Cvoid}
end

const mlx_closure = mlx_closure_

# no prototype is found for this function at closure.h:27:13, please use with caution
function mlx_closure_new()
    ccall((:mlx_closure_new, libmlxc), mlx_closure, ())
end

function mlx_closure_free(cls)
    ccall((:mlx_closure_free, libmlxc), Cint, (mlx_closure,), cls)
end

function mlx_closure_new_func(fun)
    ccall((:mlx_closure_new_func, libmlxc), mlx_closure, (Ptr{Cvoid},), fun)
end

function mlx_closure_new_func_payload(fun, payload, dtor)
    ccall((:mlx_closure_new_func_payload, libmlxc), mlx_closure, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), fun, payload, dtor)
end

function mlx_closure_set(cls, src)
    ccall((:mlx_closure_set, libmlxc), Cint, (Ptr{mlx_closure}, mlx_closure), cls, src)
end

struct mlx_vector_array_
    ctx::Ptr{Cvoid}
end

const mlx_vector_array = mlx_vector_array_

function mlx_closure_apply(res, cls, input)
    ccall((:mlx_closure_apply, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_closure, mlx_vector_array), res, cls, input)
end

function mlx_closure_new_unary(fun)
    ccall((:mlx_closure_new_unary, libmlxc), mlx_closure, (Ptr{Cvoid},), fun)
end

struct mlx_closure_value_and_grad_
    ctx::Ptr{Cvoid}
end

const mlx_closure_value_and_grad = mlx_closure_value_and_grad_

# no prototype is found for this function at closure.h:46:28, please use with caution
function mlx_closure_value_and_grad_new()
    ccall((:mlx_closure_value_and_grad_new, libmlxc), mlx_closure_value_and_grad, ())
end

function mlx_closure_value_and_grad_free(cls)
    ccall((:mlx_closure_value_and_grad_free, libmlxc), Cint, (mlx_closure_value_and_grad,), cls)
end

function mlx_closure_value_and_grad_new_func(fun)
    ccall((:mlx_closure_value_and_grad_new_func, libmlxc), mlx_closure_value_and_grad, (Ptr{Cvoid},), fun)
end

function mlx_closure_value_and_grad_new_func_payload(fun, payload, dtor)
    ccall((:mlx_closure_value_and_grad_new_func_payload, libmlxc), mlx_closure_value_and_grad, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), fun, payload, dtor)
end

function mlx_closure_value_and_grad_set(cls, src)
    ccall((:mlx_closure_value_and_grad_set, libmlxc), Cint, (Ptr{mlx_closure_value_and_grad}, mlx_closure_value_and_grad), cls, src)
end

function mlx_closure_value_and_grad_apply(res_0, res_1, cls, input)
    ccall((:mlx_closure_value_and_grad_apply, libmlxc), Cint, (Ptr{mlx_vector_array}, Ptr{mlx_vector_array}, mlx_closure_value_and_grad, mlx_vector_array), res_0, res_1, cls, input)
end

struct mlx_closure_custom_
    ctx::Ptr{Cvoid}
end

const mlx_closure_custom = mlx_closure_custom_

# no prototype is found for this function at closure.h:70:20, please use with caution
function mlx_closure_custom_new()
    ccall((:mlx_closure_custom_new, libmlxc), mlx_closure_custom, ())
end

function mlx_closure_custom_free(cls)
    ccall((:mlx_closure_custom_free, libmlxc), Cint, (mlx_closure_custom,), cls)
end

function mlx_closure_custom_new_func(fun)
    ccall((:mlx_closure_custom_new_func, libmlxc), mlx_closure_custom, (Ptr{Cvoid},), fun)
end

function mlx_closure_custom_new_func_payload(fun, payload, dtor)
    ccall((:mlx_closure_custom_new_func_payload, libmlxc), mlx_closure_custom, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), fun, payload, dtor)
end

function mlx_closure_custom_set(cls, src)
    ccall((:mlx_closure_custom_set, libmlxc), Cint, (Ptr{mlx_closure_custom}, mlx_closure_custom), cls, src)
end

function mlx_closure_custom_apply(res, cls, input_0, input_1, input_2)
    ccall((:mlx_closure_custom_apply, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_closure_custom, mlx_vector_array, mlx_vector_array, mlx_vector_array), res, cls, input_0, input_1, input_2)
end

struct mlx_closure_custom_jvp_
    ctx::Ptr{Cvoid}
end

const mlx_closure_custom_jvp = mlx_closure_custom_jvp_

# no prototype is found for this function at closure.h:99:24, please use with caution
function mlx_closure_custom_jvp_new()
    ccall((:mlx_closure_custom_jvp_new, libmlxc), mlx_closure_custom_jvp, ())
end

function mlx_closure_custom_jvp_free(cls)
    ccall((:mlx_closure_custom_jvp_free, libmlxc), Cint, (mlx_closure_custom_jvp,), cls)
end

function mlx_closure_custom_jvp_new_func(fun)
    ccall((:mlx_closure_custom_jvp_new_func, libmlxc), mlx_closure_custom_jvp, (Ptr{Cvoid},), fun)
end

function mlx_closure_custom_jvp_new_func_payload(fun, payload, dtor)
    ccall((:mlx_closure_custom_jvp_new_func_payload, libmlxc), mlx_closure_custom_jvp, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), fun, payload, dtor)
end

function mlx_closure_custom_jvp_set(cls, src)
    ccall((:mlx_closure_custom_jvp_set, libmlxc), Cint, (Ptr{mlx_closure_custom_jvp}, mlx_closure_custom_jvp), cls, src)
end

function mlx_closure_custom_jvp_apply(res, cls, input_0, input_1, input_2, input_2_num)
    ccall((:mlx_closure_custom_jvp_apply, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_closure_custom_jvp, mlx_vector_array, mlx_vector_array, Ptr{Cint}, Csize_t), res, cls, input_0, input_1, input_2, input_2_num)
end

struct mlx_closure_custom_vmap_
    ctx::Ptr{Cvoid}
end

const mlx_closure_custom_vmap = mlx_closure_custom_vmap_

# no prototype is found for this function at closure.h:131:25, please use with caution
function mlx_closure_custom_vmap_new()
    ccall((:mlx_closure_custom_vmap_new, libmlxc), mlx_closure_custom_vmap, ())
end

function mlx_closure_custom_vmap_free(cls)
    ccall((:mlx_closure_custom_vmap_free, libmlxc), Cint, (mlx_closure_custom_vmap,), cls)
end

function mlx_closure_custom_vmap_new_func(fun)
    ccall((:mlx_closure_custom_vmap_new_func, libmlxc), mlx_closure_custom_vmap, (Ptr{Cvoid},), fun)
end

function mlx_closure_custom_vmap_new_func_payload(fun, payload, dtor)
    ccall((:mlx_closure_custom_vmap_new_func_payload, libmlxc), mlx_closure_custom_vmap, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), fun, payload, dtor)
end

function mlx_closure_custom_vmap_set(cls, src)
    ccall((:mlx_closure_custom_vmap_set, libmlxc), Cint, (Ptr{mlx_closure_custom_vmap}, mlx_closure_custom_vmap), cls, src)
end

struct mlx_vector_int_
    ctx::Ptr{Cvoid}
end

const mlx_vector_int = mlx_vector_int_

function mlx_closure_custom_vmap_apply(res_0, res_1, cls, input_0, input_1, input_1_num)
    ccall((:mlx_closure_custom_vmap_apply, libmlxc), Cint, (Ptr{mlx_vector_array}, Ptr{mlx_vector_int}, mlx_closure_custom_vmap, mlx_vector_array, Ptr{Cint}, Csize_t), res_0, res_1, cls, input_0, input_1, input_1_num)
end

@cenum mlx_compile_mode_::UInt32 begin
    MLX_COMPILE_MODE_DISABLED = 0
    MLX_COMPILE_MODE_NO_SIMPLIFY = 1
    MLX_COMPILE_MODE_NO_FUSE = 2
    MLX_COMPILE_MODE_ENABLED = 3
end

const mlx_compile_mode = mlx_compile_mode_

function mlx_compile(res, fun, shapeless)
    ccall((:mlx_compile, libmlxc), Cint, (Ptr{mlx_closure}, mlx_closure, Bool), res, fun, shapeless)
end

function mlx_detail_compile(res, fun, fun_id, shapeless, constants, constants_num)
    ccall((:mlx_detail_compile, libmlxc), Cint, (Ptr{mlx_closure}, mlx_closure, Csize_t, Bool, Ptr{UInt64}, Csize_t), res, fun, fun_id, shapeless, constants, constants_num)
end

# no prototype is found for this function at compile.h:42:5, please use with caution
function mlx_detail_compile_clear_cache()
    ccall((:mlx_detail_compile_clear_cache, libmlxc), Cint, ())
end

function mlx_detail_compile_erase(fun_id)
    ccall((:mlx_detail_compile_erase, libmlxc), Cint, (Csize_t,), fun_id)
end

# no prototype is found for this function at compile.h:44:5, please use with caution
function mlx_disable_compile()
    ccall((:mlx_disable_compile, libmlxc), Cint, ())
end

# no prototype is found for this function at compile.h:45:5, please use with caution
function mlx_enable_compile()
    ccall((:mlx_enable_compile, libmlxc), Cint, ())
end

function mlx_set_compile_mode(mode)
    ccall((:mlx_set_compile_mode, libmlxc), Cint, (mlx_compile_mode,), mode)
end

struct mlx_device_
    ctx::Ptr{Cvoid}
end

const mlx_device = mlx_device_

@cenum mlx_device_type_::UInt32 begin
    MLX_CPU = 0
    MLX_GPU = 1
end

const mlx_device_type = mlx_device_type_

# no prototype is found for this function at device.h:33:12, please use with caution
function mlx_device_new()
    ccall((:mlx_device_new, libmlxc), mlx_device, ())
end

function mlx_device_new_type(type, index)
    ccall((:mlx_device_new_type, libmlxc), mlx_device, (mlx_device_type, Cint), type, index)
end

function mlx_device_free(dev)
    ccall((:mlx_device_free, libmlxc), Cint, (mlx_device,), dev)
end

function mlx_device_set(dev, src)
    ccall((:mlx_device_set, libmlxc), Cint, (Ptr{mlx_device}, mlx_device), dev, src)
end

function mlx_device_tostring(str, dev)
    ccall((:mlx_device_tostring, libmlxc), Cint, (Ptr{mlx_string}, mlx_device), str, dev)
end

function mlx_device_equal(lhs, rhs)
    ccall((:mlx_device_equal, libmlxc), Cint, (mlx_device, mlx_device), lhs, rhs)
end

function mlx_device_get_index(index, dev)
    ccall((:mlx_device_get_index, libmlxc), Cint, (Ptr{Cint}, mlx_device), index, dev)
end

function mlx_device_get_type(type, dev)
    ccall((:mlx_device_get_type, libmlxc), Cint, (Ptr{mlx_device_type}, mlx_device), type, dev)
end

function mlx_get_default_device(dev)
    ccall((:mlx_get_default_device, libmlxc), Cint, (Ptr{mlx_device},), dev)
end

function mlx_set_default_device(dev)
    ccall((:mlx_set_default_device, libmlxc), Cint, (mlx_device,), dev)
end

struct mlx_distributed_group_
    ctx::Ptr{Cvoid}
end

const mlx_distributed_group = mlx_distributed_group_

struct mlx_stream_
    ctx::Ptr{Cvoid}
end

const mlx_stream = mlx_stream_

function mlx_distributed_all_gather(res, x, group, S)
    ccall((:mlx_distributed_all_gather, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_distributed_group, mlx_stream), res, x, group, S)
end

function mlx_distributed_all_sum(res, x, group, s)
    ccall((:mlx_distributed_all_sum, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_distributed_group, mlx_stream), res, x, group, s)
end

function mlx_distributed_recv(res, shape, shape_num, dtype, src, group, s)
    ccall((:mlx_distributed_recv, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cint}, Csize_t, mlx_dtype, Cint, mlx_distributed_group, mlx_stream), res, shape, shape_num, dtype, src, group, s)
end

function mlx_distributed_recv_like(res, x, src, group, s)
    ccall((:mlx_distributed_recv_like, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_distributed_group, mlx_stream), res, x, src, group, s)
end

function mlx_distributed_send(res, x, dst, group, s)
    ccall((:mlx_distributed_send, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_distributed_group, mlx_stream), res, x, dst, group, s)
end

function mlx_distributed_group_rank(group)
    ccall((:mlx_distributed_group_rank, libmlxc), Cint, (mlx_distributed_group,), group)
end

function mlx_distributed_group_size(group)
    ccall((:mlx_distributed_group_size, libmlxc), Cint, (mlx_distributed_group,), group)
end

function mlx_distributed_group_split(group, color, key)
    ccall((:mlx_distributed_group_split, libmlxc), mlx_distributed_group, (mlx_distributed_group, Cint, Cint), group, color, key)
end

# no prototype is found for this function at distributed_group.h:43:6, please use with caution
function mlx_distributed_is_available()
    ccall((:mlx_distributed_is_available, libmlxc), Bool, ())
end

function mlx_distributed_init(strict)
    ccall((:mlx_distributed_init, libmlxc), mlx_distributed_group, (Bool,), strict)
end

# typedef void ( * mlx_error_handler_func ) ( const char * msg , void * data )
const mlx_error_handler_func = Ptr{Cvoid}

function mlx_set_error_handler(handler, data, dtor)
    ccall((:mlx_set_error_handler, libmlxc), Cvoid, (mlx_error_handler_func, Ptr{Cvoid}, Ptr{Cvoid}), handler, data, dtor)
end

function mlx_fast_affine_dequantize(res, w, scales, biases, group_size, bits, s)
    ccall((:mlx_fast_affine_dequantize, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, Cint, Cint, mlx_stream), res, w, scales, biases, group_size, bits, s)
end

function mlx_fast_affine_quantize(res_0, res_1, res_2, w, group_size, bits, s)
    ccall((:mlx_fast_affine_quantize, libmlxc), Cint, (Ptr{mlx_array}, Ptr{mlx_array}, Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res_0, res_1, res_2, w, group_size, bits, s)
end

function mlx_fast_layer_norm(res, x, weight, bias, eps, s)
    ccall((:mlx_fast_layer_norm, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, Cfloat, mlx_stream), res, x, weight, bias, eps, s)
end

struct mlx_fast_metal_kernel_
    ctx::Ptr{Cvoid}
end

const mlx_fast_metal_kernel = mlx_fast_metal_kernel_

function mlx_fast_metal_kernel_new(name, source, header)
    ccall((:mlx_fast_metal_kernel_new, libmlxc), mlx_fast_metal_kernel, (Ptr{Cchar}, Ptr{Cchar}, Ptr{Cchar}), name, source, header)
end

function mlx_fast_metal_kernel_free(cls)
    ccall((:mlx_fast_metal_kernel_free, libmlxc), Cvoid, (mlx_fast_metal_kernel,), cls)
end

function mlx_fast_metal_kernel_add_input_name(cls, name)
    ccall((:mlx_fast_metal_kernel_add_input_name, libmlxc), Cint, (mlx_fast_metal_kernel, Ptr{Cchar}), cls, name)
end

function mlx_fast_metal_kernel_add_output_name(cls, name)
    ccall((:mlx_fast_metal_kernel_add_output_name, libmlxc), Cint, (mlx_fast_metal_kernel, Ptr{Cchar}), cls, name)
end

function mlx_fast_metal_kernel_set_contiguous_rows(cls, flag)
    ccall((:mlx_fast_metal_kernel_set_contiguous_rows, libmlxc), Cint, (mlx_fast_metal_kernel, Bool), cls, flag)
end

function mlx_fast_metal_kernel_set_atomic_outputs(cls, flag)
    ccall((:mlx_fast_metal_kernel_set_atomic_outputs, libmlxc), Cint, (mlx_fast_metal_kernel, Bool), cls, flag)
end

function mlx_fast_metal_kernel_add_output_arg(cls, shape, size, dtype)
    ccall((:mlx_fast_metal_kernel_add_output_arg, libmlxc), Cint, (mlx_fast_metal_kernel, Ptr{Cint}, Csize_t, mlx_dtype), cls, shape, size, dtype)
end

function mlx_fast_metal_kernel_set_grid(cls, grid1, grid2, grid3)
    ccall((:mlx_fast_metal_kernel_set_grid, libmlxc), Cint, (mlx_fast_metal_kernel, Cint, Cint, Cint), cls, grid1, grid2, grid3)
end

function mlx_fast_metal_kernel_set_thread_group(cls, thread1, thread2, thread3)
    ccall((:mlx_fast_metal_kernel_set_thread_group, libmlxc), Cint, (mlx_fast_metal_kernel, Cint, Cint, Cint), cls, thread1, thread2, thread3)
end

function mlx_fast_metal_kernel_set_init_value(cls, value)
    ccall((:mlx_fast_metal_kernel_set_init_value, libmlxc), Cint, (mlx_fast_metal_kernel, Cfloat), cls, value)
end

function mlx_fast_metal_kernel_set_verbose(cls, verbose)
    ccall((:mlx_fast_metal_kernel_set_verbose, libmlxc), Cint, (mlx_fast_metal_kernel, Bool), cls, verbose)
end

function mlx_fast_metal_kernel_add_template_arg_dtype(cls, name, dtype)
    ccall((:mlx_fast_metal_kernel_add_template_arg_dtype, libmlxc), Cint, (mlx_fast_metal_kernel, Ptr{Cchar}, mlx_dtype), cls, name, dtype)
end

function mlx_fast_metal_kernel_add_template_arg_int(cls, name, value)
    ccall((:mlx_fast_metal_kernel_add_template_arg_int, libmlxc), Cint, (mlx_fast_metal_kernel, Ptr{Cchar}, Cint), cls, name, value)
end

function mlx_fast_metal_kernel_add_template_arg_bool(cls, name, value)
    ccall((:mlx_fast_metal_kernel_add_template_arg_bool, libmlxc), Cint, (mlx_fast_metal_kernel, Ptr{Cchar}, Bool), cls, name, value)
end

function mlx_fast_metal_kernel_apply(outputs, cls, inputs, stream)
    ccall((:mlx_fast_metal_kernel_apply, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_fast_metal_kernel, mlx_vector_array, mlx_stream), outputs, cls, inputs, stream)
end

function mlx_fast_rms_norm(res, x, weight, eps, s)
    ccall((:mlx_fast_rms_norm, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cfloat, mlx_stream), res, x, weight, eps, s)
end

struct mlx_optional_float_
    value::Cfloat
    has_value::Bool
end

const mlx_optional_float = mlx_optional_float_

function mlx_fast_rope(res, x, dims, traditional, base, scale, offset, freqs, s)
    ccall((:mlx_fast_rope, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Bool, mlx_optional_float, Cfloat, Cint, mlx_array, mlx_stream), res, x, dims, traditional, base, scale, offset, freqs, s)
end

struct mlx_optional_int_
    value::Cint
    has_value::Bool
end

const mlx_optional_int = mlx_optional_int_

function mlx_fast_scaled_dot_product_attention(res, queries, keys, values, scale, mask, memory_efficient_threshold, s)
    ccall((:mlx_fast_scaled_dot_product_attention, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, Cfloat, mlx_array, mlx_optional_int, mlx_stream), res, queries, keys, values, scale, mask, memory_efficient_threshold, s)
end

function mlx_fft_fft(res, a, n, axis, s)
    ccall((:mlx_fft_fft, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, a, n, axis, s)
end

function mlx_fft_fft2(res, a, n, n_num, axes, axes_num, s)
    ccall((:mlx_fft_fft2, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, n, n_num, axes, axes_num, s)
end

function mlx_fft_fftn(res, a, n, n_num, axes, axes_num, s)
    ccall((:mlx_fft_fftn, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, n, n_num, axes, axes_num, s)
end

function mlx_fft_ifft(res, a, n, axis, s)
    ccall((:mlx_fft_ifft, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, a, n, axis, s)
end

function mlx_fft_ifft2(res, a, n, n_num, axes, axes_num, s)
    ccall((:mlx_fft_ifft2, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, n, n_num, axes, axes_num, s)
end

function mlx_fft_ifftn(res, a, n, n_num, axes, axes_num, s)
    ccall((:mlx_fft_ifftn, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, n, n_num, axes, axes_num, s)
end

function mlx_fft_irfft(res, a, n, axis, s)
    ccall((:mlx_fft_irfft, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, a, n, axis, s)
end

function mlx_fft_irfft2(res, a, n, n_num, axes, axes_num, s)
    ccall((:mlx_fft_irfft2, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, n, n_num, axes, axes_num, s)
end

function mlx_fft_irfftn(res, a, n, n_num, axes, axes_num, s)
    ccall((:mlx_fft_irfftn, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, n, n_num, axes, axes_num, s)
end

function mlx_fft_rfft(res, a, n, axis, s)
    ccall((:mlx_fft_rfft, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, a, n, axis, s)
end

function mlx_fft_rfft2(res, a, n, n_num, axes, axes_num, s)
    ccall((:mlx_fft_rfft2, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, n, n_num, axes, axes_num, s)
end

function mlx_fft_rfftn(res, a, n, n_num, axes, axes_num, s)
    ccall((:mlx_fft_rfftn, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, n, n_num, axes, axes_num, s)
end

function mlx_load_file(res, in_stream, s)
    ccall((:mlx_load_file, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Libc.FILE}, mlx_stream), res, in_stream, s)
end

function mlx_load(res, file, s)
    ccall((:mlx_load, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cchar}, mlx_stream), res, file, s)
end

struct mlx_map_string_to_array_
    ctx::Ptr{Cvoid}
end

const mlx_map_string_to_array = mlx_map_string_to_array_

struct mlx_map_string_to_string_
    ctx::Ptr{Cvoid}
end

const mlx_map_string_to_string = mlx_map_string_to_string_

function mlx_load_safetensors_file(res_0, res_1, in_stream, s)
    ccall((:mlx_load_safetensors_file, libmlxc), Cint, (Ptr{mlx_map_string_to_array}, Ptr{mlx_map_string_to_string}, Ptr{Libc.FILE}, mlx_stream), res_0, res_1, in_stream, s)
end

function mlx_load_safetensors(res_0, res_1, file, s)
    ccall((:mlx_load_safetensors, libmlxc), Cint, (Ptr{mlx_map_string_to_array}, Ptr{mlx_map_string_to_string}, Ptr{Cchar}, mlx_stream), res_0, res_1, file, s)
end

function mlx_save_file(out_stream, a)
    ccall((:mlx_save_file, libmlxc), Cint, (Ptr{Libc.FILE}, mlx_array), out_stream, a)
end

function mlx_save(file, a)
    ccall((:mlx_save, libmlxc), Cint, (Ptr{Cchar}, mlx_array), file, a)
end

function mlx_save_safetensors_file(in_stream, param, metadata)
    ccall((:mlx_save_safetensors_file, libmlxc), Cint, (Ptr{Libc.FILE}, mlx_map_string_to_array, mlx_map_string_to_string), in_stream, param, metadata)
end

function mlx_save_safetensors(file, param, metadata)
    ccall((:mlx_save_safetensors, libmlxc), Cint, (Ptr{Cchar}, mlx_map_string_to_array, mlx_map_string_to_string), file, param, metadata)
end

function mlx_linalg_cholesky(res, a, upper, s)
    ccall((:mlx_linalg_cholesky, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, upper, s)
end

function mlx_linalg_cholesky_inv(res, a, upper, s)
    ccall((:mlx_linalg_cholesky_inv, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, upper, s)
end

function mlx_linalg_cross(res, a, b, axis, s)
    ccall((:mlx_linalg_cross, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, mlx_stream), res, a, b, axis, s)
end

function mlx_linalg_eigh(res_0, res_1, a, UPLO, s)
    ccall((:mlx_linalg_eigh, libmlxc), Cint, (Ptr{mlx_array}, Ptr{mlx_array}, mlx_array, Ptr{Cchar}, mlx_stream), res_0, res_1, a, UPLO, s)
end

function mlx_linalg_eigvalsh(res, a, UPLO, s)
    ccall((:mlx_linalg_eigvalsh, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cchar}, mlx_stream), res, a, UPLO, s)
end

function mlx_linalg_inv(res, a, s)
    ccall((:mlx_linalg_inv, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_linalg_lu(res, a, s)
    ccall((:mlx_linalg_lu, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_linalg_lu_factor(res_0, res_1, a, s)
    ccall((:mlx_linalg_lu_factor, libmlxc), Cint, (Ptr{mlx_array}, Ptr{mlx_array}, mlx_array, mlx_stream), res_0, res_1, a, s)
end

function mlx_linalg_norm_p(res, a, ord, axis, axis_num, keepdims, s)
    ccall((:mlx_linalg_norm_p, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cdouble, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, ord, axis, axis_num, keepdims, s)
end

function mlx_linalg_norm_ord(res, a, ord, axis, axis_num, keepdims, s)
    ccall((:mlx_linalg_norm_ord, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cchar}, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, ord, axis, axis_num, keepdims, s)
end

function mlx_linalg_norm(res, a, axis, axis_num, keepdims, s)
    ccall((:mlx_linalg_norm, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, axis, axis_num, keepdims, s)
end

function mlx_linalg_pinv(res, a, s)
    ccall((:mlx_linalg_pinv, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_linalg_qr(res_0, res_1, a, s)
    ccall((:mlx_linalg_qr, libmlxc), Cint, (Ptr{mlx_array}, Ptr{mlx_array}, mlx_array, mlx_stream), res_0, res_1, a, s)
end

function mlx_linalg_solve(res, a, b, s)
    ccall((:mlx_linalg_solve, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_linalg_solve_triangular(res, a, b, upper, s)
    ccall((:mlx_linalg_solve_triangular, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Bool, mlx_stream), res, a, b, upper, s)
end

function mlx_linalg_svd(res, a, s)
    ccall((:mlx_linalg_svd, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_linalg_tri_inv(res, a, upper, s)
    ccall((:mlx_linalg_tri_inv, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, upper, s)
end

function mlx_map_string_to_array_new()
    ccall((:mlx_map_string_to_array_new, libmlxc), mlx_map_string_to_array, ())
end

function mlx_map_string_to_array_set(map, src)
    ccall((:mlx_map_string_to_array_set, libmlxc), Cint, (Ptr{mlx_map_string_to_array}, mlx_map_string_to_array), map, src)
end

function mlx_map_string_to_array_free(map)
    ccall((:mlx_map_string_to_array_free, libmlxc), Cint, (mlx_map_string_to_array,), map)
end

function mlx_map_string_to_array_insert(map, key, value)
    ccall((:mlx_map_string_to_array_insert, libmlxc), Cint, (mlx_map_string_to_array, Ptr{Cchar}, mlx_array), map, key, value)
end

function mlx_map_string_to_array_get(value, map, key)
    ccall((:mlx_map_string_to_array_get, libmlxc), Cint, (Ptr{mlx_array}, mlx_map_string_to_array, Ptr{Cchar}), value, map, key)
end

struct mlx_map_string_to_array_iterator_
    ctx::Ptr{Cvoid}
    map_ctx::Ptr{Cvoid}
end

const mlx_map_string_to_array_iterator = mlx_map_string_to_array_iterator_

function mlx_map_string_to_array_iterator_new(map)
    ccall((:mlx_map_string_to_array_iterator_new, libmlxc), mlx_map_string_to_array_iterator, (mlx_map_string_to_array,), map)
end

function mlx_map_string_to_array_iterator_free(it)
    ccall((:mlx_map_string_to_array_iterator_free, libmlxc), Cint, (mlx_map_string_to_array_iterator,), it)
end

function mlx_map_string_to_array_iterator_next(key, value, it)
    ccall((:mlx_map_string_to_array_iterator_next, libmlxc), Cint, (Ptr{Ptr{Cchar}}, Ptr{mlx_array}, mlx_map_string_to_array_iterator), key, value, it)
end

function mlx_map_string_to_string_new()
    ccall((:mlx_map_string_to_string_new, libmlxc), mlx_map_string_to_string, ())
end

function mlx_map_string_to_string_set(map, src)
    ccall((:mlx_map_string_to_string_set, libmlxc), Cint, (Ptr{mlx_map_string_to_string}, mlx_map_string_to_string), map, src)
end

function mlx_map_string_to_string_free(map)
    ccall((:mlx_map_string_to_string_free, libmlxc), Cint, (mlx_map_string_to_string,), map)
end

function mlx_map_string_to_string_insert(map, key, value)
    ccall((:mlx_map_string_to_string_insert, libmlxc), Cint, (mlx_map_string_to_string, Ptr{Cchar}, Ptr{Cchar}), map, key, value)
end

function mlx_map_string_to_string_get(value, map, key)
    ccall((:mlx_map_string_to_string_get, libmlxc), Cint, (Ptr{Ptr{Cchar}}, mlx_map_string_to_string, Ptr{Cchar}), value, map, key)
end

struct mlx_map_string_to_string_iterator_
    ctx::Ptr{Cvoid}
    map_ctx::Ptr{Cvoid}
end

const mlx_map_string_to_string_iterator = mlx_map_string_to_string_iterator_

function mlx_map_string_to_string_iterator_new(map)
    ccall((:mlx_map_string_to_string_iterator_new, libmlxc), mlx_map_string_to_string_iterator, (mlx_map_string_to_string,), map)
end

function mlx_map_string_to_string_iterator_free(it)
    ccall((:mlx_map_string_to_string_iterator_free, libmlxc), Cint, (mlx_map_string_to_string_iterator,), it)
end

function mlx_map_string_to_string_iterator_next(key, value, it)
    ccall((:mlx_map_string_to_string_iterator_next, libmlxc), Cint, (Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, mlx_map_string_to_string_iterator), key, value, it)
end

# no prototype is found for this function at metal.h:28:5, please use with caution
function mlx_metal_clear_cache()
    ccall((:mlx_metal_clear_cache, libmlxc), Cint, ())
end

struct mlx_metal_device_info_t_
    architecture::NTuple{256, Cchar}
    max_buffer_length::Csize_t
    max_recommended_working_set_size::Csize_t
    memory_size::Csize_t
end

const mlx_metal_device_info_t = mlx_metal_device_info_t_

# no prototype is found for this function at metal.h:36:25, please use with caution
function mlx_metal_device_info()
    ccall((:mlx_metal_device_info, libmlxc), mlx_metal_device_info_t, ())
end

function mlx_metal_get_active_memory(res)
    ccall((:mlx_metal_get_active_memory, libmlxc), Cint, (Ptr{Csize_t},), res)
end

function mlx_metal_get_cache_memory(res)
    ccall((:mlx_metal_get_cache_memory, libmlxc), Cint, (Ptr{Csize_t},), res)
end

function mlx_metal_get_peak_memory(res)
    ccall((:mlx_metal_get_peak_memory, libmlxc), Cint, (Ptr{Csize_t},), res)
end

function mlx_metal_is_available(res)
    ccall((:mlx_metal_is_available, libmlxc), Cint, (Ptr{Bool},), res)
end

# no prototype is found for this function at metal.h:42:5, please use with caution
function mlx_metal_reset_peak_memory()
    ccall((:mlx_metal_reset_peak_memory, libmlxc), Cint, ())
end

function mlx_metal_set_cache_limit(res, limit)
    ccall((:mlx_metal_set_cache_limit, libmlxc), Cint, (Ptr{Csize_t}, Csize_t), res, limit)
end

function mlx_metal_set_memory_limit(res, limit, relaxed)
    ccall((:mlx_metal_set_memory_limit, libmlxc), Cint, (Ptr{Csize_t}, Csize_t, Bool), res, limit, relaxed)
end

function mlx_metal_set_wired_limit(res, limit)
    ccall((:mlx_metal_set_wired_limit, libmlxc), Cint, (Ptr{Csize_t}, Csize_t), res, limit)
end

function mlx_metal_start_capture(path)
    ccall((:mlx_metal_start_capture, libmlxc), Cint, (Ptr{Cchar},), path)
end

# no prototype is found for this function at metal.h:47:5, please use with caution
function mlx_metal_stop_capture()
    ccall((:mlx_metal_stop_capture, libmlxc), Cint, ())
end

function mlx_abs(res, a, s)
    ccall((:mlx_abs, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_add(res, a, b, s)
    ccall((:mlx_add, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_addmm(res, c, a, b, alpha, beta, s)
    ccall((:mlx_addmm, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, Cfloat, Cfloat, mlx_stream), res, c, a, b, alpha, beta, s)
end

function mlx_all_axes(res, a, axes, axes_num, keepdims, s)
    ccall((:mlx_all_axes, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, axes, axes_num, keepdims, s)
end

function mlx_all_axis(res, a, axis, keepdims, s)
    ccall((:mlx_all_axis, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Bool, mlx_stream), res, a, axis, keepdims, s)
end

function mlx_all_all(res, a, keepdims, s)
    ccall((:mlx_all_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, keepdims, s)
end

function mlx_allclose(res, a, b, rtol, atol, equal_nan, s)
    ccall((:mlx_allclose, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cdouble, Cdouble, Bool, mlx_stream), res, a, b, rtol, atol, equal_nan, s)
end

function mlx_any(res, a, axes, axes_num, keepdims, s)
    ccall((:mlx_any, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, axes, axes_num, keepdims, s)
end

function mlx_any_all(res, a, keepdims, s)
    ccall((:mlx_any_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, keepdims, s)
end

function mlx_arange(res, start, stop, step, dtype, s)
    ccall((:mlx_arange, libmlxc), Cint, (Ptr{mlx_array}, Cdouble, Cdouble, Cdouble, mlx_dtype, mlx_stream), res, start, stop, step, dtype, s)
end

function mlx_arccos(res, a, s)
    ccall((:mlx_arccos, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_arccosh(res, a, s)
    ccall((:mlx_arccosh, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_arcsin(res, a, s)
    ccall((:mlx_arcsin, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_arcsinh(res, a, s)
    ccall((:mlx_arcsinh, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_arctan(res, a, s)
    ccall((:mlx_arctan, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_arctan2(res, a, b, s)
    ccall((:mlx_arctan2, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_arctanh(res, a, s)
    ccall((:mlx_arctanh, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_argmax(res, a, axis, keepdims, s)
    ccall((:mlx_argmax, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Bool, mlx_stream), res, a, axis, keepdims, s)
end

function mlx_argmax_all(res, a, keepdims, s)
    ccall((:mlx_argmax_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, keepdims, s)
end

function mlx_argmin(res, a, axis, keepdims, s)
    ccall((:mlx_argmin, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Bool, mlx_stream), res, a, axis, keepdims, s)
end

function mlx_argmin_all(res, a, keepdims, s)
    ccall((:mlx_argmin_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, keepdims, s)
end

function mlx_argpartition(res, a, kth, axis, s)
    ccall((:mlx_argpartition, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, a, kth, axis, s)
end

function mlx_argpartition_all(res, a, kth, s)
    ccall((:mlx_argpartition_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, a, kth, s)
end

function mlx_argsort(res, a, axis, s)
    ccall((:mlx_argsort, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, a, axis, s)
end

function mlx_argsort_all(res, a, s)
    ccall((:mlx_argsort_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_array_equal(res, a, b, equal_nan, s)
    ccall((:mlx_array_equal, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Bool, mlx_stream), res, a, b, equal_nan, s)
end

function mlx_as_strided(res, a, shape, shape_num, strides, strides_num, offset, s)
    ccall((:mlx_as_strided, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Int64}, Csize_t, Csize_t, mlx_stream), res, a, shape, shape_num, strides, strides_num, offset, s)
end

function mlx_astype(res, a, dtype, s)
    ccall((:mlx_astype, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_dtype, mlx_stream), res, a, dtype, s)
end

function mlx_atleast_1d(res, a, s)
    ccall((:mlx_atleast_1d, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_atleast_2d(res, a, s)
    ccall((:mlx_atleast_2d, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_atleast_3d(res, a, s)
    ccall((:mlx_atleast_3d, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_bitwise_and(res, a, b, s)
    ccall((:mlx_bitwise_and, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_bitwise_invert(res, a, s)
    ccall((:mlx_bitwise_invert, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_bitwise_or(res, a, b, s)
    ccall((:mlx_bitwise_or, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_bitwise_xor(res, a, b, s)
    ccall((:mlx_bitwise_xor, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_block_masked_mm(res, a, b, block_size, mask_out, mask_lhs, mask_rhs, s)
    ccall((:mlx_block_masked_mm, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, mlx_array, mlx_array, mlx_array, mlx_stream), res, a, b, block_size, mask_out, mask_lhs, mask_rhs, s)
end

function mlx_broadcast_arrays(res, inputs, s)
    ccall((:mlx_broadcast_arrays, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_vector_array, mlx_stream), res, inputs, s)
end

function mlx_broadcast_to(res, a, shape, shape_num, s)
    ccall((:mlx_broadcast_to, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, a, shape, shape_num, s)
end

function mlx_ceil(res, a, s)
    ccall((:mlx_ceil, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_clip(res, a, a_min, a_max, s)
    ccall((:mlx_clip, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, mlx_stream), res, a, a_min, a_max, s)
end

function mlx_concatenate(res, arrays, axis, s)
    ccall((:mlx_concatenate, libmlxc), Cint, (Ptr{mlx_array}, mlx_vector_array, Cint, mlx_stream), res, arrays, axis, s)
end

function mlx_concatenate_all(res, arrays, s)
    ccall((:mlx_concatenate_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_vector_array, mlx_stream), res, arrays, s)
end

function mlx_conjugate(res, a, s)
    ccall((:mlx_conjugate, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_contiguous(res, a, allow_col_major, s)
    ccall((:mlx_contiguous, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, allow_col_major, s)
end

function mlx_conv1d(res, input, weight, stride, padding, dilation, groups, s)
    ccall((:mlx_conv1d, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, Cint, Cint, Cint, mlx_stream), res, input, weight, stride, padding, dilation, groups, s)
end

function mlx_conv2d(res, input, weight, stride_0, stride_1, padding_0, padding_1, dilation_0, dilation_1, groups, s)
    ccall((:mlx_conv2d, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, Cint, Cint, Cint, Cint, Cint, Cint, mlx_stream), res, input, weight, stride_0, stride_1, padding_0, padding_1, dilation_0, dilation_1, groups, s)
end

function mlx_conv3d(res, input, weight, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, groups, s)
    ccall((:mlx_conv3d, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, mlx_stream), res, input, weight, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, groups, s)
end

function mlx_conv_general(res, input, weight, stride, stride_num, padding_lo, padding_lo_num, padding_hi, padding_hi_num, kernel_dilation, kernel_dilation_num, input_dilation, input_dilation_num, groups, flip, s)
    ccall((:mlx_conv_general, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, Cint, Bool, mlx_stream), res, input, weight, stride, stride_num, padding_lo, padding_lo_num, padding_hi, padding_hi_num, kernel_dilation, kernel_dilation_num, input_dilation, input_dilation_num, groups, flip, s)
end

function mlx_conv_transpose1d(res, input, weight, stride, padding, dilation, groups, s)
    ccall((:mlx_conv_transpose1d, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, Cint, Cint, Cint, mlx_stream), res, input, weight, stride, padding, dilation, groups, s)
end

function mlx_conv_transpose2d(res, input, weight, stride_0, stride_1, padding_0, padding_1, dilation_0, dilation_1, groups, s)
    ccall((:mlx_conv_transpose2d, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, Cint, Cint, Cint, Cint, Cint, Cint, mlx_stream), res, input, weight, stride_0, stride_1, padding_0, padding_1, dilation_0, dilation_1, groups, s)
end

function mlx_conv_transpose3d(res, input, weight, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, groups, s)
    ccall((:mlx_conv_transpose3d, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, mlx_stream), res, input, weight, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, groups, s)
end

function mlx_copy(res, a, s)
    ccall((:mlx_copy, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_cos(res, a, s)
    ccall((:mlx_cos, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_cosh(res, a, s)
    ccall((:mlx_cosh, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_cummax(res, a, axis, reverse, inclusive, s)
    ccall((:mlx_cummax, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Bool, Bool, mlx_stream), res, a, axis, reverse, inclusive, s)
end

function mlx_cummin(res, a, axis, reverse, inclusive, s)
    ccall((:mlx_cummin, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Bool, Bool, mlx_stream), res, a, axis, reverse, inclusive, s)
end

function mlx_cumprod(res, a, axis, reverse, inclusive, s)
    ccall((:mlx_cumprod, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Bool, Bool, mlx_stream), res, a, axis, reverse, inclusive, s)
end

function mlx_cumsum(res, a, axis, reverse, inclusive, s)
    ccall((:mlx_cumsum, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Bool, Bool, mlx_stream), res, a, axis, reverse, inclusive, s)
end

function mlx_degrees(res, a, s)
    ccall((:mlx_degrees, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_depends(res, inputs, dependencies)
    ccall((:mlx_depends, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_vector_array, mlx_vector_array), res, inputs, dependencies)
end

function mlx_dequantize(res, w, scales, biases, group_size, bits, s)
    ccall((:mlx_dequantize, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, Cint, Cint, mlx_stream), res, w, scales, biases, group_size, bits, s)
end

function mlx_diag(res, a, k, s)
    ccall((:mlx_diag, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, a, k, s)
end

function mlx_diagonal(res, a, offset, axis1, axis2, s)
    ccall((:mlx_diagonal, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, Cint, mlx_stream), res, a, offset, axis1, axis2, s)
end

function mlx_divide(res, a, b, s)
    ccall((:mlx_divide, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_divmod(res, a, b, s)
    ccall((:mlx_divmod, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_einsum(res, subscripts, operands, s)
    ccall((:mlx_einsum, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cchar}, mlx_vector_array, mlx_stream), res, subscripts, operands, s)
end

function mlx_equal(res, a, b, s)
    ccall((:mlx_equal, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_erf(res, a, s)
    ccall((:mlx_erf, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_erfinv(res, a, s)
    ccall((:mlx_erfinv, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_exp(res, a, s)
    ccall((:mlx_exp, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_expand_dims(res, a, axes, axes_num, s)
    ccall((:mlx_expand_dims, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, a, axes, axes_num, s)
end

function mlx_expm1(res, a, s)
    ccall((:mlx_expm1, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_eye(res, n, m, k, dtype, s)
    ccall((:mlx_eye, libmlxc), Cint, (Ptr{mlx_array}, Cint, Cint, Cint, mlx_dtype, mlx_stream), res, n, m, k, dtype, s)
end

function mlx_flatten(res, a, start_axis, end_axis, s)
    ccall((:mlx_flatten, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, a, start_axis, end_axis, s)
end

function mlx_floor(res, a, s)
    ccall((:mlx_floor, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_floor_divide(res, a, b, s)
    ccall((:mlx_floor_divide, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_full(res, shape, shape_num, vals, dtype, s)
    ccall((:mlx_full, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cint}, Csize_t, mlx_array, mlx_dtype, mlx_stream), res, shape, shape_num, vals, dtype, s)
end

function mlx_gather(res, a, indices, axes, axes_num, slice_sizes, slice_sizes_num, s)
    ccall((:mlx_gather, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_vector_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, indices, axes, axes_num, slice_sizes, slice_sizes_num, s)
end

function mlx_gather_mm(res, a, b, lhs_indices, rhs_indices, s)
    ccall((:mlx_gather_mm, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, mlx_array, mlx_stream), res, a, b, lhs_indices, rhs_indices, s)
end

function mlx_gather_qmm(res, x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits, s)
    ccall((:mlx_gather_qmm, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, Bool, Cint, Cint, mlx_stream), res, x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits, s)
end

function mlx_greater(res, a, b, s)
    ccall((:mlx_greater, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_greater_equal(res, a, b, s)
    ccall((:mlx_greater_equal, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_hadamard_transform(res, a, scale, s)
    ccall((:mlx_hadamard_transform, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_optional_float, mlx_stream), res, a, scale, s)
end

function mlx_identity(res, n, dtype, s)
    ccall((:mlx_identity, libmlxc), Cint, (Ptr{mlx_array}, Cint, mlx_dtype, mlx_stream), res, n, dtype, s)
end

function mlx_imag(res, a, s)
    ccall((:mlx_imag, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_inner(res, a, b, s)
    ccall((:mlx_inner, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_isclose(res, a, b, rtol, atol, equal_nan, s)
    ccall((:mlx_isclose, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cdouble, Cdouble, Bool, mlx_stream), res, a, b, rtol, atol, equal_nan, s)
end

function mlx_isfinite(res, a, s)
    ccall((:mlx_isfinite, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_isinf(res, a, s)
    ccall((:mlx_isinf, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_isnan(res, a, s)
    ccall((:mlx_isnan, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_isneginf(res, a, s)
    ccall((:mlx_isneginf, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_isposinf(res, a, s)
    ccall((:mlx_isposinf, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_kron(res, a, b, s)
    ccall((:mlx_kron, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_left_shift(res, a, b, s)
    ccall((:mlx_left_shift, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_less(res, a, b, s)
    ccall((:mlx_less, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_less_equal(res, a, b, s)
    ccall((:mlx_less_equal, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_linspace(res, start, stop, num, dtype, s)
    ccall((:mlx_linspace, libmlxc), Cint, (Ptr{mlx_array}, Cdouble, Cdouble, Cint, mlx_dtype, mlx_stream), res, start, stop, num, dtype, s)
end

function mlx_log(res, a, s)
    ccall((:mlx_log, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_log10(res, a, s)
    ccall((:mlx_log10, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_log1p(res, a, s)
    ccall((:mlx_log1p, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_log2(res, a, s)
    ccall((:mlx_log2, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_logaddexp(res, a, b, s)
    ccall((:mlx_logaddexp, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_logical_and(res, a, b, s)
    ccall((:mlx_logical_and, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_logical_not(res, a, s)
    ccall((:mlx_logical_not, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_logical_or(res, a, b, s)
    ccall((:mlx_logical_or, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_logsumexp(res, a, axes, axes_num, keepdims, s)
    ccall((:mlx_logsumexp, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, axes, axes_num, keepdims, s)
end

function mlx_logsumexp_all(res, a, keepdims, s)
    ccall((:mlx_logsumexp_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, keepdims, s)
end

function mlx_matmul(res, a, b, s)
    ccall((:mlx_matmul, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_max(res, a, axes, axes_num, keepdims, s)
    ccall((:mlx_max, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, axes, axes_num, keepdims, s)
end

function mlx_max_all(res, a, keepdims, s)
    ccall((:mlx_max_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, keepdims, s)
end

function mlx_maximum(res, a, b, s)
    ccall((:mlx_maximum, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_mean(res, a, axes, axes_num, keepdims, s)
    ccall((:mlx_mean, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, axes, axes_num, keepdims, s)
end

function mlx_mean_all(res, a, keepdims, s)
    ccall((:mlx_mean_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, keepdims, s)
end

function mlx_meshgrid(res, arrays, sparse, indexing, s)
    ccall((:mlx_meshgrid, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_vector_array, Bool, Ptr{Cchar}, mlx_stream), res, arrays, sparse, indexing, s)
end

function mlx_min(res, a, axes, axes_num, keepdims, s)
    ccall((:mlx_min, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, axes, axes_num, keepdims, s)
end

function mlx_min_all(res, a, keepdims, s)
    ccall((:mlx_min_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, keepdims, s)
end

function mlx_minimum(res, a, b, s)
    ccall((:mlx_minimum, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_moveaxis(res, a, source, destination, s)
    ccall((:mlx_moveaxis, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, a, source, destination, s)
end

function mlx_multiply(res, a, b, s)
    ccall((:mlx_multiply, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_nan_to_num(res, a, nan, posinf, neginf, s)
    ccall((:mlx_nan_to_num, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cfloat, mlx_optional_float, mlx_optional_float, mlx_stream), res, a, nan, posinf, neginf, s)
end

function mlx_negative(res, a, s)
    ccall((:mlx_negative, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_not_equal(res, a, b, s)
    ccall((:mlx_not_equal, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_number_of_elements(res, a, axes, axes_num, inverted, dtype, s)
    ccall((:mlx_number_of_elements, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_dtype, mlx_stream), res, a, axes, axes_num, inverted, dtype, s)
end

function mlx_ones(res, shape, shape_num, dtype, s)
    ccall((:mlx_ones, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cint}, Csize_t, mlx_dtype, mlx_stream), res, shape, shape_num, dtype, s)
end

function mlx_ones_like(res, a, s)
    ccall((:mlx_ones_like, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_outer(res, a, b, s)
    ccall((:mlx_outer, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_pad(res, a, axes, axes_num, low_pad_size, low_pad_size_num, high_pad_size, high_pad_size_num, pad_value, mode, s)
    ccall((:mlx_pad, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_array, Ptr{Cchar}, mlx_stream), res, a, axes, axes_num, low_pad_size, low_pad_size_num, high_pad_size, high_pad_size_num, pad_value, mode, s)
end

function mlx_partition(res, a, kth, axis, s)
    ccall((:mlx_partition, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, a, kth, axis, s)
end

function mlx_partition_all(res, a, kth, s)
    ccall((:mlx_partition_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, a, kth, s)
end

function mlx_power(res, a, b, s)
    ccall((:mlx_power, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_prod(res, a, axes, axes_num, keepdims, s)
    ccall((:mlx_prod, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, axes, axes_num, keepdims, s)
end

function mlx_prod_all(res, a, keepdims, s)
    ccall((:mlx_prod_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, keepdims, s)
end

function mlx_put_along_axis(res, a, indices, values, axis, s)
    ccall((:mlx_put_along_axis, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, Cint, mlx_stream), res, a, indices, values, axis, s)
end

function mlx_quantize(res_0, res_1, res_2, w, group_size, bits, s)
    ccall((:mlx_quantize, libmlxc), Cint, (Ptr{mlx_array}, Ptr{mlx_array}, Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res_0, res_1, res_2, w, group_size, bits, s)
end

function mlx_quantized_matmul(res, x, w, scales, biases, transpose, group_size, bits, s)
    ccall((:mlx_quantized_matmul, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, mlx_array, Bool, Cint, Cint, mlx_stream), res, x, w, scales, biases, transpose, group_size, bits, s)
end

function mlx_radians(res, a, s)
    ccall((:mlx_radians, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_real(res, a, s)
    ccall((:mlx_real, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_reciprocal(res, a, s)
    ccall((:mlx_reciprocal, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_remainder(res, a, b, s)
    ccall((:mlx_remainder, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_repeat(res, arr, repeats, axis, s)
    ccall((:mlx_repeat, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, arr, repeats, axis, s)
end

function mlx_repeat_all(res, arr, repeats, s)
    ccall((:mlx_repeat_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, arr, repeats, s)
end

function mlx_reshape(res, a, shape, shape_num, s)
    ccall((:mlx_reshape, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, a, shape, shape_num, s)
end

function mlx_right_shift(res, a, b, s)
    ccall((:mlx_right_shift, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_roll(res, a, shift, axes, axes_num, s)
    ccall((:mlx_roll, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Ptr{Cint}, Csize_t, mlx_stream), res, a, shift, axes, axes_num, s)
end

function mlx_roll_all(res, a, shift, s)
    ccall((:mlx_roll_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, a, shift, s)
end

function mlx_round(res, a, decimals, s)
    ccall((:mlx_round, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, a, decimals, s)
end

function mlx_rsqrt(res, a, s)
    ccall((:mlx_rsqrt, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_scatter(res, a, indices, updates, axes, axes_num, s)
    ccall((:mlx_scatter, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_vector_array, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, a, indices, updates, axes, axes_num, s)
end

function mlx_scatter_add(res, a, indices, updates, axes, axes_num, s)
    ccall((:mlx_scatter_add, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_vector_array, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, a, indices, updates, axes, axes_num, s)
end

function mlx_scatter_add_axis(res, a, indices, values, axis, s)
    ccall((:mlx_scatter_add_axis, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, Cint, mlx_stream), res, a, indices, values, axis, s)
end

function mlx_scatter_max(res, a, indices, updates, axes, axes_num, s)
    ccall((:mlx_scatter_max, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_vector_array, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, a, indices, updates, axes, axes_num, s)
end

function mlx_scatter_min(res, a, indices, updates, axes, axes_num, s)
    ccall((:mlx_scatter_min, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_vector_array, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, a, indices, updates, axes, axes_num, s)
end

function mlx_scatter_prod(res, a, indices, updates, axes, axes_num, s)
    ccall((:mlx_scatter_prod, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_vector_array, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, a, indices, updates, axes, axes_num, s)
end

function mlx_sigmoid(res, a, s)
    ccall((:mlx_sigmoid, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_sign(res, a, s)
    ccall((:mlx_sign, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_sin(res, a, s)
    ccall((:mlx_sin, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_sinh(res, a, s)
    ccall((:mlx_sinh, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_slice(res, a, start, start_num, stop, stop_num, strides, strides_num, s)
    ccall((:mlx_slice, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, start, start_num, stop, stop_num, strides, strides_num, s)
end

function mlx_slice_update(res, src, update, start, start_num, stop, stop_num, strides, strides_num, s)
    ccall((:mlx_slice_update, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, src, update, start, start_num, stop, stop_num, strides, strides_num, s)
end

function mlx_softmax(res, a, axes, axes_num, precise, s)
    ccall((:mlx_softmax, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, axes, axes_num, precise, s)
end

function mlx_softmax_all(res, a, precise, s)
    ccall((:mlx_softmax_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, precise, s)
end

function mlx_sort(res, a, axis, s)
    ccall((:mlx_sort, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, a, axis, s)
end

function mlx_sort_all(res, a, s)
    ccall((:mlx_sort_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_split_equal_parts(res, a, num_splits, axis, s)
    ccall((:mlx_split_equal_parts, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_array, Cint, Cint, mlx_stream), res, a, num_splits, axis, s)
end

function mlx_split(res, a, indices, indices_num, axis, s)
    ccall((:mlx_split, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_array, Ptr{Cint}, Csize_t, Cint, mlx_stream), res, a, indices, indices_num, axis, s)
end

function mlx_sqrt(res, a, s)
    ccall((:mlx_sqrt, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_square(res, a, s)
    ccall((:mlx_square, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_squeeze(res, a, axes, axes_num, s)
    ccall((:mlx_squeeze, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, a, axes, axes_num, s)
end

function mlx_squeeze_all(res, a, s)
    ccall((:mlx_squeeze_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_stack(res, arrays, axis, s)
    ccall((:mlx_stack, libmlxc), Cint, (Ptr{mlx_array}, mlx_vector_array, Cint, mlx_stream), res, arrays, axis, s)
end

function mlx_stack_all(res, arrays, s)
    ccall((:mlx_stack_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_vector_array, mlx_stream), res, arrays, s)
end

function mlx_std(res, a, axes, axes_num, keepdims, ddof, s)
    ccall((:mlx_std, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, Cint, mlx_stream), res, a, axes, axes_num, keepdims, ddof, s)
end

function mlx_std_all(res, a, keepdims, ddof, s)
    ccall((:mlx_std_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, Cint, mlx_stream), res, a, keepdims, ddof, s)
end

function mlx_stop_gradient(res, a, s)
    ccall((:mlx_stop_gradient, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_subtract(res, a, b, s)
    ccall((:mlx_subtract, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, b, s)
end

function mlx_sum(res, a, axes, axes_num, keepdims, s)
    ccall((:mlx_sum, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, mlx_stream), res, a, axes, axes_num, keepdims, s)
end

function mlx_sum_all(res, a, keepdims, s)
    ccall((:mlx_sum_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, mlx_stream), res, a, keepdims, s)
end

function mlx_swapaxes(res, a, axis1, axis2, s)
    ccall((:mlx_swapaxes, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, a, axis1, axis2, s)
end

function mlx_take(res, a, indices, axis, s)
    ccall((:mlx_take, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, mlx_stream), res, a, indices, axis, s)
end

function mlx_take_all(res, a, indices, s)
    ccall((:mlx_take_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_stream), res, a, indices, s)
end

function mlx_take_along_axis(res, a, indices, axis, s)
    ccall((:mlx_take_along_axis, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, mlx_stream), res, a, indices, axis, s)
end

function mlx_tan(res, a, s)
    ccall((:mlx_tan, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_tanh(res, a, s)
    ccall((:mlx_tanh, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_tensordot(res, a, b, axes_a, axes_a_num, axes_b, axes_b_num, s)
    ccall((:mlx_tensordot, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t, mlx_stream), res, a, b, axes_a, axes_a_num, axes_b, axes_b_num, s)
end

function mlx_tensordot_along_axis(res, a, b, axis, s)
    ccall((:mlx_tensordot_along_axis, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Cint, mlx_stream), res, a, b, axis, s)
end

function mlx_tile(res, arr, reps, reps_num, s)
    ccall((:mlx_tile, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, arr, reps, reps_num, s)
end

function mlx_topk(res, a, k, axis, s)
    ccall((:mlx_topk, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_stream), res, a, k, axis, s)
end

function mlx_topk_all(res, a, k, s)
    ccall((:mlx_topk_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, a, k, s)
end

function mlx_trace(res, a, offset, axis1, axis2, dtype, s)
    ccall((:mlx_trace, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, Cint, mlx_dtype, mlx_stream), res, a, offset, axis1, axis2, dtype, s)
end

function mlx_transpose(res, a, axes, axes_num, s)
    ccall((:mlx_transpose, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, mlx_stream), res, a, axes, axes_num, s)
end

function mlx_transpose_all(res, a, s)
    ccall((:mlx_transpose_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_tri(res, n, m, k, type, s)
    ccall((:mlx_tri, libmlxc), Cint, (Ptr{mlx_array}, Cint, Cint, Cint, mlx_dtype, mlx_stream), res, n, m, k, type, s)
end

function mlx_tril(res, x, k, s)
    ccall((:mlx_tril, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, x, k, s)
end

function mlx_triu(res, x, k, s)
    ccall((:mlx_triu, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, x, k, s)
end

function mlx_unflatten(res, a, axis, shape, shape_num, s)
    ccall((:mlx_unflatten, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Ptr{Cint}, Csize_t, mlx_stream), res, a, axis, shape, shape_num, s)
end

function mlx_var(res, a, axes, axes_num, keepdims, ddof, s)
    ccall((:mlx_var, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, Bool, Cint, mlx_stream), res, a, axes, axes_num, keepdims, ddof, s)
end

function mlx_var_all(res, a, keepdims, ddof, s)
    ccall((:mlx_var_all, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Bool, Cint, mlx_stream), res, a, keepdims, ddof, s)
end

function mlx_view(res, a, dtype, s)
    ccall((:mlx_view, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_dtype, mlx_stream), res, a, dtype, s)
end

function mlx_where(res, condition, x, y, s)
    ccall((:mlx_where, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, mlx_array, mlx_stream), res, condition, x, y, s)
end

function mlx_zeros(res, shape, shape_num, dtype, s)
    ccall((:mlx_zeros, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cint}, Csize_t, mlx_dtype, mlx_stream), res, shape, shape_num, dtype, s)
end

function mlx_zeros_like(res, a, s)
    ccall((:mlx_zeros_like, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_stream), res, a, s)
end

function mlx_random_bernoulli(res, p, shape, shape_num, key, s)
    ccall((:mlx_random_bernoulli, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Ptr{Cint}, Csize_t, mlx_array, mlx_stream), res, p, shape, shape_num, key, s)
end

function mlx_random_bits(res, shape, shape_num, width, key, s)
    ccall((:mlx_random_bits, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cint}, Csize_t, Cint, mlx_array, mlx_stream), res, shape, shape_num, width, key, s)
end

function mlx_random_categorical_shape(res, logits, axis, shape, shape_num, key, s)
    ccall((:mlx_random_categorical_shape, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Ptr{Cint}, Csize_t, mlx_array, mlx_stream), res, logits, axis, shape, shape_num, key, s)
end

function mlx_random_categorical_num_samples(res, logits_, axis, num_samples, key, s)
    ccall((:mlx_random_categorical_num_samples, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, Cint, mlx_array, mlx_stream), res, logits_, axis, num_samples, key, s)
end

function mlx_random_categorical(res, logits, axis, key, s)
    ccall((:mlx_random_categorical, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_array, mlx_stream), res, logits, axis, key, s)
end

function mlx_random_gumbel(res, shape, shape_num, dtype, key, s)
    ccall((:mlx_random_gumbel, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cint}, Csize_t, mlx_dtype, mlx_array, mlx_stream), res, shape, shape_num, dtype, key, s)
end

function mlx_random_key(res, seed)
    ccall((:mlx_random_key, libmlxc), Cint, (Ptr{mlx_array}, UInt64), res, seed)
end

function mlx_random_laplace(res, shape, shape_num, dtype, loc, scale, key, s)
    ccall((:mlx_random_laplace, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cint}, Csize_t, mlx_dtype, Cfloat, Cfloat, mlx_array, mlx_stream), res, shape, shape_num, dtype, loc, scale, key, s)
end

function mlx_random_multivariate_normal(res, mean, cov, shape, shape_num, dtype, key, s)
    ccall((:mlx_random_multivariate_normal, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Ptr{Cint}, Csize_t, mlx_dtype, mlx_array, mlx_stream), res, mean, cov, shape, shape_num, dtype, key, s)
end

function mlx_random_normal(res, shape, shape_num, dtype, loc, scale, key, s)
    ccall((:mlx_random_normal, libmlxc), Cint, (Ptr{mlx_array}, Ptr{Cint}, Csize_t, mlx_dtype, Cfloat, Cfloat, mlx_array, mlx_stream), res, shape, shape_num, dtype, loc, scale, key, s)
end

function mlx_random_permutation(res, x, axis, key, s)
    ccall((:mlx_random_permutation, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_array, mlx_stream), res, x, axis, key, s)
end

function mlx_random_permutation_arange(res, x, key, s)
    ccall((:mlx_random_permutation_arange, libmlxc), Cint, (Ptr{mlx_array}, Cint, mlx_array, mlx_stream), res, x, key, s)
end

function mlx_random_randint(res, low, high, shape, shape_num, dtype, key, s)
    ccall((:mlx_random_randint, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Ptr{Cint}, Csize_t, mlx_dtype, mlx_array, mlx_stream), res, low, high, shape, shape_num, dtype, key, s)
end

function mlx_random_seed(seed)
    ccall((:mlx_random_seed, libmlxc), Cint, (UInt64,), seed)
end

function mlx_random_split_num(res, key, num, s)
    ccall((:mlx_random_split_num, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, Cint, mlx_stream), res, key, num, s)
end

function mlx_random_split(res_0, res_1, key, s)
    ccall((:mlx_random_split, libmlxc), Cint, (Ptr{mlx_array}, Ptr{mlx_array}, mlx_array, mlx_stream), res_0, res_1, key, s)
end

function mlx_random_truncated_normal(res, lower, upper, shape, shape_num, dtype, key, s)
    ccall((:mlx_random_truncated_normal, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Ptr{Cint}, Csize_t, mlx_dtype, mlx_array, mlx_stream), res, lower, upper, shape, shape_num, dtype, key, s)
end

function mlx_random_uniform(res, low, high, shape, shape_num, dtype, key, s)
    ccall((:mlx_random_uniform, libmlxc), Cint, (Ptr{mlx_array}, mlx_array, mlx_array, Ptr{Cint}, Csize_t, mlx_dtype, mlx_array, mlx_stream), res, low, high, shape, shape_num, dtype, key, s)
end

# no prototype is found for this function at stream.h:30:12, please use with caution
function mlx_stream_new()
    ccall((:mlx_stream_new, libmlxc), mlx_stream, ())
end

function mlx_stream_new_device(dev)
    ccall((:mlx_stream_new_device, libmlxc), mlx_stream, (mlx_device,), dev)
end

function mlx_stream_set(stream, src)
    ccall((:mlx_stream_set, libmlxc), Cint, (Ptr{mlx_stream}, mlx_stream), stream, src)
end

function mlx_stream_free(stream)
    ccall((:mlx_stream_free, libmlxc), Cint, (mlx_stream,), stream)
end

function mlx_stream_tostring(str, stream)
    ccall((:mlx_stream_tostring, libmlxc), Cint, (Ptr{mlx_string}, mlx_stream), str, stream)
end

function mlx_stream_equal(lhs, rhs)
    ccall((:mlx_stream_equal, libmlxc), Bool, (mlx_stream, mlx_stream), lhs, rhs)
end

function mlx_stream_get_device(dev, stream)
    ccall((:mlx_stream_get_device, libmlxc), Cint, (Ptr{mlx_device}, mlx_stream), dev, stream)
end

function mlx_stream_get_index(index, stream)
    ccall((:mlx_stream_get_index, libmlxc), Cint, (Ptr{Cint}, mlx_stream), index, stream)
end

function mlx_synchronize(stream)
    ccall((:mlx_synchronize, libmlxc), Cint, (mlx_stream,), stream)
end

function mlx_get_default_stream(stream, dev)
    ccall((:mlx_get_default_stream, libmlxc), Cint, (Ptr{mlx_stream}, mlx_device), stream, dev)
end

function mlx_set_default_stream(stream)
    ccall((:mlx_set_default_stream, libmlxc), Cint, (mlx_stream,), stream)
end

# no prototype is found for this function at stream.h:75:12, please use with caution
function mlx_default_cpu_stream_new()
    ccall((:mlx_default_cpu_stream_new, libmlxc), mlx_stream, ())
end

# no prototype is found for this function at stream.h:80:12, please use with caution
function mlx_default_gpu_stream_new()
    ccall((:mlx_default_gpu_stream_new, libmlxc), mlx_stream, ())
end

# no prototype is found for this function at string.h:26:12, please use with caution
function mlx_string_new()
    ccall((:mlx_string_new, libmlxc), mlx_string, ())
end

function mlx_string_new_data(str)
    ccall((:mlx_string_new_data, libmlxc), mlx_string, (Ptr{Cchar},), str)
end

function mlx_string_set(str, src)
    ccall((:mlx_string_set, libmlxc), Cint, (Ptr{mlx_string}, mlx_string), str, src)
end

function mlx_string_data(str)
    ccall((:mlx_string_data, libmlxc), Ptr{Cchar}, (mlx_string,), str)
end

function mlx_string_free(str)
    ccall((:mlx_string_free, libmlxc), Cint, (mlx_string,), str)
end

function mlx_async_eval(outputs)
    ccall((:mlx_async_eval, libmlxc), Cint, (mlx_vector_array,), outputs)
end

function mlx_checkpoint(res, fun)
    ccall((:mlx_checkpoint, libmlxc), Cint, (Ptr{mlx_closure}, mlx_closure), res, fun)
end

function mlx_custom_function(res, fun, fun_vjp, fun_jvp, fun_vmap)
    ccall((:mlx_custom_function, libmlxc), Cint, (Ptr{mlx_closure}, mlx_closure, mlx_closure_custom, mlx_closure_custom_jvp, mlx_closure_custom_vmap), res, fun, fun_vjp, fun_jvp, fun_vmap)
end

function mlx_custom_vjp(res, fun, fun_vjp)
    ccall((:mlx_custom_vjp, libmlxc), Cint, (Ptr{mlx_closure}, mlx_closure, mlx_closure_custom), res, fun, fun_vjp)
end

function mlx_eval(outputs)
    ccall((:mlx_eval, libmlxc), Cint, (mlx_vector_array,), outputs)
end

function mlx_jvp(res_0, res_1, fun, primals, tangents)
    ccall((:mlx_jvp, libmlxc), Cint, (Ptr{mlx_vector_array}, Ptr{mlx_vector_array}, mlx_closure, mlx_vector_array, mlx_vector_array), res_0, res_1, fun, primals, tangents)
end

function mlx_value_and_grad(res, fun, argnums, argnums_num)
    ccall((:mlx_value_and_grad, libmlxc), Cint, (Ptr{mlx_closure_value_and_grad}, mlx_closure, Ptr{Cint}, Csize_t), res, fun, argnums, argnums_num)
end

function mlx_vjp(res_0, res_1, fun, primals, cotangents)
    ccall((:mlx_vjp, libmlxc), Cint, (Ptr{mlx_vector_array}, Ptr{mlx_vector_array}, mlx_closure, mlx_vector_array, mlx_vector_array), res_0, res_1, fun, primals, cotangents)
end

function mlx_detail_vmap_replace(res, inputs, s_inputs, s_outputs, in_axes, in_axes_num, out_axes, out_axes_num)
    ccall((:mlx_detail_vmap_replace, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_vector_array, mlx_vector_array, mlx_vector_array, Ptr{Cint}, Csize_t, Ptr{Cint}, Csize_t), res, inputs, s_inputs, s_outputs, in_axes, in_axes_num, out_axes, out_axes_num)
end

function mlx_detail_vmap_trace(res_0, res_1, fun, inputs, in_axes, in_axes_num)
    ccall((:mlx_detail_vmap_trace, libmlxc), Cint, (Ptr{mlx_vector_array}, Ptr{mlx_vector_array}, mlx_closure, mlx_vector_array, Ptr{Cint}, Csize_t), res_0, res_1, fun, inputs, in_axes, in_axes_num)
end

# no prototype is found for this function at vector.h:28:18, please use with caution
function mlx_vector_array_new()
    ccall((:mlx_vector_array_new, libmlxc), mlx_vector_array, ())
end

function mlx_vector_array_set(vec, src)
    ccall((:mlx_vector_array_set, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_vector_array), vec, src)
end

function mlx_vector_array_free(vec)
    ccall((:mlx_vector_array_free, libmlxc), Cint, (mlx_vector_array,), vec)
end

function mlx_vector_array_new_data(data, size)
    ccall((:mlx_vector_array_new_data, libmlxc), mlx_vector_array, (Ptr{mlx_array}, Csize_t), data, size)
end

function mlx_vector_array_new_value(val)
    ccall((:mlx_vector_array_new_value, libmlxc), mlx_vector_array, (mlx_array,), val)
end

function mlx_vector_array_set_data(vec, data, size)
    ccall((:mlx_vector_array_set_data, libmlxc), Cint, (Ptr{mlx_vector_array}, Ptr{mlx_array}, Csize_t), vec, data, size)
end

function mlx_vector_array_set_value(vec, val)
    ccall((:mlx_vector_array_set_value, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_array), vec, val)
end

function mlx_vector_array_append_data(vec, data, size)
    ccall((:mlx_vector_array_append_data, libmlxc), Cint, (mlx_vector_array, Ptr{mlx_array}, Csize_t), vec, data, size)
end

function mlx_vector_array_append_value(vec, val)
    ccall((:mlx_vector_array_append_value, libmlxc), Cint, (mlx_vector_array, mlx_array), vec, val)
end

function mlx_vector_array_size(vec)
    ccall((:mlx_vector_array_size, libmlxc), Csize_t, (mlx_vector_array,), vec)
end

function mlx_vector_array_get(res, vec, idx)
    ccall((:mlx_vector_array_get, libmlxc), Cint, (Ptr{mlx_array}, mlx_vector_array, Csize_t), res, vec, idx)
end

struct mlx_vector_vector_array_
    ctx::Ptr{Cvoid}
end

const mlx_vector_vector_array = mlx_vector_vector_array_

# no prototype is found for this function at vector.h:55:25, please use with caution
function mlx_vector_vector_array_new()
    ccall((:mlx_vector_vector_array_new, libmlxc), mlx_vector_vector_array, ())
end

function mlx_vector_vector_array_set(vec, src)
    ccall((:mlx_vector_vector_array_set, libmlxc), Cint, (Ptr{mlx_vector_vector_array}, mlx_vector_vector_array), vec, src)
end

function mlx_vector_vector_array_free(vec)
    ccall((:mlx_vector_vector_array_free, libmlxc), Cint, (mlx_vector_vector_array,), vec)
end

function mlx_vector_vector_array_new_data(data, size)
    ccall((:mlx_vector_vector_array_new_data, libmlxc), mlx_vector_vector_array, (Ptr{mlx_vector_array}, Csize_t), data, size)
end

function mlx_vector_vector_array_new_value(val)
    ccall((:mlx_vector_vector_array_new_value, libmlxc), mlx_vector_vector_array, (mlx_vector_array,), val)
end

function mlx_vector_vector_array_set_data(vec, data, size)
    ccall((:mlx_vector_vector_array_set_data, libmlxc), Cint, (Ptr{mlx_vector_vector_array}, Ptr{mlx_vector_array}, Csize_t), vec, data, size)
end

function mlx_vector_vector_array_set_value(vec, val)
    ccall((:mlx_vector_vector_array_set_value, libmlxc), Cint, (Ptr{mlx_vector_vector_array}, mlx_vector_array), vec, val)
end

function mlx_vector_vector_array_append_data(vec, data, size)
    ccall((:mlx_vector_vector_array_append_data, libmlxc), Cint, (mlx_vector_vector_array, Ptr{mlx_vector_array}, Csize_t), vec, data, size)
end

function mlx_vector_vector_array_append_value(vec, val)
    ccall((:mlx_vector_vector_array_append_value, libmlxc), Cint, (mlx_vector_vector_array, mlx_vector_array), vec, val)
end

function mlx_vector_vector_array_size(vec)
    ccall((:mlx_vector_vector_array_size, libmlxc), Csize_t, (mlx_vector_vector_array,), vec)
end

function mlx_vector_vector_array_get(res, vec, idx)
    ccall((:mlx_vector_vector_array_get, libmlxc), Cint, (Ptr{mlx_vector_array}, mlx_vector_vector_array, Csize_t), res, vec, idx)
end

# no prototype is found for this function at vector.h:91:16, please use with caution
function mlx_vector_int_new()
    ccall((:mlx_vector_int_new, libmlxc), mlx_vector_int, ())
end

function mlx_vector_int_set(vec, src)
    ccall((:mlx_vector_int_set, libmlxc), Cint, (Ptr{mlx_vector_int}, mlx_vector_int), vec, src)
end

function mlx_vector_int_free(vec)
    ccall((:mlx_vector_int_free, libmlxc), Cint, (mlx_vector_int,), vec)
end

function mlx_vector_int_new_data(data, size)
    ccall((:mlx_vector_int_new_data, libmlxc), mlx_vector_int, (Ptr{Cint}, Csize_t), data, size)
end

function mlx_vector_int_new_value(val)
    ccall((:mlx_vector_int_new_value, libmlxc), mlx_vector_int, (Cint,), val)
end

function mlx_vector_int_set_data(vec, data, size)
    ccall((:mlx_vector_int_set_data, libmlxc), Cint, (Ptr{mlx_vector_int}, Ptr{Cint}, Csize_t), vec, data, size)
end

function mlx_vector_int_set_value(vec, val)
    ccall((:mlx_vector_int_set_value, libmlxc), Cint, (Ptr{mlx_vector_int}, Cint), vec, val)
end

function mlx_vector_int_append_data(vec, data, size)
    ccall((:mlx_vector_int_append_data, libmlxc), Cint, (mlx_vector_int, Ptr{Cint}, Csize_t), vec, data, size)
end

function mlx_vector_int_append_value(vec, val)
    ccall((:mlx_vector_int_append_value, libmlxc), Cint, (mlx_vector_int, Cint), vec, val)
end

function mlx_vector_int_size(vec)
    ccall((:mlx_vector_int_size, libmlxc), Csize_t, (mlx_vector_int,), vec)
end

function mlx_vector_int_get(res, vec, idx)
    ccall((:mlx_vector_int_get, libmlxc), Cint, (Ptr{Cint}, mlx_vector_int, Csize_t), res, vec, idx)
end

struct mlx_vector_string_
    ctx::Ptr{Cvoid}
end

const mlx_vector_string = mlx_vector_string_

# no prototype is found for this function at vector.h:109:19, please use with caution
function mlx_vector_string_new()
    ccall((:mlx_vector_string_new, libmlxc), mlx_vector_string, ())
end

function mlx_vector_string_set(vec, src)
    ccall((:mlx_vector_string_set, libmlxc), Cint, (Ptr{mlx_vector_string}, mlx_vector_string), vec, src)
end

function mlx_vector_string_free(vec)
    ccall((:mlx_vector_string_free, libmlxc), Cint, (mlx_vector_string,), vec)
end

function mlx_vector_string_new_data(data, size)
    ccall((:mlx_vector_string_new_data, libmlxc), mlx_vector_string, (Ptr{Ptr{Cchar}}, Csize_t), data, size)
end

function mlx_vector_string_new_value(val)
    ccall((:mlx_vector_string_new_value, libmlxc), mlx_vector_string, (Ptr{Cchar},), val)
end

function mlx_vector_string_set_data(vec, data, size)
    ccall((:mlx_vector_string_set_data, libmlxc), Cint, (Ptr{mlx_vector_string}, Ptr{Ptr{Cchar}}, Csize_t), vec, data, size)
end

function mlx_vector_string_set_value(vec, val)
    ccall((:mlx_vector_string_set_value, libmlxc), Cint, (Ptr{mlx_vector_string}, Ptr{Cchar}), vec, val)
end

function mlx_vector_string_append_data(vec, data, size)
    ccall((:mlx_vector_string_append_data, libmlxc), Cint, (mlx_vector_string, Ptr{Ptr{Cchar}}, Csize_t), vec, data, size)
end

function mlx_vector_string_append_value(vec, val)
    ccall((:mlx_vector_string_append_value, libmlxc), Cint, (mlx_vector_string, Ptr{Cchar}), vec, val)
end

function mlx_vector_string_size(vec)
    ccall((:mlx_vector_string_size, libmlxc), Csize_t, (mlx_vector_string,), vec)
end

function mlx_vector_string_get(res, vec, idx)
    ccall((:mlx_vector_string_get, libmlxc), Cint, (Ptr{Ptr{Cchar}}, mlx_vector_string, Csize_t), res, vec, idx)
end

# exports
const PREFIXES = ["mlx"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
