#pragma once

#include <cstdio>
#include <vector>

#include "nttconf.cuh"
#include "ulvt/utils/common.cuh"

// inplace additive ntt butterfly
template <typename T, typename P>
static constexpr __device__ __host__ void antt_butterfly(T &u, T &v, T w)
{
	u = P::add(u, P::multiply(w, v));
	v = P::add(u, v);
}

template <typename T, typename P>
static constexpr __device__ __host__ T subspace_map(const T element, const T constant)
{
	return P::add(P::square(element), P::multiply(constant, element));
}

static constexpr __device__ int get_v_offset(const int uidx, const int stage) { return uidx | (1 << stage); }

static constexpr __device__ int get_u_offset(const int stage, const int butterfly_block, const int butterfly_idx)
{
	return butterfly_block << (stage + 1) | butterfly_idx;
}

// the j variable from the model
static constexpr __device__ int get_butterfly_block(const int thread_id, const int stage)
{
	return thread_id / (1 << stage);
}

// the k variable from the model
static constexpr __device__ int get_butterfly(const int thread_id, const int stage) { return thread_id % (1 << stage); }

template <typename T>
static constexpr __device__ __host__ T *_flat_array_2d(
	T *_2d_data, const size_t width_bytes, const int row, const int col)
{
	return ((T *)((char *)_2d_data + row * width_bytes) + col);
}

// For convenience this is operating on sizeof(T) widths as opposed to cuda's byte widths
template <typename T>
static constexpr __device__ __host__ T &flat_array_3d(
	T *_3d_data, const int width, const int height, const int x, const int y, const int z)
{
	return _3d_data[x + width * (y + z * height)];
}

template <typename T>
static constexpr __device__ __host__ T &flat_array_2d(
	T *_2d_data, const size_t width_bytes, const int row, const int col)
{
	return *_flat_array_2d(_2d_data, width_bytes, row, col);
}

static constexpr __device__ __host__ bool is_bit_set(const int x, const int i) { return (x >> i) & 1; }

template <typename T, typename P>
static constexpr __device__ __host__ T calculate_twiddle(
	const T *constants,
	const size_t c_pitch,
	const int log_h,
	const int log_rate,
	const int coset,
	const int stage,
	const int butterfly_block)
{
	T sum = P::ZERO();
	for (int k = 0; k < log_h + log_rate - 1 - stage; k++)
	{
		const int coset_shift = (stage < log_h) ? (log_h - 1 - stage) : (stage - log_h + 1);
		int indicator = (coset_shift > 0 ? (coset << coset_shift) : coset) | butterfly_block;
		if (is_bit_set(indicator, k))
		{
			sum = P::add(sum, flat_array_2d(constants, c_pitch, stage, k));
		}
	}
	return sum;
}

template <typename T>
struct AdditiveNTTKernelParams
{
	T *data_io;
	size_t data_pitch;
	T *constants;
	size_t constants_pitch;
	int log_h;
	int log_rate;
	int start_stage;
	int end_stage;
};

#define BLOCK_SIZE_NEW 512
#define STAGE_IN_BLOCK 9
#define CLUSTER_SIZE_CONST 8

template <typename T, typename P>
void cpu_antt(AdditiveNTTKernelParams<T> kernel_params)
{
	// CPU implementation of the additive NTT
	T *data_io = kernel_params.data_io;
	T *constants = kernel_params.constants;
	size_t constants_pitch = kernel_params.constants_pitch;
	int log_h = kernel_params.log_h;
	int log_rate = kernel_params.log_rate;

	for (int stage = log_h - 1; stage >= 0; stage--)
	{
		size_t curr_shift = 1 << stage;
		for (size_t idx = 0; idx < (1 << log_h); idx++)
		{
			size_t partner_idx = idx ^ curr_shift;
			size_t butterfly_block = idx / (1 << (stage + 1));
			if (partner_idx > idx)
			{
				T u = data_io[idx];
				T v = data_io[partner_idx];
				T twiddle = calculate_twiddle<T, P>(constants, constants_pitch, log_h, log_rate, 0, stage, butterfly_block);
				antt_butterfly<T, P>(u, v, twiddle);
				data_io[idx] = u;
				data_io[partner_idx] = v;
			}
		}
	}
}

#include <cuda/barrier>
#include <cuda/ptx>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;
using barrier = cuda::barrier<cuda::thread_scope_block>;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

template <typename T, typename P>
static __global__ void coop_antt(AdditiveNTTKernelParams<T> kernel_params, __const__ T *pre_computed)
{
	cg::grid_group grid = cg::this_grid();

	T *data_io = kernel_params.data_io;
	size_t data_pitch = kernel_params.data_pitch;

	int log_h = kernel_params.log_h;
	int log_rate = kernel_params.log_rate;
	const int input_size = 1 << log_h;
	const int num_cosets = 1 << log_rate;
	const size_t total_elements = (size_t)input_size * (size_t)num_cosets;
	const int start_stage = kernel_params.start_stage;
	const int end_stage = kernel_params.end_stage;

	const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;

	__shared__ alignas(16) T smem_u[2][BLOCK_SIZE_NEW];
	__shared__ alignas(16) T smem_v[2][BLOCK_SIZE_NEW];

#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ barrier bar[2];

	if (threadIdx.x == 0)
	{
		init(&bar[0], blockDim.x);
		init(&bar[1], blockDim.x);
	}
	__syncthreads();

	for (int stage = start_stage; stage >= end_stage; stage--)
	{
		int pipe_idx = 0;
		int iter = 0;
		size_t curr_pos = blockIdx.x * blockDim.x;

		while (curr_pos < total_elements)
		{
			if ((curr_pos ^ (1u << stage)) > curr_pos)
				break;
			iter++;
			curr_pos += stride;
		}

		if (curr_pos >= total_elements)
		{
			grid.sync();
			continue;
		}

		int curr_coset = curr_pos / input_size;
		int curr_col = curr_pos % input_size;
		T *u_ptr = (T *)((char *)data_io + curr_coset * data_pitch) + curr_col;

		size_t curr_partner = curr_pos ^ (1u << stage);
		int partner_coset = curr_partner / input_size;
		int partner_col = curr_partner % input_size;
		T *v_ptr = (T *)((char *)data_io + partner_coset * data_pitch) + partner_col;

		if (threadIdx.x == 0)
		{
			cuda::memcpy_async(smem_u[pipe_idx], u_ptr, cuda::aligned_size_t<16>(sizeof(T) * blockDim.x), bar[pipe_idx]);
			cuda::memcpy_async(smem_v[pipe_idx], v_ptr, cuda::aligned_size_t<16>(sizeof(T) * blockDim.x), bar[pipe_idx]);
		}

		while (curr_pos < total_elements)
		{

			bar[pipe_idx].wait(bar[pipe_idx].arrive());

			const size_t thread_curr_pos = curr_pos + threadIdx.x;
			const int thread_coset = thread_curr_pos / input_size;
			const int thread_col = thread_curr_pos % input_size;
			const size_t butterfly_block_global = thread_col / (1 << (stage + 1));

			T twiddle = calculate_twiddle<T, P>(pre_computed, kernel_params.constants_pitch, log_h, log_rate, thread_coset, stage, butterfly_block_global);

			T u = smem_u[pipe_idx][threadIdx.x];
			T v = smem_v[pipe_idx][threadIdx.x];
			antt_butterfly<T, P>(u, v, twiddle);
			smem_u[pipe_idx][threadIdx.x] = u;
			smem_v[pipe_idx][threadIdx.x] = v;

			ptx::fence_proxy_async(ptx::space_shared);
			__syncthreads();

			if (threadIdx.x == 0)
			{
				ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, u_ptr, smem_u[pipe_idx], sizeof(T) * blockDim.x);
				ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, v_ptr, smem_v[pipe_idx], sizeof(T) * blockDim.x);
				ptx::cp_async_bulk_commit_group();
			}

			iter++;
			size_t next_pos = blockIdx.x * blockDim.x + iter * stride;
			while (next_pos < total_elements)
			{
				if ((next_pos ^ (1u << stage)) > next_pos)
					break;
				iter++;
				next_pos += stride;
			}

			int next_pipe = 1 - pipe_idx;
			T *next_u_ptr = nullptr;
			T *next_v_ptr = nullptr;

			if (threadIdx.x == 0)
			{
				ptx::cp_async_bulk_wait_group_read(ptx::n32_t<1>());
			}
			__syncthreads();

			if (next_pos < total_elements)
			{
				int next_coset = next_pos / input_size;
				int next_col = next_pos % input_size;
				next_u_ptr = (T *)((char *)data_io + next_coset * data_pitch) + next_col;

				size_t next_partner = next_pos ^ (1u << stage);
				int n_p_coset = next_partner / input_size;
				int n_p_col = next_partner % input_size;
				next_v_ptr = (T *)((char *)data_io + n_p_coset * data_pitch) + n_p_col;

				if (threadIdx.x == 0)
				{
					cuda::memcpy_async(smem_u[next_pipe], next_u_ptr, cuda::aligned_size_t<16>(sizeof(T) * blockDim.x), bar[next_pipe]);
					cuda::memcpy_async(smem_v[next_pipe], next_v_ptr, cuda::aligned_size_t<16>(sizeof(T) * blockDim.x), bar[next_pipe]);
				}
			}

			curr_pos = next_pos;
			u_ptr = next_u_ptr;
			v_ptr = next_v_ptr;
			pipe_idx = next_pipe;
		}

		if (threadIdx.x == 0)
		{
			ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
		}
		__syncthreads();

		ptx::fence_proxy_async(ptx::space_global);

		grid.sync();
	}
}

#else

template <typename T, typename P>
static __global__ void coop_antt(AdditiveNTTKernelParams<T> kernel_params, __const__ T *pre_computed)
{
	cg::grid_group grid = cg::this_grid();

	T *data_io = kernel_params.data_io;
	size_t data_pitch = kernel_params.data_pitch;

	int log_h = kernel_params.log_h;
	int log_rate = kernel_params.log_rate;
	const int input_size = 1 << log_h;

	const int num_cosets = 1 << log_rate;
	const size_t total_elements = (size_t)input_size * (size_t)num_cosets;
	const int start_stage = kernel_params.start_stage;
	const int end_stage = kernel_params.end_stage;

	const size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;

	for (int stage = start_stage; stage >= end_stage; stage--)
	{
		for (size_t pos = global_tid; pos < total_elements; pos += stride)
		{
			size_t partner_pos = pos ^ (1u << stage);

			if (partner_pos > pos)
			{
				const int coset = pos / input_size;
				const int col = pos % input_size;
				const int partner_coset = partner_pos / input_size;
				const int partner_col = partner_pos % input_size;

				size_t butterfly_block_global = col / (1 << (stage + 1));
				T twiddle = calculate_twiddle<T, P>(pre_computed, kernel_params.constants_pitch, log_h, log_rate, coset, stage, butterfly_block_global);

				T u = flat_array_2d<T>(data_io, data_pitch, coset, col);
				T v = flat_array_2d<T>(data_io, data_pitch, partner_coset, partner_col);

				antt_butterfly<T, P>(u, v, twiddle);

				flat_array_2d<T>(data_io, data_pitch, coset, col) = u;
				flat_array_2d<T>(data_io, data_pitch, partner_coset, partner_col) = v;
			}
		}
		grid.sync();
	}
}
#endif

template <typename T, typename P>
static __global__ void antt_inwarp(AdditiveNTTKernelParams<T> kernel_params, __const__ T *pre_computed)
{
	T *data_io = kernel_params.data_io;
	int log_h = kernel_params.log_h;
	int log_rate = kernel_params.log_rate;
	const int input_size = 1 << log_h;
	const int num_cosets = 1 << log_rate;
	const size_t total_elements = (size_t)input_size * (size_t)num_cosets;
	const int start_stage = kernel_params.start_stage;
	const int end_stage = kernel_params.end_stage;

	const size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int local_idx = threadIdx.x;

	extern __shared__ T shared_data[];

	if (global_idx < total_elements)
	{
		const int coset = global_idx / input_size;
		const int col = global_idx % input_size;
		shared_data[local_idx] = flat_array_2d<T>(data_io, kernel_params.data_pitch, coset, col);
	}
	__syncthreads();

	for (int stage = start_stage; stage >= end_stage; stage--)
	{
		if (stage >= log_h)
			continue;

		if (global_idx < total_elements)
		{
			const size_t partner_local_idx = local_idx ^ (1u << stage);
			const size_t partner_global_idx = blockIdx.x * blockDim.x + partner_local_idx;

			if (partner_local_idx > (size_t)local_idx && partner_global_idx < total_elements)
			{
				const int coset = global_idx / input_size;
				const int col = global_idx % input_size;
				const size_t butterfly_block_global = col / (1 << (stage + 1));
				T twiddle = calculate_twiddle<T, P>(pre_computed, kernel_params.constants_pitch, log_h, log_rate, coset, stage, butterfly_block_global);

				T u = shared_data[local_idx];
				T v = shared_data[partner_local_idx];

				antt_butterfly<T, P>(u, v, twiddle);

				shared_data[local_idx] = u;
				shared_data[partner_local_idx] = v;
			}
		}
		__syncthreads();
	}

	if (global_idx < total_elements)
	{
		const int coset = global_idx / input_size;
		const int col = global_idx % input_size;
		flat_array_2d<T>(data_io, kernel_params.data_pitch, coset, col) = shared_data[local_idx];
	}
}

__device__ inline bool is_elected()
{
	unsigned int tid = threadIdx.x;
	unsigned int warp_id = tid / 32;
	unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
	unsigned int lane_id = tid & 31u;
	return (uniform_warp_id == 0 && lane_id == 0);
}

#define CLUSTER_SIZE 8
template <typename T, typename P>
static __global__ void antt_dsmem(AdditiveNTTKernelParams<T> kernel_params, __const__ T *pre_computed)
{
	extern __shared__ T shared_data[];
	#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	cg::cluster_group cluster = cg::this_cluster();
	const int cluster_size = cluster.dim_blocks().x;
	const int local_rank = blockIdx.x % cluster_size;
	const int cluster_id = blockIdx.x / cluster_size;
	const size_t block_base = (size_t)cluster_id * (size_t)cluster_size * (size_t)blockDim.x + (size_t)local_rank * (size_t)blockDim.x;
	const size_t global_idx = block_base + (size_t)threadIdx.x;
	const int input_size = 1 << kernel_params.log_h;
	const int log_h = kernel_params.log_h;
	const int log_rate = kernel_params.log_rate;
	const size_t total_elements = (size_t)input_size * (size_t)(1 << log_rate);
	const int start_stage = kernel_params.start_stage;
	const int end_stage = kernel_params.end_stage;
	const bool active = global_idx < total_elements;

	T *curr_u = (T *)shared_data;
	T *next_u = (T *)(shared_data + blockDim.x);

	if (active)
	{
		const int coset = global_idx / input_size;
		const int col = global_idx % input_size;
		next_u[threadIdx.x] = flat_array_2d<T>(kernel_params.data_io, kernel_params.data_pitch, coset, col);
	}

	int loop_bound = ((total_elements + blockDim.x * cluster_size - 1) / (blockDim.x * cluster_size)) * (blockDim.x * cluster_size);

	for (int idx = global_idx; idx < loop_bound; idx += gridDim.x * blockDim.x)
	{

		cluster.sync();

		T *temp = curr_u;
		curr_u = next_u;
		next_u = temp;

		const bool is_active = idx < total_elements;

		for (int stage = start_stage; stage >= end_stage; stage--)
		{
			const size_t partner_global_idx = idx ^ (1u << stage);

			if (is_active && partner_global_idx > idx && partner_global_idx < total_elements)
			{
				size_t partner_block = partner_global_idx / blockDim.x;
				size_t block_id_in_cluster = partner_block % cluster_size;
				size_t partner_thread = partner_global_idx % blockDim.x;
				T v;
				T *neighbor_smem = nullptr;

				if (block_id_in_cluster == local_rank)
				{
					v = curr_u[partner_thread];
				}
				else
				{
					neighbor_smem = cluster.map_shared_rank(curr_u, block_id_in_cluster);
					v = neighbor_smem[partner_thread];
				}

				const int coset = idx / input_size;
				const int col = idx % input_size;
				const size_t butterfly_block_global = col / (1 << (stage + 1));
				T twiddle = calculate_twiddle<T, P>(pre_computed, kernel_params.constants_pitch, log_h, log_rate, coset, stage, butterfly_block_global);

				T u = curr_u[threadIdx.x];
				antt_butterfly<T, P>(u, v, twiddle);
				curr_u[threadIdx.x] = u;

				if (block_id_in_cluster == local_rank)
				{
					curr_u[partner_thread] = v;
				}
				else
				{
					neighbor_smem[partner_thread] = v;
				}
			}
			cluster.sync();
		}

		if (idx + gridDim.x * blockDim.x < total_elements)
		{
			size_t n_idx = idx + gridDim.x * blockDim.x;
			const int coset = n_idx / input_size;
			const int col = n_idx % input_size;
			next_u[threadIdx.x] = flat_array_2d<T>(kernel_params.data_io, kernel_params.data_pitch, coset, col);
		}

		if (is_active)
		{
			const int coset = idx / input_size;
			const int col = idx % input_size;
			flat_array_2d<T>(kernel_params.data_io, kernel_params.data_pitch, coset, col) = curr_u[threadIdx.x];
		}
	}
	#endif
}

#define ELEMS_PER_BLOCK 8192

// playground kernel for ANTT
template <typename T, typename P>
__global__ void tma_dsmem_kernel(AdditiveNTTKernelParams<int> kernel_params, __const__ int *pre_computed)
{

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	// dynamic shared memory
	extern __shared__ int smem[]; // ELEMS_PER_BLOCK per block expected

	cg::cluster_group cluster = cg::this_cluster();
	// block-local barrier for TMA
	__shared__ barrier read_global_bar, write_global_bar, read_neighbor_bar;
	if (threadIdx.x == 0)
	{
		init(&read_global_bar, blockDim.x);
		init(&write_global_bar, blockDim.x);
		init(&read_neighbor_bar, blockDim.x);
	}
	__syncthreads();

	__shared__ T *neighbor_smem[CLUSTER_SIZE_CONST];

	if (is_elected())
	{
		for (int neighbor_id = 0; neighbor_id < cluster.dim_blocks().x; neighbor_id++)
		{
			neighbor_smem[neighbor_id] = cluster.map_shared_rank(smem, neighbor_id);
		}
	}
	__syncthreads();
	// compute pointers
	size_t block_elems = ELEMS_PER_BLOCK;
	size_t bytes = block_elems * sizeof(T);
	size_t input_size = 1 << kernel_params.log_h;
	size_t total_elements = (size_t)input_size * ((size_t)1 << kernel_params.log_rate);
	size_t cluster_elems = block_elems * cluster.dim_blocks().x;

	int offset = blockIdx.x * block_elems;
	int per_thread = block_elems / blockDim.x;
	int inter_block_stride = blockDim.x * per_thread;
	size_t idx = blockIdx.x * blockDim.x * per_thread + threadIdx.x;

	int *global_src = _flat_array_2d<T>(kernel_params.data_io, kernel_params.data_pitch, offset / input_size, offset % input_size);
	// 1) Elected thread issues async memcpy to copy global -> shared//kernel_params.data_io + offset;
	if (is_elected())
	{
		// use the high-level API which accepts the block barrier directly
		cuda::memcpy_async(
			smem, global_src,
			cuda::aligned_size_t<16>(bytes),
			read_global_bar);
	}

	int start_stage = kernel_params.start_stage;
	int end_stage = kernel_params.end_stage;

	size_t block_level_stride = gridDim.x * blockDim.x * per_thread;

	for (int outer_idx = idx; outer_idx < total_elements; outer_idx += block_level_stride)
	{
		read_global_bar.wait(std::move(read_global_bar.arrive()));

		int local_idx = threadIdx.x;
		for (int stage = start_stage; stage >= end_stage; stage--)
		{

			for (int id = local_idx; id < block_elems && id < total_elements; id += blockDim.x)
			{
				size_t partner_idx = id ^ (1u << start_stage);
				if (partner_idx > id && partner_idx < offset + block_elems && partner_idx < total_elements)
				{
					size_t partner_block = partner_idx / block_elems;
					size_t partner_cluster = partner_block / (cluster.dim_blocks().x * block_elems);
					size_t id_in_cluster = partner_block % (cluster.dim_blocks().x * block_elems);
					int butterfly_block_global = (id % input_size) / (1 << (stage + 1));
					T twiddle = calculate_twiddle<T, P>(pre_computed, kernel_params.constants_pitch, kernel_params.log_h, kernel_params.log_rate, id / input_size, stage, butterfly_block_global);
					T u = neighbor_smem[cluster.cluster_rank()][id];
					T v = neighbor_smem[partner_cluster][id_in_cluster];
					antt_butterfly<T, P>(u, v, twiddle);
					neighbor_smem[cluster.cluster_rank()][id] = u;
					neighbor_smem[partner_cluster][id_in_cluster] = v;
				}
			}
			ptx::fence_proxy_async(ptx::space_shared);
			__syncthreads();

		}
		if (is_elected())
		{
			ptx::cp_async_bulk(
				ptx::space_global, ptx::space_shared,
				global_src, smem,
				bytes);
			// commit and wait for read of shared by TMA engine
			ptx::cp_async_bulk_commit_group();
			ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
		}
	}
#endif
}

template <typename T, typename P>
static __global__ void additive_ntt_kernel(AdditiveNTTKernelParams<T> kernel_params)
{
	__shared__ char shared_mem[MAX_SHARED_MEM]; // 32KB
	T *data_io = kernel_params.data_io;
	size_t d_pitch = kernel_params.data_pitch;
	const T *pre_computed = kernel_params.constants;
	size_t c_pitch = kernel_params.constants_pitch;
	int log_h = kernel_params.log_h;
	int log_rate = kernel_params.log_rate;
	int start_stage = kernel_params.start_stage;
	int end_stage = kernel_params.end_stage;

	T *uv_mem = (T *)shared_mem; // shared memory for UV

	const int local_id = threadIdx.x;
	const int coset = threadIdx.z; // coset in z dimension
	const int max_stages_per_kernel = MAX_STAGES_PER_KERNEL - log_rate;

	int unit_vec[3] = {0};
	unit_vec[start_stage / max_stages_per_kernel] = 1;

	const int exec_id_1 = local_id + blockDim.x * blockIdx.x;

	const int exec_id_2 = threadIdx.x * gridDim.y * blockDim.y + blockIdx.y +
						  gridDim.y * blockDim.y * blockDim.x * blockIdx.z + gridDim.y * threadIdx.y;

	const int exec_id_3 = threadIdx.x * gridDim.z * gridDim.y * blockDim.y + blockIdx.z + gridDim.z * blockIdx.y +
						  gridDim.z * gridDim.y * threadIdx.y;

	const int exec_id = unit_vec[0] * exec_id_1 + unit_vec[1] * exec_id_2 + unit_vec[2] * exec_id_3;

	const int local_off = blockDim.x;

	const int uv_width = blockDim.x * 2;
	const int uv_height = blockDim.y;

	const int butterfly_block = get_butterfly_block(exec_id, start_stage);
	const int butterfly_idx = get_butterfly(exec_id, start_stage);
	const int uoff = get_u_offset(start_stage, butterfly_block, butterfly_idx);
	const int voff = get_v_offset(uoff, start_stage);

	// copy from global memory into shared memory here
	// data_io -> uv_mem
	flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x, threadIdx.y, coset) = // data, width, height, x, y, z
		flat_array_2d<T>(data_io, d_pitch, coset, uoff);							 // _3d_data[x + width * (y + z * height)] = _2d_data[row * width_bytes + col]   //data, width_bytes, row, col
	flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x + local_off, threadIdx.y, coset) =
		flat_array_2d<T>(data_io, d_pitch, coset, voff);

	for (int stage = end_stage - 1; stage >= start_stage; stage--)
	{
		// calculate twiddle, this stage has to be from the global context
		int butterfly_block_global = get_butterfly_block(exec_id, stage);
		T twiddle =
			calculate_twiddle<T, P>(pre_computed, c_pitch, log_h, log_rate, coset, stage, butterfly_block_global);

		// These stages have to be from the local context
		int butterfly_block = get_butterfly_block(local_id, stage - start_stage);
		int butterfly_idx = get_butterfly(local_id, stage - start_stage);
		int uoff_local = get_u_offset(stage - start_stage, butterfly_block, butterfly_idx);
		int voff_local = get_v_offset(uoff_local, stage - start_stage);
		T &u = flat_array_3d<T>(uv_mem, uv_width, uv_height, uoff_local, threadIdx.y, coset);
		T &v = flat_array_3d<T>(uv_mem, uv_width, uv_height, voff_local, threadIdx.y, coset);
		antt_butterfly<T, P>(u, v, twiddle);

		__syncthreads();
	}

	flat_array_2d<T>(data_io, d_pitch, coset, uoff) =
		flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x, threadIdx.y, coset);
	flat_array_2d<T>(data_io, d_pitch, coset, voff) =
		flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x + local_off, threadIdx.y, coset);
}

static constexpr void print_kern_launch(dim3 dim_grids, dim3 dim_blocks, int kern)
{
	printf(
		"Kernel %d launch configuration blocks: (%d, %d, %d) grids: (%d, %d, %d)\n",
		kern,
		dim_blocks.x,
		dim_blocks.y,
		dim_blocks.z,
		dim_grids.x,
		dim_grids.y,
		dim_grids.z);
}

template <typename T, typename P>
class AdditiveNTT
{
public:
	AdditiveNTT(const AdditiveNTTConf<T, P> &nttconf) : ntt_conf(nttconf)
	{
		printf("Initializing AdditiveNTT with log_h = %d, log_rate = %d\n", ntt_conf.log_h, ntt_conf.log_rate);
		const int input_size = 1 << ntt_conf.log_h;
		const int output_size = 1 << (ntt_conf.log_h + ntt_conf.log_rate);

		auto s_evals = precompute_subspace_evals();
		auto largest_width = ntt_conf.log_h + ntt_conf.log_rate - 1;
		CUDA_CHECK(cudaMallocPitch(&pre_computed, &constants_pitch, sizeof(T) * largest_width, ntt_conf.log_h));

		CUDA_CHECK(cudaMemcpy2D(
			pre_computed,
			constants_pitch,
			s_evals,
			largest_width * sizeof(T),
			largest_width * sizeof(T),
			ntt_conf.log_h,
			cudaMemcpyHostToDevice));

		delete[] s_evals;

		CUDA_CHECK(cudaMallocPitch(&data_in_out, &out_pitch, sizeof(T) * input_size, 1 << ntt_conf.log_rate));
	}

	bool apply(const NTTData<T> &input, NTTData<T> &output)
	{
		auto log_h = ntt_conf.log_h;
		auto log_rate = ntt_conf.log_rate;
		size_t input_size = 1 << log_h;
		size_t output_size = 1 << (log_h + log_rate);
		if (input.size != input_size || input.order != DataOrder::IN_ORDER)
		{
			return false;
		}

		char *data_io = (char *)data_in_out;
		// copy data into output buffer first, which will operated on by the kernel
		// This copies the address of input 2^log_rate times into our temporary buffer
		for (size_t i = 0; i < (1 << log_rate); i++)
		{
			CUDA_CHECK(cudaMemcpy(&data_io[i * out_pitch], input.data.get(), input.byte_len(), cudaMemcpyHostToDevice));
		}

		AdditiveNTTKernelParams<T> kernel_params;
		kernel_params.data_io = data_in_out;
		kernel_params.data_pitch = out_pitch;
		kernel_params.constants = pre_computed;
		kernel_params.constants_pitch = constants_pitch;
		kernel_params.log_h = log_h;
		kernel_params.log_rate = log_rate;
		kernel_params.start_stage = 0;
		kernel_params.end_stage = 0;

		auto [kernel_launch_conf, num_kerns] = ntt_conf.get_kernel_launch_confs();

		cudaEvent_t start, stop;
		CUDA_CHECK(cudaEventCreate(&start));
		CUDA_CHECK(cudaEventCreate(&stop));
		CUDA_CHECK(cudaEventRecord(start));
		CUDA_CHECK(cudaEventSynchronize(start));

		// 		for (int kern = num_kerns - 1; kern >= 0; kern--)
		// 		{
		// 			// int
		// 			kernel_params.start_stage = kern * max_stages_per_kernel;
		// 			kernel_params.end_stage = std::min(stages, max_stages_per_kernel * (kern + 1));
		// #ifndef NDEBUG
		// 			print_kern_launch(kernel_launch_conf[kern][1], kernel_launch_conf[kern][0], kern);
		// 			printf(
		// 				"Kernel %d (..., start_stage = %d, end_stage = %d)\n",
		// 				kern,
		// 				kernel_params.start_stage,
		// 				kernel_params.end_stage);
		// #endif // DEBUG
		// 			additive_ntt_kernel<T, P><<<kernel_launch_conf[kern][1], kernel_launch_conf[kern][0]>>>(kernel_params);
		// 		}

		cudaDeviceProp prop;
		CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

		int sm_core_count = prop.multiProcessorCount;

		const int top_stage = log_h - 1;

		int dsmem_start_stage = std::min(top_stage, 17);

		int inwarp_start_stage = std::min(top_stage, 4);

		if (top_stage > inwarp_start_stage)
		{
			int blocks = sm_core_count * 3;
			if (blocks == 0)
				blocks = 1;

			int max_active_blocks_per_sm = 0;
			CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
				&max_active_blocks_per_sm, coop_antt<T, P>, BLOCK_SIZE_NEW, 0));
			if (max_active_blocks_per_sm > 0)
			{
				const int max_blocks = max_active_blocks_per_sm * sm_core_count;
				if (blocks > max_blocks)
					blocks = max_blocks;
			}

			kernel_params.start_stage = top_stage;
			kernel_params.end_stage = inwarp_start_stage + 1; // Hand-off point

			void *args[] = {&kernel_params, &pre_computed};
			CUDA_CHECK(cudaLaunchCooperativeKernel((void *)coop_antt<T, P>,
												   dim3(blocks, 1, 1),
												   dim3(BLOCK_SIZE_NEW, 1, 1),
												   args));
		}
		if(top_stage > dsmem_start_stage)
		{
			int blocks = sm_core_count * 4;
			if (blocks == 0)
				blocks = 1;

			int max_active_blocks_per_sm = 0;
			CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
				&max_active_blocks_per_sm, antt_dsmem<T, P>, BLOCK_SIZE_NEW, 0));
			
			if (max_active_blocks_per_sm > 0)
			{
				const int max_blocks = max_active_blocks_per_sm * sm_core_count;
				if (blocks > max_blocks)
					blocks = max_blocks;
			}

			kernel_params.start_stage = dsmem_start_stage;
			kernel_params.end_stage = inwarp_start_stage + 1; // Hand-off point

			blocks = (blocks + CLUSTER_SIZE_CONST - 1) / CLUSTER_SIZE_CONST * CLUSTER_SIZE_CONST; 
			cudaLaunchConfig_t config = {};
			config.gridDim = dim3(blocks, 1, 1);
			config.blockDim = dim3(BLOCK_SIZE_NEW, 1, 1);
			config.dynamicSmemBytes = ELEMS_PER_BLOCK * sizeof(T) * 2;

			cudaLaunchAttribute attrs[2];
			attrs[0].id = cudaLaunchAttributeClusterDimension;
			attrs[0].val.clusterDim.x = CLUSTER_SIZE_CONST;
			attrs[0].val.clusterDim.y = 1;
			attrs[0].val.clusterDim.z = 1;

			attrs[1].id = cudaLaunchAttributeCooperative;
			attrs[1].val.cooperative = true;

			config.attrs = attrs;
			config.numAttrs = 2;

			void *args[] = {&kernel_params, &pre_computed};
			CUDA_CHECK(cudaLaunchKernelEx(&config, tma_dsmem_kernel<T, P>, args));
		}
		if (inwarp_start_stage >= 0)
		{
			kernel_params.start_stage = inwarp_start_stage;
			kernel_params.end_stage = 0;

			int blocks_inwarp = (output_size + BLOCK_SIZE_NEW - 1) / BLOCK_SIZE_NEW;
			antt_inwarp<T, P><<<dim3(blocks_inwarp, 1, 1), dim3(BLOCK_SIZE_NEW, 1, 1), BLOCK_SIZE_NEW * sizeof(T)>>>(kernel_params, pre_computed);
		}

		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		float milliseconds = 0;
		CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
		printf("Log_h: %d, log_rate: %d, Total kernel execution time: %f ms\n", ntt_conf.log_h, ntt_conf.log_rate, milliseconds);
		T *constants_cpu = new T[ntt_conf.log_h * (ntt_conf.log_h + ntt_conf.log_rate - 1)];
		CUDA_CHECK(cudaMemcpy2D(
			constants_cpu,
			sizeof(T) * (ntt_conf.log_h + ntt_conf.log_rate - 1),
			pre_computed,
			constants_pitch,
			sizeof(T) * (ntt_conf.log_h + ntt_conf.log_rate - 1),
			ntt_conf.log_h,
			cudaMemcpyDeviceToHost));

		// AdditiveNTTKernelParams<T> kernel_params_cpu{};
		// kernel_params_cpu.data_io = input.data.get();
		// kernel_params_cpu.data_pitch = kernel_params.data_pitch;
		// kernel_params_cpu.constants = constants_cpu;
		// kernel_params_cpu.constants_pitch = sizeof(T) * (ntt_conf.log_h + ntt_conf.log_rate - 1);
		// kernel_params_cpu.log_h = ntt_conf.log_h;
		// kernel_params_cpu.log_rate = ntt_conf.log_rate;

		// cpu_antt<T, P>(kernel_params_cpu);

		// dim3 block = BLOCK_SIZE_NEW;
		// dim3 grid((output_size + block.x - 1) / block.x);
		// additive_ntt_kernel_new<T, P><<<grid, block>>>(kernel_params, pre_computed);

		// At the end copy back the data from the output into a single contiguous memory
		output.order = DataOrder::IN_ORDER;
		CUDA_CHECK(cudaMemcpy2D(
			output.data.get(),
			input.byte_len(),
			data_in_out,
			out_pitch,
			input.byte_len(),
			1 << log_rate,
			cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaDeviceSynchronize());

		return true;
	}

	~AdditiveNTT()
	{
		CUDA_CHECK(cudaFree(pre_computed));
		CUDA_CHECK(cudaFree(data_in_out));
	}

private:
	inline T *precompute_subspace_evals() const
	{
		auto largest_width = ntt_conf.log_h + ntt_conf.log_rate - 1;
		auto pitch = largest_width * sizeof(T);
		T *constants = new T[ntt_conf.log_h * largest_width];

		std::vector<T> norm_consts;
		norm_consts.reserve(ntt_conf.log_h);

		for (int i = 1; i < ntt_conf.log_rate + ntt_conf.log_h; i++)
		{
			flat_array_2d(constants, pitch, 0, i - 1) = T(1 << i);
		}
		norm_consts.push_back(P::ONE());

		for (int i = 1; i < ntt_conf.log_h; i++)
		{
			T norm_prev = norm_consts.back();
			T *s_evals_prev = _flat_array_2d(constants, pitch, i - 1, 0);

			T norm_const_i = subspace_map<T, P>(s_evals_prev[0], norm_prev); // s_evals_prev^2 + norm_prev * s_evals_prev

			for (size_t j = 1; j < ntt_conf.log_h + ntt_conf.log_rate - i; j++)
			{
				T sij_prev = s_evals_prev[j];
				flat_array_2d(constants, pitch, i, j - 1) = subspace_map<T, P>(sij_prev, norm_prev);
			}

			norm_consts.push_back(norm_const_i);
		}

		for (size_t i = 0; i < ntt_conf.log_h; i++)
		{
			T inv_norm_const = P::inverse(norm_consts[i]);
			T *si_evals = _flat_array_2d(constants, pitch, i, 0);
			for (size_t j = 0; j < ntt_conf.log_h + ntt_conf.log_rate - i - 1; j++)
			{
				si_evals[j] = P::multiply(inv_norm_const, si_evals[j]);
			}
		}

		return constants;
	}

	size_t constants_pitch;
	size_t out_pitch;
	AdditiveNTTConf<T, P> ntt_conf;
	// host memory pointers

	// gpu memory pointers
	T *pre_computed;
	T *data_in_out;
};
