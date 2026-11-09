//! ==============================================================================
//! LAYER 2: SHELFDB GPU BACKEND - PHYSICAL VRAM & CUDA GRAPH INTERFACE
//! Silicon Target: NVIDIA Tesla T4 (sm_75) / Compute Capability >= 7.5
//! ==============================================================================

use std::ffi::c_void;

#[cfg(feature = "gpu")]
#[link(name = "sheaf_gpu")]
unsafe extern "C" {
    fn init_sheaf_gpu_vram(
        num_nodes: i32,
        num_edges: i32,
        stalk_dim: i32,
        channels: i32,
        h_row_ptr: *const i32,
        h_col_ind: *const i32,
        h_rho: *const f32,
        h_x_init: *const f32,
        out_handle: *mut *mut c_void,
        out_vram_bytes: *mut usize,
    ) -> i32;

    fn reset_sheaf_input(handle: *mut c_void, h_x_init: *const f32) -> i32;

    fn run_sheaf_conveyor_gpu_steps(
        handle: *mut c_void,
        num_layers: i32,
        tau: f32,
        out_time_ms: *mut f32,
        out_dirichlet_energy: *mut f32,
    ) -> i32;

    fn run_sheaf_conveyor_gpu_steps_precision(
        handle: *mut c_void,
        num_layers: i32,
        tau: f32,
        precision_mode: i32,
        out_time_ms: *mut f32,
        out_dirichlet_energy: *mut f32,
    ) -> i32;

    fn instantiate_sheaf_cuda_graph(
        handle: *mut c_void,
        num_layers: i32,
        tau: f32,
    ) -> i32;

    fn instantiate_sheaf_cuda_graph_precision(
        handle: *mut c_void,
        num_layers: i32,
        tau: f32,
        precision_mode: i32,
    ) -> i32;

    fn run_sheaf_cuda_graph(
        handle: *mut c_void,
        out_time_ms: *mut f32,
        out_dirichlet_energy: *mut f32,
    ) -> i32;

    fn read_sheaf_gpu_results(
        handle: *mut c_void,
        h_out_dest: *mut f32,
        num_layers: i32,
    ) -> i32;

    fn free_sheaf_gpu_vram(handle: *mut c_void) -> i32;
    fn get_gpu_vram_info(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SheafPrecision {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    FP8 = 3,
    FP4 = 4,
}

pub struct GpuSheafSession {
    handle: *mut c_void,
    pub allocated_vram_bytes: usize,
    pub total_elements: usize,
}

unsafe impl Send for GpuSheafSession {}

#[cfg(feature = "gpu")]
impl GpuSheafSession {
    pub fn init(
        num_nodes: usize,
        num_edges: usize,
        stalk_dim: usize,
        channels: usize,
        row_ptr: &[i32],
        col_ind: &[i32],
        rho_data: &[f32],
        x_init: &[f32],
    ) -> Result<Self, String> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut vram_bytes: usize = 0;
        let ret = unsafe {
            init_sheaf_gpu_vram(
                num_nodes as i32,
                num_edges as i32,
                stalk_dim as i32,
                channels as i32,
                row_ptr.as_ptr(),
                col_ind.as_ptr(),
                rho_data.as_ptr(),
                x_init.as_ptr(),
                &mut handle,
                &mut vram_bytes,
            )
        };
        if ret != 0 {
            return Err(format!("Failed to initialize GPU VRAM: code {}", ret));
        }
        Ok(Self {
            handle,
            allocated_vram_bytes: vram_bytes,
            total_elements: num_nodes * stalk_dim * channels,
        })
    }

    pub fn reset_input(&self, x_init: &[f32]) -> Result<(), String> {
        let ret = unsafe { reset_sheaf_input(self.handle, x_init.as_ptr()) };
        if ret != 0 {
            return Err("Failed to reset GPU input buffer".into());
        }
        Ok(())
    }

    pub fn run_steps(&self, num_layers: usize, tau: f32) -> Result<(f32, f32), String> {
        let mut time_ms = 0.0f32;
        let mut dirichlet = 0.0f32;
        let ret = unsafe {
            run_sheaf_conveyor_gpu_steps(
                self.handle,
                num_layers as i32,
                tau,
                &mut time_ms,
                &mut dirichlet,
            )
        };
        if ret != 0 {
            return Err(format!("GPU kernel execution failed: code {}", ret));
        }
        Ok((time_ms, dirichlet))
    }

    pub fn run_steps_precision(
        &self,
        num_layers: usize,
        tau: f32,
        precision: SheafPrecision,
    ) -> Result<(f32, f32), String> {
        let mut time_ms = 0.0f32;
        let mut dirichlet = 0.0f32;
        let ret = unsafe {
            run_sheaf_conveyor_gpu_steps_precision(
                self.handle,
                num_layers as i32,
                tau,
                precision as i32,
                &mut time_ms,
                &mut dirichlet,
            )
        };
        if ret != 0 {
            return Err(format!("GPU kernel precision execution failed: code {}", ret));
        }
        Ok((time_ms, dirichlet))
    }

    pub fn instantiate_cuda_graph(&self, num_layers: usize, tau: f32) -> Result<(), String> {
        self.instantiate_cuda_graph_precision(num_layers, tau, SheafPrecision::FP32)
    }

    pub fn instantiate_cuda_graph_precision(
        &self,
        num_layers: usize,
        tau: f32,
        precision: SheafPrecision,
    ) -> Result<(), String> {
        let ret = unsafe {
            instantiate_sheaf_cuda_graph_precision(self.handle, num_layers as i32, tau, precision as i32)
        };
        if ret != 0 {
            return Err(format!("CUDA Graph instantiation failed: code {}", ret));
        }
        Ok(())
    }

    pub fn run_cuda_graph(&self) -> Result<(f32, f32), String> {
        let mut time_ms = 0.0f32;
        let mut dirichlet = 0.0f32;
        let ret = unsafe {
            run_sheaf_cuda_graph(self.handle, &mut time_ms, &mut dirichlet)
        };
        if ret != 0 {
            return Err(format!("CUDA Graph launch failed: code {}", ret));
        }
        Ok((time_ms, dirichlet))
    }

    pub fn read_results(&self, num_layers: usize) -> Vec<f32> {
        let mut dest = vec![0.0f32; self.total_elements];
        unsafe {
            read_sheaf_gpu_results(self.handle, dest.as_mut_ptr(), num_layers as i32);
        }
        dest
    }

    pub fn get_vram_info() -> (usize, usize) {
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe {
            get_gpu_vram_info(&mut free, &mut total);
        }
        (free, total)
    }
}

#[cfg(not(feature = "gpu"))]
impl GpuSheafSession {
    pub fn init(
        _num_nodes: usize,
        _num_edges: usize,
        _stalk_dim: usize,
        _channels: usize,
        _row_ptr: &[i32],
        _col_ind: &[i32],
        _rho_data: &[f32],
        _x_init: &[f32],
    ) -> Result<Self, String> {
        Err("GPU backend is disabled. Recompile with --features gpu on an NVIDIA CUDA host.".into())
    }

    pub fn reset_input(&self, _x_init: &[f32]) -> Result<(), String> {
        Err("GPU backend is disabled.".into())
    }

    pub fn run_steps(&self, _num_layers: usize, _tau: f32) -> Result<(f32, f32), String> {
        Err("GPU backend is disabled.".into())
    }

    pub fn instantiate_cuda_graph(&self, _num_layers: usize, _tau: f32) -> Result<(), String> {
        Err("GPU backend is disabled.".into())
    }

    pub fn run_cuda_graph(&self) -> Result<(f32, f32), String> {
        Err("GPU backend is disabled.".into())
    }

    pub fn read_results(&self, _num_layers: usize) -> Vec<f32> {
        vec![]
    }

    pub fn get_vram_info() -> (usize, usize) {
        (0, 0)
    }
}

impl Drop for GpuSheafSession {
    fn drop(&mut self) {
        #[cfg(feature = "gpu")]
        {
            if !self.handle.is_null() {
                unsafe {
                    free_sheaf_gpu_vram(self.handle);
                }
            }
        }
    }
}
