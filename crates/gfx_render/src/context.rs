//! Device acquisition.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no compatible graphics adapter found: {0}")]
    Adapter(#[from] wgpu::RequestAdapterError),
    #[error("failed to open the graphics device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
}

/// Backends to try, in priority order per platform:
/// Vulkan on Linux/Android (GL as fallback), DX12 then Vulkan on Windows,
/// Metal on Apple, WebGPU then WebGL in the browser.
///
/// `WGPU_BACKEND=gl` (etc.) overrides this at run time if the caller applies
/// [`wgpu::InstanceDescriptor::with_env`].
pub fn preferred_backends() -> wgpu::Backends {
    if cfg!(target_arch = "wasm32") {
        wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
    } else if cfg!(target_os = "windows") {
        wgpu::Backends::DX12 | wgpu::Backends::VULKAN
    } else if cfg!(target_vendor = "apple") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN | wgpu::Backends::GL
    }
}

/// The device/queue pair every renderer shares. Cheap to clone.
#[derive(Debug, Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Whether compute shaders (GPU cloth simulation) are available. False
    /// on WebGL2 and some OpenGL ES drivers.
    pub supports_compute: bool,
}

impl GpuContext {
    /// Picks the best adapter that can present to `surface` and opens a
    /// device on it. Async so the same code path works on the web.
    pub async fn request(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<(Self, wgpu::Adapter), GpuError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await?;
        let info = adapter.get_info();
        log::info!(
            "adapter: {} ({:?}, {:?})",
            info.name,
            info.backend,
            info.device_type
        );

        let supports_compute = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS);
        // Ask for the smallest limit set that covers what we use, at the
        // adapter's real texture resolution: the downlevel baseline where
        // compute exists, the WebGL2 one (render only) where it does not.
        let limits = if supports_compute {
            wgpu::Limits::downlevel_defaults()
        } else {
            wgpu::Limits::downlevel_webgl2_defaults()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("clo360 device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await?;
        Ok((
            Self {
                device,
                queue,
                supports_compute,
            },
            adapter,
        ))
    }
}
