use std::sync::Arc;
use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueFlags, QueueCreateInfo};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryTypeFilter};
use vulkano::swapchain::Surface;
use vulkano::device::DeviceExtensions;
use vulkano::device::Queue;
use vulkano::image::ImageUsage;
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};


fn select_physical_device(
    instance: &Arc<Instance>,
    event_loop: &EventLoop<()>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                // Find the first first queue family that is suitable.
                // If none is found, `None` is returned to `filter_map`,
                // which disqualifies this physical device.
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, event_loop).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,

            // Note that there exists `PhysicalDeviceType::Other`, however,
            // `PhysicalDeviceType` is a non-exhaustive enum. Thus, one should
            // match wildcard `_` to catch all unknown device types.
            _ => 4,
        })
        .expect("no device available")
}


struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

struct RenderContext {
    window: Arc<Window>,
    swapxhain: Arc<Swapchain>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().expect("Vulkan library files was not found.");

        // Vulkan API Instance
        let required_extensions = Surface::required_extensions(&event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        ).expect("Failed to create VkInstance");

        // Physical Devices supporting Vulkan
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = select_physical_device(
            &instance, 
            event_loop, 
            &device_extensions,
        );

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        // Create a Logical Device
        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        ).expect("failed to create device");

        // Command queue
        let queue = queues.next().unwrap();

        // Memory Allocation
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        App {
            instance,
            device,
            queue,
        }
    }
}

// winit requires implementing ApplicationHandler, and recommend the Window
// only to be created once resumed has been called.
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(Window::default_attributes()).unwrap());
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        // Swapchain
        let surface_capabilities = self.device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        let dimensions = window.inner_size();
        let composite_alpha = surface_capabilities.supported_composite_alpha.into_iter().next().unwrap();
        let image_format =  self.device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        let (swapchain, images) = Swapchain::new(
            self.device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count + 1, // How many buffers to use in the swapchain
                image_format,
                image_extent: window_size.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT, // What the images are going to be used for
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap();


    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                // Draw.

                // Queue a RedrawRequested event.
                //
                // You only need to call this if you've determined that you need to redraw in
                // applications which do not always need to. Applications that redraw continuously
                // can render here instead.
                //self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }    
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app);
}
