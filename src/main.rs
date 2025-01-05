use std::sync::Arc;
use vulkano::{
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    device::physical::{PhysicalDevice, PhysicalDeviceType},
    device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueFlags, QueueCreateInfo},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image},
    memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryTypeFilter},
    buffer::{Buffer, BufferCreateInfo, BufferUsage, BufferContents, Subbuffer},
    image::{Image, ImageUsage, view::ImageView},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexDefinition,
            vertex_input::Vertex as VkVertex,
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo, DynamicState,
    },
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents},
    command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    descriptor_set::{WriteDescriptorSet},
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::{self, GpuFuture},
    VulkanLibrary, VulkanError, Validated,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    render_context: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

#[derive(BufferContents, VkVertex)]
#[repr(C)]
struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}


fn select_physical_device(instance: &Arc<Instance>, event_loop: &EventLoop<()>, device_extensions: &DeviceExtensions) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                // Find the first first queue family that is suitable.
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, event_loop).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        // Prioritize dedicated GPUs
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("no device available")
}

fn get_framebuffers(images: &[Arc<Image>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().expect("Vulkan library files was not found.");

        // Vulkan API Instance
        let required_extensions = Surface::required_extensions(&event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY, // MacOS, iOS memes
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        ).expect("Failed to create VkInstance");

        // Physical Devices supporting Vulkan in the way we want
        let device_extensions = DeviceExtensions {khr_swapchain: true, ..DeviceExtensions::empty()};
        let (physical_device, queue_family_index) = select_physical_device(&instance, event_loop, &device_extensions);

        println!("Using device: {} (type: {:?})", physical_device.properties().device_name, physical_device.properties().device_type);

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

        // Command Buffer Allocator
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        App {
            instance,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            render_context: None
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
        ).unwrap();

        // Shaders
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450
                    layout(location = 0) in vec2 position;
                    void main() {
                        gl_Position = vec4(position, 0.0, 1.0);
                    }
                ",
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: r"
                    #version 450
                    layout(location = 0) out vec4 f_color;
                    void main() {
                        f_color = vec4(1.0, 0.0, 0.0, 1.0);
                    }
                ",
            }
        }

        // Render Pass
        let render_pass = vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        ).unwrap();

        // Framebuffers
        let framebuffers = get_framebuffers(&images, &render_pass);

        // Pipeline
        let pipeline = {
            let vs = vs::load(self.device.clone()).unwrap().entry_point("main").unwrap();
            let fs = fs::load(self.device.clone()).unwrap().entry_point("main").unwrap();

            let vertex_input_state = Vertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
                    .unwrap(),
            ).unwrap();

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()), // Triangles
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                }
            ).unwrap()
        };

        // Dynamic viewports allows recreating just the viewport instead of the entire
        // pipeline when the window is resized.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.render_context = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            recreate_swapchain: false,
            previous_frame_end,
        });
    }

    /*
    // Descriptor Set
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
        [],
    )
    .unwrap();
    */

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let render_context = self.render_context.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => render_context.recreate_swapchain = true,
            WindowEvent::RedrawRequested => {
                let window_size = render_context.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // Poll fences to determine what the GPU has processed and free resources
                // which are no longer needed.
                render_context.previous_frame_end.as_mut().unwrap().cleanup_finished();

                if render_context.recreate_swapchain {
                    // Recreate swapchain to use the new window size
                    let (new_swapchain, new_images) = render_context
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..render_context.swapchain.create_info()
                        }).expect("failed to recreate swapchain");
                    render_context.swapchain = new_swapchain;

                    // Framebuffers are dependent on the swapchain
                    render_context.framebuffers = get_framebuffers(&new_images, &render_context.render_pass);
                    render_context.viewport.extent = window_size.into();
                    render_context.recreate_swapchain = false;
                }

                // Get next image from swapcahin
                let (image_index, suboptimal, acquire_future) = match acquire_next_image(render_context.swapchain.clone(), None)
                .map_err(Validated::unwrap) {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        render_context.recreate_swapchain = true;
                        return;
                    },
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };
                
                if suboptimal {
                    render_context.recreate_swapchain = true;
                }


                // Triangle Data
                let vertices = [
                    Vertex { position: [-0.5, -0.25] },
                    Vertex { position: [0.0, 0.5] },
                    Vertex { position: [0.25, -0.1] },
                ];
                let vertex_buffer = Buffer::from_iter(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {usage: BufferUsage::VERTEX_BUFFER, ..Default::default()},
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    vertices,
                ).unwrap();

                // Build the command buffer
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                ).unwrap();

                // Commands
                builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 0.1, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(
                            render_context.framebuffers[image_index as usize].clone(),
                        )
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                ).unwrap()
                .set_viewport(0, [render_context.viewport.clone()].into_iter().collect()).unwrap()
                .bind_pipeline_graphics(render_context.pipeline.clone()).unwrap()
                .bind_vertex_buffers(0, vertex_buffer.clone()).unwrap();
                unsafe { builder.draw(vertex_buffer.len() as u32, 1, 0, 0).unwrap(); }
                builder.end_render_pass(Default::default()).unwrap();

                let command_buffer = builder.build().unwrap();

                let future = render_context
                .previous_frame_end.take().unwrap()
                .join(acquire_future).then_execute(self.queue.clone(), command_buffer).unwrap()
                .then_swapchain_present(
                    self.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(render_context.swapchain.clone(), image_index)
                )
                .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => render_context.previous_frame_end = Some(future.boxed()),
                    Err(VulkanError::OutOfDate) => {
                        render_context.recreate_swapchain = true;
                        render_context.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    },
                    Err(e) => panic!("failed to flush future: {e}"),
                }
            },
            _ => {},
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let render_context = self.render_context.as_mut().unwrap();
        render_context.window.request_redraw();
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app);
}
