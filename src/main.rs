use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}
use vulkano::device::{Device, DeviceCreateInfo, QueueFlags, QueueCreateInfo}

fn main() {
    let library = VulkanLibrary::new().expect("Vulkan library files was not found.")
    
    // Vulkan API Instance
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        }
    ).expect("Failed to create VkInstance")

    // Physical Devices supporting Vulkan
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");

    // Find a queue family on the physical device which supports graphics commands
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        }).expect("couldn't find a graphical queue family") as u32;
    
    // Create a Logical Device
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    ).expect("failed to create device");

    let queue = queues.next().unwrap();
}
