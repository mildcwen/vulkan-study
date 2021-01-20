//! 提供相关操作的调用接口

use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use ash::vk;
use ash::extensions::khr::Swapchain;
use anyhow::Result;

use super::{ErrorUse, SurfaceUse, QueueFamilyIndices};


pub(super) fn create_instance(
    entry: &ash::Entry,
)-> Result<ash::Instance, ash::InstanceError> {

    //need-change.
    let enginename = std::ffi::CString::new("first").unwrap();
    let appname = std::ffi::CString::new("first").unwrap();
    let enginev: u32 = vk::make_version(0, 0, 1);
    let appv: u32 = vk::make_version(0, 0, 1);    


    //获取API version, 获取失败 按v1.0算,,V1.0不支持apiVersion,没有.api_version()方法
    /* 
    let _apiV: u32 = match entry.try_enumerate_instance_version().unwrap_or(None) {
        // Vulkan 1.1+
        Some(version) => version,

        // Vulkan 1.0
        None => vk::make_version(1, 0, 0),
    };
    */
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&appname)
        .application_version(appv)
        .engine_name(&enginename)
        .engine_version(enginev)
        // .api_verion(apiV)
    ;
    //启用的扩展
    let extension_names: Vec<*const i8> = vec![
        ash::extensions::ext::DebugUtils::name().as_ptr(),
        ash::extensions::khr::Surface::name().as_ptr(),
        ash::extensions::khr::Win32Surface::name().as_ptr(),
    ];

  
    let layer_names =  vec![std::ffi::CString::new("VK_LAYER_KHRONOS_validation").unwrap().into_raw() as *const i8];
    let instance_create_info = vk::InstanceCreateInfo::builder()
        // .push_next(&mut debugcreateinfo)
        .application_info(&app_info)
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&layer_names)
    ;
    unsafe { entry.create_instance(&instance_create_info, None) }        
}

/// 选择可用的GPU, 此处实现为选择第一个可用的.
pub(super) fn pick_physical_device(instance: &impl InstanceV1_0, surface: &SurfaceUse)
-> Result<(vk::PhysicalDevice, QueueFamilyIndices)>{
    let phys_devs = unsafe { instance.enumerate_physical_devices()? };
    if phys_devs.is_empty(){
        return Err(From::from(ErrorUse::NoDevice));
    }
    let mut picked: Option<(vk::PhysicalDevice, QueueFamilyIndices)> = None;
    for device in phys_devs {
        
        let  (flag, indices) = is_device_suitable(instance, device, surface)?;
        if  flag{
            picked = Some((device, indices));
            break;
        }
        
    }
    match picked {
        Some(x) => Ok(x),
        None => Err(From::from(ErrorUse::DeviceNoSupport))
    }
}

/// 设备需求检测
fn is_device_suitable(
    instance: &impl InstanceV1_0, 
    device: vk::PhysicalDevice,
    surface: &SurfaceUse,
)-> Result<(bool, QueueFamilyIndices), vk::Result> {
    
    let properties = unsafe{instance.get_physical_device_properties(device)};
    // let features = unsafe{instance.get_physical_device_features(device)};

    let family_indices = find_queue_families(instance, device, surface)?;

    if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU &&
        family_indices.is_complete() &&
        //交换链需要至少支持一种图像格式和一种支持我们的窗口表面的呈现模式
        !surface.get_physical_device_surface_formats(device)?.is_empty() &&
        !surface.get_physical_device_surface_present_modes(device)?.is_empty()      
    {        
        Ok((true, family_indices))
    }else{
        Ok((false, family_indices))
    }
}

fn find_queue_families(
    instance: &impl InstanceV1_0, 
    physical_device: vk::PhysicalDevice,
    surface: &SurfaceUse,
)-> Result<QueueFamilyIndices, vk::Result>{

    let queue_family_properties =  unsafe { 
            instance.get_physical_device_queue_family_properties(physical_device) 
    };

   let mut find = QueueFamilyIndices::new();
   for (i, queue_family) in (0u32..).zip(queue_family_properties){
       
        if  queue_family.queue_count>0 &&
            queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) &&
            //此处强制要求 绘制和呈现同一队列
            surface.get_physical_device_surface_support(physical_device, i)?
        {
            find.graphics = Some(i);
        }else if find.transfer.is_none() && queue_family.queue_count>0 &&
            queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER)
        {
            find.transfer = Some(i);
        }

        if find.is_complete(){
            break;
        }
       
   }
   Ok(find)

}

/// 创建逻辑设备.
pub(super) fn create_logical_device<I: InstanceV1_0>(
    instance: &I,
    physical_device: vk::PhysicalDevice,
    indices: QueueFamilyIndices
)-> Result<I::Device, vk::Result>{

    let device_extension_name_pointers: Vec<*const i8> = vec![Swapchain::name().as_ptr()];

    let priority = vec![1.0f32];
    let queue_create_infos = [
        vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(indices.graphics.unwrap())
            .queue_priorities(&priority)
            .build(),
        
        vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(indices.transfer.unwrap())
            .queue_priorities(&priority)
            .build(),

    ];
   

    let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extension_name_pointers)
            // .enabled_features()
    ;

    unsafe { instance.create_device(physical_device, &device_create_info, None) }

}

/// 创建图像视图
pub(super) fn create_image_views(device: &impl DeviceV1_0, images: &[vk::Image], format: vk::Format)
->Result< Vec<vk::ImageView>, vk::Result>{

    let mut image_views = Vec::with_capacity(images.len());
    for &image in images{
        
        let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();
        let imageview_create_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range)
        ;
        let imageview =
            unsafe { device.create_image_view(&imageview_create_info, None) }?;
        
        image_views.push(imageview);
    }
    Ok(image_views)
}