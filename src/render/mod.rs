
#[path= "common-func.rs"]
mod cf;

use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use ash::vk;
use ash::extensions::khr::{Win32Surface, Surface, Swapchain};

use thiserror::Error;
use anyhow::Result;

#[derive(Error, Debug,)]
pub enum ErrorUse{
    // failed to find GPUs with Vulkan support!
    #[error("找不到支持Vulkan的GPU！\n请检测是否安装Vulkan驱动")]
    NoDevice,
    // failed to find a suitable GPU!
    #[error("设备不兼容！\n无法支持相关操作")]
    DeviceNoSupport,
    #[error("Vulkan版本太低！")]
    OutdatedVersion,
   

}
pub struct Vulkan{

    instance: ash::Instance,
    debug: DebugUse,
    surface: SurfaceUse,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    transfer_queue: vk::Queue,
    swapchain: SwapchainUse,
    renderpass: vk::RenderPass,
    pipeline: Pipeline,
    graphics_command_pool: vk::CommandPool,
    graphics_command_buffers: Vec<vk::CommandBuffer>,
}
impl Vulkan{

    pub fn new(hwnd: vk::HWND, hinstance: vk::HINSTANCE) -> Result<Vulkan> {
        let entry = ash::Entry::new()?;
        let instance = cf::create_instance(&entry)?;
        let debug = DebugUse::new(&entry, &instance)?;
        let surface = SurfaceUse::new(&entry, &instance, hwnd, hinstance)?;        
        let (physical_device, indices) = cf::pick_physical_device(&instance, &surface)?;
        let device = cf::create_logical_device(&instance, physical_device, indices)?;
        let graphics_queue = unsafe {device.get_device_queue(indices.graphics.unwrap(), 0) };
        let transfer_queue = unsafe {device.get_device_queue(indices.transfer.unwrap(), 0) };
        
        let mut swapchain = SwapchainUse::new(&instance, &device, &surface, physical_device)?;
        let renderpass = Self::create_render_pass(&device, swapchain.image_format)?;
        swapchain.create_framebuffer(&device, renderpass)?;
        let pipeline = Pipeline::new(&device, swapchain.image_extent, renderpass)?;
        
        let graphics_command_pool = create_command_pool(&device, indices.graphics.unwrap())?;
        let graphics_command_buffers = create_command_buffers(&device, graphics_command_pool,
                     swapchain.image_count as u32, 
                     vk::CommandBufferLevel::PRIMARY,
        )?;
        

        //记录指令到command buffer.
        fill_command_buffers(&device, &graphics_command_buffers, &swapchain, renderpass, pipeline.pipeline)?;
         
                

       

        Ok(Vulkan{
            instance,
            debug,
            surface,
            physical_device,
            device,
            graphics_queue,
            transfer_queue,
            swapchain,
            renderpass,
            pipeline,
            graphics_command_pool,
            graphics_command_buffers,
        })

    }   
 
    pub fn draw_frame(&self){
     
        let index = self.swapchain.current_image.get();
        let (image_index, _) = unsafe{
            self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                std::u64::MAX, //等待时间,ns为单位
                self.swapchain.image_available[index],
                vk::Fence::null(),
            ).expect("image acquisition trouble")
        };
        unsafe{
            self.device.wait_for_fences( 
                 &[self.swapchain.may_begin_drawing[index]], 
                 true, 
                 std::u64::MAX
                )
                .expect("fence-waiting");
            self.device.reset_fences(&[self.swapchain.may_begin_drawing[index]])
                .expect("resetting fences");
        }
        let wait_semaphores = [self.swapchain.image_available[index] ];
        let semaphores_finished = [self.swapchain.rendering_finished[index] ];
        let waiting_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let commandbuffers = [self.graphics_command_buffers[image_index as usize] ];
        let submit_info = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&waiting_stages)
            .command_buffers(&commandbuffers)
            .signal_semaphores(&semaphores_finished)
            .build()];
        unsafe{
            self.device.queue_submit(self.graphics_queue, &submit_info, self.swapchain.may_begin_drawing[index])
                .expect("queue submission");
        }
        // 呈现...
        let swapchains = [self.swapchain.swapchain];
        let indices = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&semaphores_finished)
                .swapchains(&swapchains)
                .image_indices(&indices);

        unsafe{ 
            self.swapchain.swapchain_loader.queue_present(self.graphics_queue, &present_info)
                .expect("queue presentation");
        }
        
        self.swapchain.current_image.set(
            (index+1)%self.swapchain.image_count as usize
        );
    }

    fn create_render_pass(device: &impl DeviceV1_0, image_format: vk::Format)
    -> Result<vk::RenderPass, vk::Result>{
        let attachments = [vk::AttachmentDescription::builder()
            .format(image_format)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .samples(vk::SampleCountFlags::TYPE_1)
            .build()];
        let color_attachment_references = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let subpasses = [vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_references)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS).build()];
        let subpass_dependencies = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_subpass(0)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build()];
        let renderpass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&subpass_dependencies);
        let renderpass = unsafe { device.create_render_pass(&renderpass_info, None)? };
        
        Ok(renderpass)
    }    
}

impl Drop for Vulkan{
    fn drop(&mut self){
        unsafe {
            self.device.device_wait_idle().expect("something wrong while waiting");
            self.device.destroy_command_pool(self.graphics_command_pool, None);
            self.pipeline.cleanup(&self.device);
            self.device.destroy_render_pass(self.renderpass, None);
            // self.swapchain.cleanup(&self.device);
            // self.device.destroy_device(None);
            self.surface.loader.destroy_surface(self.surface.surface, None);
            // self.debug.cleanup();

            self.instance.destroy_instance(None); 
        }
    }
}
#[cfg(debug_assertions)]
unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    if message_severity.as_raw() >=  vk::DebugUtilsMessageSeverityFlagsEXT::WARNING.as_raw(){
        eprintln!("[Debug: ][{}][{}] {:?}", severity, ty, message);
    }
    vk::FALSE
}
#[cfg(debug_assertions)]
struct DebugUse{

    loader: ash::extensions::ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

#[cfg(debug_assertions)]
impl DebugUse {
    fn new(entry: &impl EntryV1_0, instance: &impl InstanceV1_0) -> Result<Self, vk::Result> { 
        let  debugcreateinfo = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                 vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                |vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));

        let loader = ash::extensions::ext::DebugUtils::new(entry, instance);
        let messenger = unsafe { loader.create_debug_utils_messenger(&debugcreateinfo, None)? };
        Ok(DebugUse { loader, messenger })

    }
    
    unsafe fn cleanup(&self){
        self.loader.destroy_debug_utils_messenger(self.messenger, None);
    }
}


struct SurfaceUse{
    surface: vk::SurfaceKHR,
    loader: Surface,
}
impl SurfaceUse{

    /// 创建win32的surface,传入HWND和HINSTANCE
    fn new<E: EntryV1_0, I: InstanceV1_0>(
        entry: &E,
        instance: &I,
        hwnd: vk::HWND,
        hinstance: vk::HINSTANCE
    )-> Result<Self, vk::Result> {

        let win32_surface_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
            .hwnd(hwnd)
            .hinstance(hinstance);
        let win32_loader = Win32Surface::new(entry, instance);
        let surface = unsafe { win32_loader.create_win32_surface(&win32_surface_create_info, None) }?;
        let loader = Surface::new(entry, instance);
        
        Ok(SurfaceUse{
            surface,
            loader,
        })
    }

 
    #[inline]
    fn get_physical_device_surface_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
    )-> Result<vk::SurfaceCapabilitiesKHR, vk::Result> {
        unsafe {
            self.loader
                .get_physical_device_surface_capabilities(physical_device, self.surface)
        }
    }
    #[inline]
    fn get_physical_device_surface_present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    )-> Result<Vec<vk::PresentModeKHR>, vk::Result> {
        unsafe {
            self.loader
                .get_physical_device_surface_present_modes(physical_device, self.surface)
        }
    }
    #[inline]
    fn get_physical_device_surface_formats(
        &self,
        physical_device: vk::PhysicalDevice,
    )-> Result<Vec<vk::SurfaceFormatKHR>, vk::Result> {
        unsafe {
            self.loader
                .get_physical_device_surface_formats(physical_device, self.surface)
        }
    }
    #[inline]
    fn get_physical_device_surface_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_index: u32,
    )-> Result<bool, vk::Result>{
        unsafe {
            self.loader.get_physical_device_surface_support(
                physical_device,
                queue_index,
                self.surface,
            )
        }
    }
}
use std::cell::Cell;
struct SwapchainUse{
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    image_format: vk::Format,
    image_extent: vk::Extent2D,
    image_count: usize,
    // images: Vec<vk::Image>,
    imageviews: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    current_image: Cell<usize>,
    image_available: Vec<vk::Semaphore>,
    rendering_finished: Vec<vk::Semaphore>,
    may_begin_drawing: Vec<vk::Fence>,
}
impl SwapchainUse {
    fn new<I: InstanceV1_0,  D: DeviceV1_0>(
        instance: &I,
        device: &D,
        surface: &SurfaceUse,
        physical_device: vk::PhysicalDevice,
    )-> Result<Self, vk::Result>{

        let surface_capabilities = surface.get_physical_device_surface_capabilities(physical_device)?;
        let present_mode = 
                Self::choose_swap_present_mode(surface.get_physical_device_surface_present_modes(physical_device)?);
        let surface_format = 
                Self::choose_swap_surface_format(surface.get_physical_device_surface_formats(physical_device)?);
        let image_format = surface_format.format;
        let image_extent = surface_capabilities.current_extent;
        let image_count = 3.max(surface_capabilities.min_image_count).min(surface_capabilities.max_image_count);
        
        let swap_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface.surface)

                .min_image_count(image_count)
                .image_format(image_format)
                .image_color_space(surface_format.color_space)
                .image_extent(image_extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                // .old_swapchain(vk::SwapchainKHR::null())
        ;       
        
        let swapchain_loader = Swapchain::new(instance, device);
        let swapchain = unsafe {
                swapchain_loader.create_swapchain(&swap_info, None )? 
        };

        let  images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let image_count = images.len();
        //Vulkan 的具体实现可能会比 minImageCount 多,显示确定图像数量.
        let mut imageviews = Vec::with_capacity(image_count);
        for &image in images.iter(){
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
                .format(image_format)
                .subresource_range(subresource_range)
                ;
            let imageview =
                unsafe { device.create_image_view(&imageview_create_info, None) }?;
            
            imageviews.push(imageview);
        }

        let  framebuffers: Vec<vk::Framebuffer> = Vec::with_capacity(image_count);

        let mut image_available = vec![];
        let mut rendering_finished = vec![];
        let mut may_begin_drawing = vec![];
        let semaphoreinfo = vk::SemaphoreCreateInfo::builder();
        let fenceinfo = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..image_count{
            let semaphore_available =
                unsafe {device.create_semaphore(&semaphoreinfo, None) }?;
            let semaphore_finished =
                unsafe {device.create_semaphore(&semaphoreinfo, None) }?;
            image_available.push(semaphore_available);
            rendering_finished.push(semaphore_finished);
            let fence = unsafe { device.create_fence(&fenceinfo, None) }?;
            may_begin_drawing.push(fence);
        }       
        let current_image = Cell::new(0_usize);

        Ok(SwapchainUse{
            swapchain_loader,
            swapchain,
            image_format,
            image_extent,
            image_count,    
            // images,
            imageviews,
            framebuffers,
            current_image,
            image_available,
            rendering_finished,
            may_begin_drawing,
            

        })
    }
    fn create_framebuffer(&mut self, device: &impl DeviceV1_0, renderpass: vk::RenderPass,)
    -> Result<(), vk::Result> {

        for &iv in &self.imageviews {
            let iview = [iv];
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass)
                .attachments(&iview)
                .width(self.image_extent.width)
                .height(self.image_extent.height)
                .layers(1);
            let fb = unsafe { device.create_framebuffer(&framebuffer_info, None) }?;
            self.framebuffers.push(fb);
        }
        Ok(())

    }
    unsafe fn cleanup(&self, device: &impl DeviceV1_0){

        for &fence in &self.may_begin_drawing {
            device.destroy_fence(fence, None);
        }
        for &semaphore in &self.image_available{
            device.destroy_semaphore(semaphore, None);

        }   
        for &semaphore in &self.rendering_finished{
            device.destroy_semaphore(semaphore, None);

        }
        for &fb in &self.framebuffers {
            device.destroy_framebuffer(fb, None);
        }
        for &iv in self.imageviews.iter() {
            device.destroy_image_view(iv, None);
        }
        self.swapchain_loader.destroy_swapchain(self.swapchain, None);

    }

    //选择颜色格式
    #[inline]
    fn choose_swap_surface_format(available: Vec<vk::SurfaceFormatKHR>)-> vk::SurfaceFormatKHR{

        if available.len() == 1 && available[0].format == vk::Format::UNDEFINED{
            return vk::SurfaceFormatKHR{
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
        }
        //直接选第一个
        available.first().unwrap().clone()
    }
    //选择呈现模式
    #[inline]
    fn choose_swap_present_mode(available: Vec<vk::PresentModeKHR>)-> vk::PresentModeKHR{
        for mode in available {
            if mode == vk::PresentModeKHR::MAILBOX { return mode;}
        }
        vk::PresentModeKHR::FIFO
    }


}


/// 很少需要一个以上队列,可以在多个线程创建指令缓冲，然后在主线程一次将它们全部提交，降低调用开销。
/// SAFETY: in the pick_phsical_device(),need is_complete() return true
///         so i unwrap() directly after that.
#[derive(Default, Clone, Copy)]
struct QueueFamilyIndices{
    //呈现(Presentation) 和 绘制 强制同一个queueFamilyIndex
    graphics: Option<u32>,    //vk::QueueFlags::GRAPHICS
    transfer: Option<u32>,    //vk::QueueFlags::TRANSFER
}
impl QueueFamilyIndices{
    #[inline]
    fn new()->Self{
        Default::default()
    }
    #[inline]
    fn is_complete(&self) -> bool {
         self.graphics.is_some() && self.transfer.is_some()
    }    
}


#[inline]
fn create_command_pool(device: &impl DeviceV1_0, family_index: u32)->Result<vk::CommandPool, vk::Result>{
    let pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(family_index)
        // .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
    ;
    unsafe { device.create_command_pool(&pool_info, None) }
}
#[inline]
fn create_command_buffers(device: &impl DeviceV1_0, pool: vk::CommandPool, count:u32, level: vk::CommandBufferLevel)
-> Result<Vec<vk::CommandBuffer>, vk::Result>{

    let commandbuf_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .command_buffer_count(count)
            .level(level)
    ;
    let commandbuffers = unsafe { device.allocate_command_buffers(&commandbuf_allocate_info)?};
    Ok(commandbuffers)
}

fn fill_command_buffers(
    device: &impl DeviceV1_0,
    commandbuffers: &[vk::CommandBuffer],
    swapchain: &SwapchainUse,
    renderpass: vk::RenderPass, 
    pipeline: vk::Pipeline,
)-> Result<(), vk::Result> {
    for (i, &commandbuffer) in commandbuffers.iter().enumerate() {
        let begininfo = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
        unsafe{ device.begin_command_buffer(commandbuffer, &begininfo)?; }
        let clearvalues = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [1.0, 1.0, 1.0, 1.0],
            },
        }];
        let renderpass_begininfo = vk::RenderPassBeginInfo::builder()
            .render_pass(renderpass)
            .framebuffer(swapchain.framebuffers[i])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.image_extent,
            })
            .clear_values(&clearvalues);
        unsafe{

            device.cmd_begin_render_pass(commandbuffer, &renderpass_begininfo, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(
                commandbuffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            device.cmd_draw(commandbuffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(commandbuffer);
            device.end_command_buffer(commandbuffer)?;
        }
    }
    Ok(())

}

struct Pipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}
impl Pipeline {
    fn new(
        device: &impl DeviceV1_0,
        extent: vk::Extent2D,
        renderpass: vk::RenderPass,
    )-> Result<Self>{   
       
     
        let vertexshader_createinfo = vk::ShaderModuleCreateInfo::builder().code(
            vk_shader_macros::include_glsl!("./shaders/shader.vert"),
        );
        let vertexshader_module = unsafe {device.create_shader_module(&vertexshader_createinfo, None)? };
        let fragment_createinfo = vk::ShaderModuleCreateInfo::builder().code(
            vk_shader_macros::include_glsl!("./shaders/shader.frag"),
        );
        let fragmentshader_module = unsafe {device.create_shader_module(&fragment_createinfo, None)? };
        let mainfunctionname = std::ffi::CString::new("main").unwrap();

        let vertexshader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertexshader_module)
            .name(&mainfunctionname);
        let fragmentshader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragmentshader_module)
            .name(&mainfunctionname);
        let shader_stages = vec![vertexshader_stage.build(), fragmentshader_stage.build()];
        //顶点输入
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder();
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
                // .topology(vk::PrimitiveTopology::POINT_LIST);
        let viewports = [vk::Viewport {
            x: 0.,
            y: 0.,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.,
            max_depth: 1.,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: extent,
        }];

        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::BACK)
            .polygon_mode(vk::PolygonMode::FILL);
        let multisampler_info = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let colourblend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .build()];
        let colourblend_info =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&colourblend_attachments);
        
        let pipelinelayout_info = vk::PipelineLayoutCreateInfo::builder();
        let pipelinelayout =
            unsafe { device.create_pipeline_layout(&pipelinelayout_info, None) }?;
        
        let pipeline_info = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampler_info)
            .color_blend_state(&colourblend_info)
            .layout(pipelinelayout)
            .render_pass(renderpass)
            .subpass(0)
            .build(),
        ];
        let pipeline = unsafe {            
                device.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &pipeline_info,
                    None,
                ).expect("A problem with the pipeline creation")
        }[0];

        unsafe {
            device.destroy_shader_module(fragmentshader_module, None);
            device.destroy_shader_module(vertexshader_module, None);
        }

        Ok(
            Pipeline{
                pipeline,
                layout: pipelinelayout,
            }
        )
        
    }
    fn cleanup(&self, device: &impl DeviceV1_0) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

