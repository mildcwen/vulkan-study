
mod render;

use render::Vulkan;
fn main() -> Result<(), Box<dyn std::error::Error>>{

    let eventloop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&eventloop).unwrap();
    use winit::event::{Event, WindowEvent};
    use winit::platform::windows::WindowExtWindows;
    let render = Vulkan::new(window.hwnd(), window.hinstance())?;

    eventloop.run(move |event, _, controlflow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *controlflow = winit::event_loop::ControlFlow::Exit;
            },
            Event::MainEventsCleared => {
                
                window.request_redraw();

            },
            // Event::RedrawRequested(_) => {
            Event::RedrawEventsCleared => {
                //render here (later)
            render.draw_frame();
            },
            _ => {},
        }
    });
    Ok(())
}
