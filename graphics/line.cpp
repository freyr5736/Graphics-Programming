#include "display_window.hpp"

int main() {
    init_window();

    // Wait for window to appear
    XEvent e;
    XMapWindow(display, window);
    XNextEvent(display, &e); // wait for MapNotify
    // create graphics content
    GC graphics_content = XCreateGC(display, window, 0, NULL);
    XSetForeground(display, graphics_content, WhitePixel(display, screen));

    // Draw a line: y = 0.4x + 200
    for (int x = 0; x < WIDTH; ++x) {
        float y = 0.4f * (float)x + 200.0f;
        if (y >= 0 && y < HEIGHT)
            XDrawPoint(display, window, graphics_content, x, (int)y);
    }

    XFlush(display);
    run_window();
    XFreeGC(display, graphics_content);
    destroy_window();

    return 0;
}