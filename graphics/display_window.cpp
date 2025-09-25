#include <X11/Xlib.h>
#include <err.h>
#include <stdio.h>

static Display *display;
static int screen;
static Window root;

#define POSX    500
#define POSY    500
#define WIDTH   500
#define HEIGHT  500
#define BORDER  25

static Window create_window(int x, int y, int w, int h, int b){
    Window window;
    XSetWindowAttributes attributes;

    attributes.background_pixel = BlackPixel(display,screen);
    attributes.border_pixel = WhitePixel(display,screen);
    attributes.event_mask = ButtonPress;

    // creating simple window
    window = XCreateWindow(display,root,x,y,w,h,b,DefaultDepth(display,screen), InputOutput, DefaultVisual(display,screen),CWBackPixel | CWBorderPixel | CWEventMask, &attributes);

    return window;
}

void run(){
    XEvent event;
    while(XNextEvent(display,&event)==0){
        switch(event.type){
            case ButtonPress : return;
        }
    }
}

int main(){
    Window window;
    display=XOpenDisplay(NULL);
    if(display==NULL){
        errx(1,"can't open display");
    }
    
    //get default screen and root window
    screen = DefaultScreen(display);
    root = RootWindow(display,screen);

    window = create_window(POSX,POSY,WIDTH,HEIGHT,BORDER);

    // map our window to display screen
    XMapWindow(display,window);

    run();

    // unmap our simple window
    XUnmapWindow(display,window);
    // freeing resources
    XDestroyWindow(display,window);
    // close connection to display
    XCloseDisplay(display);


    return 0;
}
