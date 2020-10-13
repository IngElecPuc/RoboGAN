import win32api as api
import win32gui as gui
import win32con as con 
import time
import math
import random

xb, yb = 770, 420
tx = 10

def ruido():
    while True:
        xa, ya = xb, yb
        for i in range(1,1000):
            xa += random.randint(-7,7)
            ya += random.randint(-4,4)
            api.SetCursorPos((xa,ya))
            time.sleep(0.01)    

def click(x,y):
    api.SetCursorPos((x,y))
    api.mouse_event(con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    api.mouse_event(con.MOUSEEVENTF_LEFTUP,x,y,0,0)

while True:
    time.sleep(6)
    click(970, 500)
    time.sleep(6)
    click(1200, 180)
    #x, y = api.GetCursorPos()
    #print('X = ', x, ' Y = ', y)

'''
def moveMouse(x,y):
    api.SetCursorPos((x,y))

def move_slowly(x2,y2,total_time):
    x0, y0 = api.GetCursorPos()

    draw_steps = int(math.sqrt(math.pow(x0-x2,2) + math.pow(y0-y2,2)))

    dx = (x2-x0)/draw_steps #how much x to move each step
    dy = (y2-y0)/draw_steps #how much y to move each step
    dt = total_time/draw_steps #time between each step

    for n in range(draw_steps):
        x = int(x0+dx*n)
        y = int(y0+dy*n)
        moveMouse(x,y)
        time.sleep(dt)
'''