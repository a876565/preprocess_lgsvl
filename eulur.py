import math

rad2degree=180.0/math.pi
x,y,z,w=0.0,0.0,0.0,0.0
a,b,c=0.0,0.0,0.0

def to_eulur(x,y,z,w):
    a=rad2degree*math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    b=rad2degree*math.asin(2*(w*y-z*x))
    c=rad2degree*math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

    print(a,b,c)

to_eulur(0.0,0.0,-0.3,1.0)
to_eulur(0.0,0.0,-0.9,-0.3)
to_eulur(0.0,0.0,-0.5,-0.9)