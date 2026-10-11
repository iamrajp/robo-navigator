from gpiozero import PWMLED
from gpiozero import LED
import time

class Robot:
    def __init__(self, lm_pin1=6, lm_pin2=12, rm_pin1=13, rm_pin2=16, motor1_speed_pin =20, motor2_speed_pin = 21):

        self.lm_pin1 = LED(lm_pin1)
        self.lm_pin2 = LED(lm_pin2)
        self.rm_pin1 = LED(rm_pin1)
        self.rm_pin2 = LED(rm_pin2)
        self.motor1_speed_pin = PWMLED(motor1_speed_pin)
        self.motor2_speed_pin = PWMLED(motor2_speed_pin)
        self.stop()
        
    def forward(self, speed):
        
        self.motor1_speed_pin.value = speed
        self.motor2_speed_pin.value = speed
        self.lm_pin1.off()
        self.lm_pin2.on()
        self.rm_pin1.off()
        self.rm_pin2.on()
       
    def stop(self):

        self.lm_pin1.off()
        self.lm_pin2.off()
        self.rm_pin1.off()
        self.rm_pin2.off()
       
   

    def backward(self, speed):
        
    
        self.motor1_speed_pin.value = speed
        self.motor2_speed_pin.value = speed


        self.lm_pin1.on()
        self.lm_pin2.off()
        self.rm_pin1.on()
        self.rm_pin2.off()
     
      
    def left(self, speed):
        self.motor1_speed_pin.value = speed
        self.motor2_speed_pin.value = speed
 
        self.lm_pin1.off()
        self.lm_pin2.off()
        self.rm_pin1.off()
        self.rm_pin2.on()

       
    def right(self, speed):
        self.motor1_speed_pin.value = speed
        self.motor2_speed_pin.value = speed

        self.lm_pin1.off()
        self.lm_pin2.on()
        self.rm_pin1.off()
        self.rm_pin2.off()

       
#robo = Robot()
#robo.stop()                                                                                                                                                                                                                          
