#Interrupt handler for tracking encorder
#########################################

import sys
import RPi.GPIO as GPIO
import threading
from time import sleep

left_count = 0
right_count = 0
stop = False
GPIO.setmode(GPIO.BCM)

def interrupt_init():
    LEFT_ENCODER = 11
    RIGHT_ENCODER = 7
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LEFT_ENCODER, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(RIGHT_ENCODER, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(LEFT_ENCODER, GPIO.FALLING, 
        callback=left_tick)
    GPIO.add_event_detect(RIGHT_ENCODER, GPIO.FALLING, 
        callback=right_tick)

def left_tick(channel):
    global left_count
    left_count += 1

def right_tick(channel):
    global right_count
    right_count += 1

def read_encoder():
    f = open("motor.txt", "a")
    global left_count
    global right_count
    left = left_count
    right = right_count

    f.write('M {} {}\n'.format(left, right))
    if not stop:
        threading.Timer(0.5, read_encoder).start()
    else:
        print("quitting")
    #tick_sample.daemon = True
    #tick_sample.start()

def exit_encoder():
    print("Exiting Encoder")
    GPIO.cleanup()
    print("Done exiting Encoder")
    

        
