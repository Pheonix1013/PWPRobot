#imports all the necessary libraries
from PCA9685 import PCA9685

import time

import requests



#This is list of all of our commands
Dir = [

    'forward',

    'backward',

    'right',

    'left'	

]




#set motor speed to 75% of total possible power to motors
MOTOR_SPEED = 75



#api url
API_STATUS_URL = "http://192.168.240.8:5000/status" 



#connects motor hat to PCA9685
pwm = PCA9685(0x40, debug=False)
pwm.setPWMFreq(50)



class MotorDriver():
    """
    This class works with the motor driver, and makes the motors run based on what we pick
    
    Parameters:
    None
    
    Return: 
    None
    """
    def __init__(self):
		"""
		This is what creates all of the variables used for this class
        
        Parameters:
        self
        
        Return:
        None
        """
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2
        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4



    def MotorRun(self, motor_id, index, speed):
        """
		This is the function that sends the power to the motor
        
        Parameters:
        self, motor_id, index, speed
        
        Return:
        None
        """

        if speed > 100:

            return




        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, speed)



        if motor_id == 0:

            if index == 'forward':

                print("Left Motor: Forward")

                pwm.setLevel(self.AIN1, 0)

                pwm.setLevel(self.AIN2, 1)

            else:

                print("Left Motor: Backward")

                pwm.setLevel(self.AIN1, 1)

                pwm.setLevel(self.AIN2, 0)

        else:

            if index == 'forward':

                print("Right Motor: Forward")

                pwm.setLevel(self.BIN1, 1)

                pwm.setLevel(self.BIN2, 0)

            else:

                print("Right Motor: Backward")

                pwm.setLevel(self.BIN1, 0)

                pwm.setLevel(self.BIN2, 1)
        print("True")


    def MotorStop(self, motor_id):
		"""
		Stops the motor from moving
        
        Parameters:
        Self, motor_id
        
        Return:
        None        
        """

        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, 0)
Motor = MotorDriver()






def execute_command(command):
    """
    Connects to API and is constantly sending GET requests so that it knows when the next button is pressed.
    
    Parameters:
    command - which new command is being given
    
    Return:
    None
    """
    print(f"Executing new command: {command}")


    if command == 'forward':

        Motor.MotorRun(0, 'forward', MOTOR_SPEED)

        Motor.MotorRun(1, 'forward', MOTOR_SPEED)

    elif command == 'backward':

        Motor.MotorRun(0, 'backward', MOTOR_SPEED)

        Motor.MotorRun(1, 'backward', MOTOR_SPEED)

    elif command == 'left':


        Motor.MotorRun(0, 'backward', MOTOR_SPEED)

        Motor.MotorRun(1, 'forward', MOTOR_SPEED)

    elif command == 'right':


        Motor.MotorRun(0, 'forward', MOTOR_SPEED)

        Motor.MotorRun(1, 'backward', MOTOR_SPEED)

    elif command == 'stop':

        Motor.MotorStop(0)

        Motor.MotorStop(1)

    else:

        Motor.MotorStop(0)

        Motor.MotorStop(1)


#set state to stop so that it resets
current_state = 'stop'



print("Raspberry Pi Motor Client Starting. \nConnecting to API...")

while True:

    try:

        # Ask the API for status (send get request)

        response = requests.get(API_STATUS_URL, timeout=1)

        response.raise_for_status()

        

		#Store this new data into a variable
        data=response.json()
        print(f"JSON Data: {data}")

        new_state = "stop"


		#set new_state to new direction
        for direction in data:
                if data[direction] == True:

                        new_state = direction

		#Only change the motor function if the state is changed

        if new_state != current_state:

            print(f"Executing new command: {new_state}")

            execute_command(new_state)

            current_state = new_state

        
#Error handling------------------------
    except requests.exceptions.ConnectionError:



        if current_state != 'error':

            print(f"WARNING: Could not connect to API at {API_STATUS_URL}. Stopping motors.")

            execute_command('stop')

            current_state = 'error' # Prevent repeated warnings

    except requests.exceptions.Timeout:

        # Handle case where the API is slow to respond

        if current_state != 'timeout':

            print("WARNING: API request timed out. Stopping motors.")

            execute_command('stop')

            current_state = 'timeout'

    except Exception as e:

        # General error handling

        print(f"An unexpected error occurred: {e}. Stopping motors.")

        execute_command('stop')

        current_state = 'error'



#get the get request stuff every .1 seconds


    time.sleep(0.1)
