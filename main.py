resting = int(input("Enter your resting heart rate.\nIf you aren't sure of your resting heart rate, you measure your pulse \nfor 60 seconds first thing in the morning before getting out of bed: "))
intensity = int(input("Enter the percent intensity you would like to train at. Enter integer numbers with no percent sign: "))
age = int(input("Enter your age as a integer number: "))



print(f"Your target heart rate is {(((220-age)-resting)*(intensity/100))+resting} beats per minute")
