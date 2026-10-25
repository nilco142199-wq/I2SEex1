Math_command=input("Enter the math command you want to perform (add, subtract, multiply, divide): ")
if Math_command == "add":
    x=int(input("Enter the first number: "))
    y=int(input("Enter the second number: "))
    result = x + y
    print(f"The result of {x} + {y} is {result}")
elif Math_command == "subtract":
    x=int(input("Enter the first number: "))
    y=int(input("Enter the second number: "))
    result = x - y
    print(f"The result of {x} - {y} is {result}")
elif Math_command == "multiply":
    x=int(input("Enter the first number: "))
    y=int(input("Enter the second number: "))
    result = x * y
    print(f"The result of {x} * {y} is {result}")
elif Math_command == "divide":
    x=int(input("Enter the first number: "))
    y=int(input("Enter the second number: "))
    if y == 0:
        print("Error: Division by zero is not allowed.")
    else:
        result = x / y
        print(f"The result of {x} / {y} is {result}")
else:
    print("Invalid math command. Please enter add, subtract, multiply, or divide.")
    ## Notes##
    