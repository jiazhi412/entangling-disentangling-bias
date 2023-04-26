import utils

command_template = 'python main_Diabetes.py --bias_type {} --minority {} --minority_size {}'
p2 = ['I']
p3 = ['young', 'old']
p4 = [2, 20, 40, 100, 200]

utils.run(command_template, "flexible", 1, p2, p3, p4)