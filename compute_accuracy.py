
def compute_accuracy(input, data_num, class_num):
	correct_count = 0;

	for i in range(data_num):
		if input[i] == i/ (data_num/class_num):
			correct_count = correct_count + 1

	return float(correct_count) / data_num