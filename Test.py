def test_model(weight1, weight2, feature1_1, feature1_2, feature2_1, feature2_2, label1, label2, bias):
	correct = 0
	input_feature = list(zip(feature1_1, feature1_2, label1)) + list(zip(feature2_1, feature2_2, label2))
	for i in range(len(input_feature)):
		feature0 = float(input_feature[i][0])
		feature1 = float(input_feature[i][1])
		feature2 = float(input_feature[i][2])
		net = feature0 * weight1 + feature1 * weight2 + bias
		if net > 0:
			y = 1
		else:
			y = -1
		if y == feature2:
			correct += 1
	return correct/len(input_feature)