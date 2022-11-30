from Train import signum, calculate_net
def test_model(weight1, weight2, feature1_1, feature1_2, feature2_1, feature2_2, label1, label2, bias):
	correct = 0
	input_feature = list(zip(feature1_1, feature1_2, label1)) + list(zip(feature2_1, feature2_2, label2))
	for i in range(len(input_feature)):
		net = calculate_net(weight1, weight2, bias, input_feature[i][0], input_feature[i][1])
		y = signum(net)
		if y == input_feature[i][2]:
			correct += 1
   
	return correct/len(input_feature)