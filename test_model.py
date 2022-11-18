import cv2
from os import listdir
from tqdm import tqdm
import torch
from os import listdir
from test_model import test_model


def parse_file(filename):
	with open(filename) as f:
		lines = f.readlines()
	lines = [x.strip() for x in lines]
	lines = [x.split(',') for x in lines]
	lines = [[int(y) for y in x] for x in lines]
	return lines

path = 'dataset/eval/'
ann_path = 'dataset/eval/annotations/'
img_path = 'dataset/eval/images/'

file_names = listdir(ann_path)

def test_model(recog, print_res = False):

	corr_guesses = 0
	total_guesses = 0
	full_correct = 0
	col_right = 0
	shape_right = 0
	fill_right = 0
	num_right = 0

	for filename in (tqdm(file_names) if print_res else file_names):
		ann = parse_file(ann_path+filename)[0]
		res = torch.zeros(4,3)
		for i in range(4):
			res[i][ann[i]]=1
		
		image = cv2.imread(img_path+filename[:-4]+'.JPG')
		image = cv2.resize(image, (256,256))
		image = torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0)/255

		output = recog(image)
		scores = torch.max(output*res, dim=2).values.data[0]
		correct = torch.sum(scores>0.5)
		corr_guesses += correct
		if scores[0]>0.5:
			col_right += 1
		if scores[1]>0.5:
			shape_right += 1
		if scores[2]>0.5:
			fill_right += 1
		if scores[3]>0.5:
			num_right += 1
		total_guesses += 4
		if correct == 4:
			full_correct += 1

	if print_res:
		print("Accuracy: {:2.2f}%".format(corr_guesses/total_guesses*100))
		print("Full accuracy: {:2.2f}%".format(full_correct/len(file_names)*100))
		print("=====================================")
		print("Color accuracy: {:2.2f}%".format(col_right/len(file_names)*100))
		print("Shape accuracy: {:2.2f}%".format(shape_right/len(file_names)*100))
		print("Fill accuracy: {:2.2f}%".format(fill_right/len(file_names)*100))
		print("Number accuracy: {:2.2f}%".format(num_right/len(file_names)*100))

	return corr_guesses/total_guesses*100, full_correct/len(file_names)*100



def draw_accuracy_graphs(recog):
	xs = []
	ys1 = []
	ys2 = []

	for i in tqdm(range(1, 30)):
		xs.append(i)
		
		recog.load_state_dict(torch.load('weights/recognizer_{}.pth'.format(i*100-1)))
		recog.eval()
		classwise, total = test_model(recog, print_res=False)
		ys1.append(total)
		ys2.append(classwise)


	plt.plot(xs, ys1)
	plt.plot(xs, ys2)
	plt.show()
