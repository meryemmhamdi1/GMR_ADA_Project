import fasttext


file_name = "/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/Results/fast_text.txt"
model_file_name = "/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/Models/fast_text_cbow"

# Train cbow model and save it in external file
model = fasttext.cbow(file_name, model_file_name)
