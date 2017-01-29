import fasttext


file_name = "/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/Results/fast_text.txt"
model_file_name = "/media/diskD/EPFL/Fall 2016/ADA/Project/GMR_ADA_Project/Models/fast_text_skip_gram"

# Train Skipgram model and save it in external file
model = fasttext.skipgram(file_name, model_file_name)

