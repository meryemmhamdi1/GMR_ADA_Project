import pandas as pd

tweets_df1 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/PMIBased/Tweets_Labelled_PMI1.csv',encoding='utf-8')
tweets_df2 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/PMIBased/Tweets_Labelled_PMI2.csv',encoding='utf-8')
tweets_df3 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/PMIBased/Tweets_Labelled_PMI3.csv',encoding='utf-8')
tweets_df4 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/PMIBased/Tweets_Labelled_PMI4.csv',encoding='utf-8')
tweets_df5 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/PMIBased/Tweets_Labelled_PMI5.csv',encoding='utf-8')
tweets_df6 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/PMIBased/Tweets_Labelled_PMI6.csv',encoding='utf-8')
tweets_df7 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/PMIBased/Tweets_Labelled_PMI7.csv',encoding='utf-8')
tweets_df8 = pd.read_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/PMIBased/Tweets_Labelled_PMI8.csv',encoding='utf-8')

frames = [tweets_df1,tweets_df2,tweets_df3,tweets_df4,tweets_df5,tweets_df6,tweets_df7,tweets_df8]
whole_df = pd.concat(frames)
whole_df.to_csv('/home/meryem/Dropbox/meryem/algorithms/EmotionRecognition/FinalResults/PMIBased/Tweets_Labelled_PMI_Whole.csv',encoding='utf-8',index=False)
