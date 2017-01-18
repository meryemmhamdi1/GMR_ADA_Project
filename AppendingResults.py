import pandas as pd

tweets_df1 = pd.read_csv('Results/Unannotated_Representation_part1.csv')
tweets_df2 = pd.read_csv('Results/Unannotated_Representation_part2.csv')

frames = [tweets_df1,tweets_df2]
whole_df = pd.concat(frames)
whole_df.to_csv('Results/Unannotated_Representation_whole.csv')
