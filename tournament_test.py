from tournament import Bracket, Team, get_row

#Simple example function
def lower_id_wins(t1,t2):
    if t1.id > t2.id:
        return t2
    return t1

#Example function using a row from a data_matrix
#t1 and t2 are team objects
#they have .name and .id member variables (and others that shouldn't be needed in these types of functions)
def higher_glicko_wins(t1,t2):
    row = get_row(t1,t2)#This returns the row from the data_matrix representing the game between teams t1 and t2
    #use it to predict just like you normally would
    if row.iloc[0].glicko_0 > row.iloc[0].glicko_1:
        return t1 if t1.id > t2.id else t2
    else:
        return t2 if t1.id > t2.id else t1

#make prediction function as shown in examples above
prediction_functions = {"Glicko Predictor": higher_glicko_wins}

#source: https://www.ncaa.com/news/basketball-men/bracket-beat/2017-01-10/march-madness-how-do-your-past-brackets-stack
#For 2018: ESPN Tournament Challenge app has a bracket getting a 60 as 50.5 percentile
average_scores = {2011: 53.12637, 2012: 82.98597, 2013: 69.97803, 2014: 60.14319, 2015: 83.25845, 2016: 68.17819, 2017: 65.66010}
mean = sum(average_scores)/len(average_scores)

print("Average score from 2011-2017: {}".format(mean))

test_mode = False#Set to true and add code below to test your functions


if not test_mode:#Warning, this takes a few minutes to run
    for name, prediction_function in prediction_functions.items():
        print("Using {}".format(name))
        #Need to specify path for calculation of team stats, results and glicko scores
        path = "./data_matrices/DataMatrices/1_seasons/"
        filename = '1_seasons_combined.csv'

        s = []
        for year in range(2003,2018):
            b = Bracket(path,filename,year)
            if year in average_scores:
                s.append(b.score_tournament(prediction_function))
        print("\t1 season average score: {}".format(sum(s)/len(s)))

        path = "./data_matrices/DataMatrices/2_seasons/"
        filename = '2_seasons_combined.csv'

        s = []
        for year in range(2003,2018):
            b = Bracket(path,filename,year)
            if year in average_scores:
                s.append(b.score_tournament(prediction_function))
        print("\t2 season average score: {}".format(sum(s)/len(s)))

        path = "./data_matrices/DataMatrices/3_seasons/"
        filename = '3_seasons_combined.csv'

        s = []
        for year in range(2003,2018):
            b = Bracket(path,filename,year)
            if year in average_scores:
                s.append(b.score_tournament(prediction_function))
        print("\t3 season average score: {}".format(sum(s)/len(s)))

        path = "./data_matrices/DataMatrices/4_seasons/"
        filename = '4_seasons_combined.csv'

        s = []
        for year in range(2003,2018):
            b = Bracket(path,filename,year)
            if year in average_scores:
                s.append(b.score_tournament(prediction_function))
        print("\t4 season average score: {}".format(sum(s)/len(s)))

        path = "./data_matrices/DataMatrices/old_glicko_1_seasons/"
        filename = 'old_glicko_1_seasons_combined.csv'

        s = []
        for year in range(2003,2018):
            b = Bracket(path,filename,year)
            if year in average_scores:
                s.append(b.score_tournament(prediction_function))
        print("\tOld glicko 1 season average score: {}".format(sum(s)/len(s)))
