## Sameh Algharabli -- CNG 514 Assignment 1 ##
import pandas
import matplotlib.pyplot as plot
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dataSet = pandas.read_csv("yourData", index_col='theUniqueIdOrAttributeOfYourData')

#----------------------------Mean------------------------#
print("\n##########---------- Mean ----------##########\n")
print("AnnualHouseholdIncome :    " + str(dataSet['AnnualHouseholdIncome'].mean()))
print("BirthYear2020  :    " + str(dataSet['BirthYear2020'].mean()))
print("CoronavirusIntent_Mask:    " + str(dataSet['CoronavirusIntent_Mask'].mean()))
print("---------------------------------------------")
#----------------------------------------------------------------------------------------#

#-----------------------------Median Calculations-----------------------#
print("\n##########---------- Median ----------##########\n")
print("AnnualHouseholdIncome :    " + str(dataSet['AnnualHouseholdIncome'].median()))
print("BirthYear2020  :    " + str(dataSet['BirthYear2020'].median()))
print("CoronavirusIntent_Mask:    " + str(dataSet['CoronavirusIntent_Mask'].median()))
print("---------------------------------------------")
#----------------------------------------------------------------------------------------#

#-----------------------------Mode Calculations-----------------------#
print("\n##########---------- Mode ----------##########\n")
print("AnnualHouseholdIncome:\n" + str(dataSet['AnnualHouseholdIncome'].mode()))
print("---------------------")
print("\nBirthYear2020:\n" + str(dataSet['BirthYear2020'].mode()))
print("---------------------")
print("\nCoronavirusIntent_Mask:\n" + str(dataSet['CoronavirusIntent_Mask'].mode()))
print("---------------------------------------------")
#----------------------------------------------------------------------------------------#

#-----------------------------Range Calculations-----------------------#
print("\n##########---------- Range ----------##########\n")
print("AnnualHouseholdIncome :" + str(max(dataSet['AnnualHouseholdIncome']) - min(dataSet['AnnualHouseholdIncome'])))
print("BirthYear2020  :" + str(max(dataSet['BirthYear2020']) - min(dataSet['BirthYear2020'])))
print("CoronavirusIntent_Mask:" + str(max(dataSet['CoronavirusIntent_Mask']) - min(dataSet['CoronavirusIntent_Mask'])))
print("---------------------------------------------")
#----------------------------------------------------------------------------------------#

#-----------------------------Quartiles Calculations-----------------------#
print("\n##########---------- Quartiles ----------##########\n")
print("AnnualHouseholdIncome:\n" + str(dataSet['AnnualHouseholdIncome'].quantile([0.25,0.5,0.75])))
print("---------------------")
print("\nBirthYear2020:\n" + str(dataSet['BirthYear2020'].quantile([0.25,0.5,0.75])))
print("---------------------")
print("\nCoronavirusIntent_Mask:\n" + str(dataSet['CoronavirusIntent_Mask'].quantile([0.25,0.5,0.75])))
print("---------------------------------------------")
#----------------------------------------------------------------------------------------#

#-----------------------------Variance Calculations-----------------------#
print("\n##########---------- Variance -------------##########\n")
print("AnnualHouseholdIncome :   " + str(dataSet['AnnualHouseholdIncome'].var()))
print("BirthYear2020  :   " + str(dataSet['BirthYear2020'].var()))
print("CoronavirusIntent_Mask:   " + str(dataSet['CoronavirusIntent_Mask'].var()))
print("---------------------------------------------\n\n")
#----------------------------------------------------------------------------------------#



print("-------------########## Boxplots/Histograms/Scatterplots -------------##########")

#-------------------AnnualHouseholdIncome plots---------------------------#
plot.figure(1)
dataSet.boxplot(column = ['AnnualHouseholdIncome'], grid = False)
plot.suptitle("AnnualHouseholdIncome Boxplot")

plot.figure(2)
AnnualHouseholdIncomeHist = dataSet['AnnualHouseholdIncome'].plot.hist(bins = 100)
plot.suptitle("AnnualHouseholdIncome histogram")

AnnualHouseholdIncomeScatter = dataSet['AnnualHouseholdIncome'].value_counts()
plot.figure(3)
plot.suptitle("AnnualHouseholdIncome scatterplot")
plot.scatter(AnnualHouseholdIncomeScatter.index, AnnualHouseholdIncomeScatter)
#-------------------------------------------------------#

#-------------------BirthYear2020 plots---------------------------#
plot.figure(4)
dataSet.boxplot(column = ['BirthYear2020'], grid = False)
plot.suptitle("BirthYear2020 boxplot")

plot.figure(5)
HasCoronavirusBeliefHist = dataSet['BirthYear2020'].plot.hist(bins = 100)
plot.suptitle("BirthYear2020 histogram")

HasCoronavirusBeliefScatter = dataSet['BirthYear2020'].value_counts()
plot.figure(6)
plot.suptitle("BirthYear2020 scatterplot")
plot.scatter(HasCoronavirusBeliefScatter.index, HasCoronavirusBeliefScatter)

#---------------------------------------------------#

#-------------------CoronavirusIntent_Mask plots---------------------------#
plot.figure(7)
dataSet.boxplot(column = ['CoronavirusIntent_Mask'], grid = False)
plot.suptitle("CoronavirusIntent_Mask boxplot")

plot.figure(8)
CoronavirusIntent_MaskHist = dataSet['CoronavirusIntent_Mask'].plot.hist(bins = 100)
plot.suptitle("CoronavirusIntent_Mask histogram")

CoronavirusIntent_MaskScatter = dataSet['CoronavirusIntent_Mask'].value_counts()
plot.figure(9)
plot.suptitle("CoronavirusIntent_Mask scatterplot")
plot.scatter(CoronavirusIntent_MaskScatter.index, CoronavirusIntent_MaskScatter)

plot.show()

#//--------------------------------------------------------------------------------------------------//#

print("##########------------- Missing Data Checking -------------##########")

missingData = dataSet['AnnualHouseholdIncome'].isnull().any()
print("Missing data in 'AnnualHouseholdIncome': ", end='')
print(missingData)

missingData = dataSet['BirthYear2020'].isnull().any()
print("Missing data in 'BirthYear2020': ", end='')
print(missingData)

missingData = dataSet['CoronavirusIntent_Mask'].isnull().any()
print("Missing data in 'CoronavirusIntent_Mask': ", end='')
print(missingData)

#-------------------------------------------------------------------------#

print("\n\n##########------------- Attributes Normalization -------------##########")

#Using Min-Max normalization formula
#scaler = MinMaxScaler()
#minMax_NormalizedData = scaler.fit_transform(dataSet)

minMax_NormalizedData = (dataSet - dataSet.min())/(dataSet.max() - dataSet.min())
print("Min-Max normalization: \n")
print("Normalized AnnualHouseholdIncome attribute using MinMax Normalization")
print(minMax_NormalizedData['AnnualHouseholdIncome'])

print("\n\nNormalized BirthYear2020 attribute using MinMax Normalization")
print(minMax_NormalizedData['BirthYear2020'])

print("\n\nNormalized CoronavirusIntent_Mask attribute using MinMax Normalization")
print(minMax_NormalizedData['CoronavirusIntent_Mask'])

#-------------------------------------------------------------------------#

#Z-score Normalization
zScore_NormalizedData = (dataSet - dataSet.mean())/dataSet.std()
print("---------------------------------------------------")
print("\n\nZ-Score normalization: \n --------------------------")
print("AnnualHouseholdIncome attribute normalized using Z-score")
print(zScore_NormalizedData['AnnualHouseholdIncome'])
print("-----------------------------")
print("\n\nBirthYear2020 attribute normalized using Z-score")
print(zScore_NormalizedData['BirthYear2020'])
print("-----------------------------")
print("\n\nCoronavirusIntent_Mask attribute normalized using Z-score")
print(zScore_NormalizedData['CoronavirusIntent_Mask'])
print("---------------------------------------------------")

#-----------------------------------------------------------------------------#
print("\n\n##########------------- Binning Data -------------##########")

print("Binning AnnualHouseholdIncome attribute into 10 bins: ")
dataSet['new_AnnualHouseholdIncome_bin'] = pandas.qcut(dataSet['AnnualHouseholdIncome'], q=10)
print(dataSet['new_AnnualHouseholdIncome_bin'])
print("-----------------------------------------------")

print("\nBinning BirthYear2020 attribute into 10 bins: ")
dataSet['new_BirthYear2020_bin'] = pandas.qcut(dataSet['BirthYear2020'], q=10)
print(dataSet['new_BirthYear2020_bin'])
print("-----------------------------------------------")

print("\nBinning CoronavirusIntent_Mask attribute into 3 bins: ")
dataSet['new_CoronavirusIntent_Mask_bin'] = pandas.qcut(dataSet['CoronavirusIntent_Mask'].drop_duplicates(), q=3)
#print(dataSet['new_CoronavirusIntent_Mask_bin'])
print("---------------------------------------------------------------------------------------")


#--------------------------------------------------------------------------------#

print("\n##########------------- Correlation -------------##########\n")

print("Correlation between AnnualHouseholdIncome and BirthYear2020: ", end='')
print(dataSet['AnnualHouseholdIncome'].corr(dataSet['BirthYear2020']))

print("\nCorrelation between AnnualHouseholdIncome and CoronavirusIntent_Mask: ", end='')
print(dataSet['AnnualHouseholdIncome'].corr(dataSet['CoronavirusIntent_Mask']))

print("\nCorrelation between BirthYear2020 and new_CoronavirusIntent_Mask: ", end='')
print(dataSet['BirthYear2020'].corr(dataSet['CoronavirusIntent_Mask']))

#-------------------------------------------------------------------------------#
