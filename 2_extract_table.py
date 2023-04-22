import os
import pandas as pd
results = [os.path.join(dp, f) for dp, dn, filenames in os.walk("figures") for f in filenames if f == 'stats.txt']

'precision','recall','f_score','ord_error'

'precision','recall','f_score','ord_error'

table_7 = pd.DataFrame(columns=['Approach','Input','Graded Precision','Graded Recall','F-Score','OE'])
rows = ['SVM-RBF','SVM-L','RF','FFNN','CNN']

table_8 = pd.DataFrame(columns=['Approach','Input','Graded Precision','Graded Recall','F-Score','OE'])
rows = ['SVM-RBF','SVM-L','RF','FFNN','CNN']

table_9 = pd.DataFrame(columns=['Approach','Input','Graded Precision','Graded Recall','F-Score','OE'])
rows = ['SVM-L','CNN']

for result in results:
	print(result)
	with open(result) as file:
		is_first = True
		for line in file:
			if(is_first==True):
				is_first = False
				continue

			# print(result.split("_MODEL_CHOICE_"))
			# print(result.split("_MODEL_CHOICE_"))
			# exit()

			MODEL_CHOICE = result.split("_MODEL_CHOICE_")[1].split("_")[0]
			LABEL_CHOICE = result.split("_LABEL_CHOICE_")[1].split("_")[0]
			USE_CF = result.split("_USE_CF_")[1].split("_")[0]
			print("USE_CF")
			print(USE_CF)
			if(USE_CF=="0"):
				USE_FC = "I1 (Just TF)"
			else:
				USE_FC = "I2 (CF AND TF)"

			row = [MODEL_CHOICE,USE_FC]
			row_numbers = line.split()
			print(row)
			row_numbers = [str(round(float(i),2)) for i in row_numbers]
			row = row + row_numbers
			print(row_numbers)
			print("##########")
			print(row)
			print("##########")

			if(LABEL_CHOICE=="5"):
				table_7.loc[len(table_7)] = row
			elif(LABEL_CHOICE=="4"):
				table_8.loc[len(table_8)] = row
			elif(LABEL_CHOICE=="3+1"):
				table_9.loc[len(table_9)] = row
			else:
				print("This should not happen..")

table_7 = table_7.sort_values(by=['Approach','Input'], ascending=[False, True])
table_8 = table_8.sort_values(by=['Approach','Input'], ascending=[False, True])
table_9 = table_9.sort_values(by=['Approach','Input'], ascending=[False, True])

print(table_7)
print(table_8)
print(table_9)
table_7.to_csv("table_7.csv")
table_8.to_csv("table_8.csv")
table_9.to_csv("table_9.csv")